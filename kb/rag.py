"""Generasi jawaban LLM (Groq via LangChain) berbasis chunk, dengan prompt terpusat, retry terbatas,
klasifikasi error, validasi sitasi terhadap sumber yang benar-benar dikirim, dan snapshot prompt final.
"""
from __future__ import annotations
import random
import re
import threading
import time
from dataclasses import dataclass, field

from .config import S
from .query_plan import analyze, stem, terms

MAX_CONTEXT_CHARS = S.llm_max_context_chars

SYSTEM_PROMPT = (
    "Anda adalah asisten knowledge base profesional Bappenas (ARCANA). Anda menjawab dalam bahasa Indonesia formal, "
    "jelas, dan langsung ke inti pertanyaan, HANYA berdasarkan sumber bernomor yang diberikan pada bagian KONTEKS.\n\n"
    "ATURAN:\n"
    "1. Pahami dulu apa yang ditanyakan, lalu pilih hanya sumber yang benar-benar membahasnya. Kata kunci yang sama "
    "tanpa isi yang relevan bukan bukti.\n"
    "2. Kalimat pertama menjawab inti pertanyaan. Setelah itu tambahkan rincian seperlunya (gunakan bullet bila ada "
    "beberapa poin), tanpa bertele-tele.\n"
    "2a. Jika pengguna meminta seluruh daftar atau poin (misalnya 'apa saja'), pertahankan semua butir sumber secara "
    "lengkap dan dalam urutan aslinya. Jangan mengganti butir sumber dengan ringkasan tema atau kesimpulan umum.\n"
    "2b. Untuk pertanyaan daftar, KONTEKS memuat kalimat pengantar daftar (butir bernomor 0) dan butir-butirnya. Tulis ulang \n"
    "kalimat pengantar secara ringkas, lalu semua butir bernomor sesuai urutan dan isi aslinya (boleh disajikan sebagai daftar \n"
    "bernomor). Jangan menghilangkan butir, jangan menggabungkan butir, dan jangan menambah butir yang tidak ada. Jangan \n"
    "menyimpulkan 'tidak ditemukan' selama ada sumber pada KONTEKS yang bagian/judulnya sesuai dengan yang ditanyakan.\n"
    "3. Pertahankan angka, tanggal, nama orang/organisasi, dan istilah persis seperti di sumber. Jangan menebak "
    "istilah yang tampak salah dengar pada transkrip; sebutkan apa adanya.\n"
    "3a. Jawab langsung pada substansi pertanyaan. Jangan mengawali atau mengulang jawaban dengan nama "
    "pembicara transkrip atau frasa seperti 'menurut [nama]', 'pembicara menyampaikan', atau 'X mengatakan'. "
    "Sebut nama hanya jika pertanyaan memang menanyakan identitas orang atau nama itu diperlukan untuk "
    "membedakan fakta dalam konteks.\n"
    "4. Cantumkan nomor sumber [n] di akhir kalimat terakhir dari setiap kelompok kalimat atau butir yang memuat fakta dari \n"
    "sumber tersebut, format [n], misalnya [1] atau [1][3]. JANGAN menaruh sitasi di akhir setiap kalimat: bila beberapa \n"
    "kalimat berurutan dalam satu paragraf berasal dari sumber yang sama, cukup satu sitasi di akhir paragraf itu. Pada daftar \n"
    "bernomor atau bullet, beri sitasi per butir hanya bila butir-butir itu berasal dari sumber yang berbeda; bila seluruh \n"
    "daftar berasal dari satu sumber, cukup satu sitasi di akhir daftar. Kalimat pengantar yang berakhir titik dua, kalimat \n"
    "transisi, dan kalimat penutup yang tidak memuat fakta tidak perlu sitasi. Gunakan HANYA kurung siku biasa, JANGAN \n"
    "gunakan kurung tebal/unicode seperti 【1】. Jangan membuat nomor, kutipan, halaman, atau judul sendiri.\n"
    "5. Jangan menggabungkan fakta dari sumber berbeda menjadi kesimpulan yang tidak dinyatakan oleh sumber. "
    "Jangan menambah fakta dari pengetahuan di luar KONTEKS.\n"
    "6. Jika sumber hanya menjawab sebagian pertanyaan, jawab bagian yang didukung lalu nyatakan jelas bagian yang "
    "belum ditemukan dalam sumber.\n"
    "7. HANYA jika setelah memeriksa SEMUA sumber tidak ada satu pun yang membahas hal yang ditanyakan, tulis tepat satu kalimat: "
    "\"Informasi tersebut belum ditemukan dalam sumber yang diperiksa.\" tanpa nomor sumber.\n"
    "8. Jika sumber saling bertentangan, tampilkan kedua versi beserta nomor sumbernya; jangan memilih salah satu."
)
# Aturan tambahan hasil audit (faithfulness). Dapat dimatikan (LLM_STRICT_GROUNDING=false) untuk ablasi yang bersih;
# prompt yang sama dipakai Graph RAG dan Vector RAG.
STRICT_RULE = (
    "\n9. Tulis hanya apa yang secara eksplisit tertulis pada sumber. Jangan menambah penjelasan, alasan, tujuan, atau "
    "kesimpulan pembuka yang tidak tertulis; lebih baik jawaban singkat yang seluruhnya didukung daripada lengkap tetapi "
    "memuat tafsir. Teks berlabel '(percakapan sebelumnya/berikutnya)' adalah konteks pendamping dari sumber [n] yang sama "
    "dan boleh dipakai sebagai bukti untuk sumber [n] tersebut."
)
if S.llm_strict_grounding:
    SYSTEM_PROMPT += STRICT_RULE
HUMAN_TEMPLATE = (
    "PERTANYAAN:\n{question}\n\n"
    "KONTEKS (nomor sumber valid: {valid_ids}):\n{context}\n\n"
    "Tulis jawaban sesuai aturan."
)

NOT_FOUND_RE = re.compile(
    r"(belum (dapat )?ditemukan|tidak (ditemukan|tersedia|disebutkan|terdapat)|tidak ada informasi|"
    r"tidak (memuat|membahas|menyebut))", re.I)
# UPDATE: Toleransi regex untuk membaca kurung biasa [1] maupun kurung tebal 【1】
CITE_RE = re.compile(r"[\[【](\d+)[\]】]")


# --------------------------------------------------------------------------------------
# Prompt terpusat
# --------------------------------------------------------------------------------------
@dataclass
class PromptPayload:
    question: str
    evidence: list[dict]          # chunk yang BENAR-BENAR ada di konteks final (n = nomor sumber di prompt)
    context_text: str
    system: str
    user: str
    valid_ids: list[int] = field(default_factory=list)

    def messages(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        return [SystemMessage(content=self.system), HumanMessage(content=self.user)]

    def snapshot(self) -> dict:
        """Salinan prompt final untuk log/debug (tanpa kredensial; tidak ada rekonstruksi terpisah)."""
        return {
            "model": S.groq_model, "question": self.question, "system_prompt": mask_secrets(self.system),
            "user_prompt": mask_secrets(self.user), "context_text": mask_secrets(self.context_text),
            "context_chars": len(self.context_text), "valid_source_numbers": list(self.valid_ids),
            "chunks": [{"n": e.get("n"), "chunk_id": e.get("chunk_id"), "doc_id": e.get("doc_id"),
                        "file_name": e.get("file_name"), "title": e.get("title"), "section": e.get("section"),
                        "retrieval_methods": e.get("retrieval_methods"), "retrieval_scores": e.get("retrieval_scores"),
                        "neighbor_chunk_ids": [x.get("chunk_id") for x in e.get("neighbors", [])],
                        "text_chars": len(e.get("text") or "")} for e in self.evidence],
        }


def mask_secrets(text: str) -> str:
    out = text or ""
    for secret in (S.groq_api_key, S.neo4j_password):
        if secret and len(secret) >= 8:
            out = out.replace(secret, "***")
    return re.sub(r"\bgsk_[A-Za-z0-9]{10,}\b", "gsk_***", out)


def _render_chunk(e: dict, n: int) -> str:
    head = (f'[{n}] dokumen: {e.get("title", "")}; bagian: {e.get("breadcrumb") or e.get("section") or "isi"}; '
            f'chunk: {e.get("chunk_id", "")}')
    parts = [head]
    if e.get("list_group_id") and e.get("list_size"):
        pos = e.get("list_item_pos") or 0
        parts.append("(kalimat pengantar daftar)" if pos == 0 else f"(butir {pos} dari {e['list_size']} pada daftar yang sama)")
    before = [x for x in e.get("neighbors", []) if x.get("position") == "before"]
    after = [x for x in e.get("neighbors", []) if x.get("position") == "after"]
    for x in before:
        parts.append(f'(paragraf sebelumnya, chunk {x["chunk_id"]}) {x["text"]}')
    parts.append(f'ISI SUMBER: {e.get("text", "")}')
    for x in after:
        parts.append(f'(paragraf berikutnya, chunk {x["chunk_id"]}) {x["text"]}')
    return "\n".join(parts)


def build_prompt(question: str, evidence: list[dict]) -> PromptPayload:
    """Satu-satunya tempat prompt dibentuk. Nomor [n] = urutan di konteks final, dipetakan ulang bila perlu."""
    sent = []
    for i, e in enumerate(evidence, 1):
        item = dict(e)
        item["orig_n"] = e.get("n", i)   # nomor sebelum penomoran ulang (untuk penelusuran)
        item["n"] = i                    # nomor yang tampil di prompt = nomor yang boleh disitasi model
        sent.append(item)
    context = "\n\n".join(_render_chunk(e, e["n"]) for e in sent)
    ids = [e["n"] for e in sent]
    user = HUMAN_TEMPLATE.format(question=question, valid_ids=", ".join(f"[{i}]" for i in ids), context=context)
    return PromptPayload(question, sent, context, SYSTEM_PROMPT, user, ids)


def context_has_answer(question: str, chosen: list[dict]) -> bool:
    """Apakah konteks final jelas memuat bagian yang ditanyakan? (dipakai untuk mencegah NOT_FOUND palsu)."""
    plan = analyze(question)
    words = {stem(w) for ph in plan.section_phrases for w in terms(ph)}
    if words:
        for e in chosen:
            head = {stem(t) for t in terms(f'{e.get("breadcrumb", "")} {e.get("section_path", "")} {e.get("section", "")}')}
            if len(words & head) / len(words) >= 0.6:
                return True
    qn = {stem(t) for t in (plan.doc_topic or plan.topic) if not t.isdigit()} | {stem(t) for t in plan.section_terms}
    if not qn:
        return False
    return any(len(qn & {stem(t) for t in terms(e.get("text", ""))}) / len(qn) >= 0.6 for e in chosen)


RECHECK_NOTE = ("\n\nCATATAN PEMERIKSAAN ULANG: KONTEKS di atas SUDAH memuat bagian dokumen yang sesuai dengan pertanyaan "
                "(lihat judul bagian/breadcrumb dan butir bernomor). Baca ulang seluruh sumber, lalu jawab berdasarkan isinya "
                "dengan sitasi [n]. Jangan menjawab 'belum ditemukan'.")


def available() -> bool:
    return bool(S.use_llm and S.groq_api_key)


def _is_exhaustive(question: str | None) -> bool:
    if not question:
        return False
    plan = analyze(question)
    return bool(plan.list_request or plan.section_phrases)


def select_context(evidence: list[dict], max_chars: int | None = None,
                   question: str | None = None, max_chunks: int = 8) -> list[dict]:
    """Pilih chunk untuk konteks (relevansi + anggaran karakter).

    Pertanyaan daftar/bagian ("apa saja", "identifikasi masalah"): tidak ada batas jumlah chunk, grup daftar
    (pengantar + semua butir) dipilih UTUH atau tidak sama sekali, dan hasil akhir diurutkan menurut urutan dokumen.
    Bila anggaran tidak cukup, grup/chunk berprioritas rendah yang dibuang, bukan butir terakhir daftar.
    """
    max_chars = max_chars or MAX_CONTEXT_CHARS
    ranked = list(evidence)
    exhaustive = _is_exhaustive(question)
    plan = analyze(question) if question else None
    qterms = {stem(t) for t in (plan.topic if plan else []) if not t.isdigit()}
    sec_stems = {stem(t) for t in (plan.section_terms if plan else [])}
    original = {id(item): rank for rank, item in enumerate(ranked)}

    def cov(item: dict) -> float:
        searchable = f'{item.get("title", "")} {item.get("section", "")} {item.get("breadcrumb", "")} {item.get("list_context", "")} {item.get("text", "")}'
        found = {stem(t) for t in terms(searchable)}
        base = len(qterms & found) / len(qterms) if qterms else 0.0
        sec = len(sec_stems & found) / len(sec_stems) if sec_stems else 0.0
        return max(base, sec)

    if question and len(ranked) > 1 and not exhaustive and qterms:
        scored = [(0.75 * cov(i) + 0.25 * (1.0 - original[id(i)] / max(len(ranked), 1)), i) for i in ranked]
        if any(sc > 0.05 for sc, _ in scored):
            ranked = [i for _, i in sorted(scored, key=lambda pair: pair[0], reverse=True)]
    if exhaustive:
        max_chunks = max(max_chunks, len(ranked))

    def cost(item: dict) -> int:
        return len(item.get("text", "")) + sum(len(x.get("text", "")) for x in item.get("neighbors", [])) + 240

    if not exhaustive:
        selected, used = [], 0
        for item in ranked[:max_chunks]:
            c = cost(item)
            if c > max_chars or (selected and used + c > max_chars):
                continue
            selected.append(item); used += c
        return selected

    # --- mode daftar: unit seleksi = grup daftar utuh atau chunk tunggal; prioritas = kecocokan bagian + peringkat retrieval ---
    units: dict[str, list[dict]] = {}
    for item in ranked:
        units.setdefault(item.get("list_group_id") or f"single:{item.get('chunk_id')}", []).append(item)
    def unit_priority(group: list[dict]) -> float:
        return max(0.7 * cov(i) + 0.3 * (1.0 - original[id(i)] / max(len(ranked), 1)) for i in group)
    selected, used = [], 0
    for group in sorted(units.values(), key=unit_priority, reverse=True):
        # neighbors untuk butir daftar tidak diperlukan (sudah ada di grup); hindari menggandakan konteks
        for item in group:
            if item.get("list_group_id"):
                item["neighbors"] = []
        c = sum(cost(i) for i in group)
        if c > max_chars:
            group = group[:1] if len(group) > 1 and group[0].get("list_item_pos", 0) != 0 else group
            c = sum(cost(i) for i in group)
            if c > max_chars:
                continue
        if selected and used + c > max_chars:
            continue
        selected += group; used += c
    selected.sort(key=lambda i: (i.get("doc_id") or "", i.get("paragraph_index") or 0, original[id(i)]))
    return selected


# --------------------------------------------------------------------------------------
# Klasifikasi error & pemanggilan Groq
# --------------------------------------------------------------------------------------
@dataclass
class LLMFailure(Exception):
    category: str            # timeout | rate_limit | context_too_large | server_error | bad_request | auth | connection | empty_output | unknown
    message: str
    status_code: int | None = None
    retry_after: float | None = None

    def __str__(self):
        return f"{self.category}: {self.message}"

    @property
    def eval_status(self) -> str:
        return {"timeout": "LLM_TIMEOUT", "rate_limit": "LLM_RATE_LIMIT"}.get(self.category, "LLM_ERROR")

    @property
    def retryable(self) -> bool:
        return self.category in {"timeout", "rate_limit", "server_error", "connection"}


def classify_exception(exc: Exception) -> LLMFailure:
    if isinstance(exc, LLMFailure):
        return exc
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) or {}
    retry_after = None
    try:
        ra = headers.get("retry-after") or headers.get("Retry-After")
        retry_after = float(ra) if ra else None
    except (TypeError, ValueError):
        retry_after = None
    msg = mask_secrets(str(exc))[:400]
    if retry_after is None:   # Groq menulis "Please try again in 7.5s" / "1m2.3s" pada pesan 429
        m = re.search(r"try again in (?:(\d+)m)?\s*([\d.]+)s", msg)
        if m:
            retry_after = int(m.group(1) or 0) * 60 + float(m.group(2))
    low = msg.lower()
    if "timeout" in name.lower() or "timed out" in low:
        return LLMFailure("timeout", msg, status, retry_after)
    if status == 429 or "ratelimit" in name.lower() or "rate limit" in low or "rate_limit" in low:
        return LLMFailure("rate_limit", msg, status or 429, retry_after)
    if status == 413 or "request too large" in low or "context length" in low or "maximum context" in low:
        return LLMFailure("context_too_large", msg, status or 413, retry_after)
    if status in (401, 403) or "authentication" in name.lower():
        return LLMFailure("auth", msg, status)
    if status and 500 <= status < 600:
        return LLMFailure("server_error", msg, status, retry_after)
    if status and 400 <= status < 500:
        return LLMFailure("bad_request", msg, status)
    if "connection" in name.lower() or "connect" in low:
        return LLMFailure("connection", msg, status)
    return LLMFailure("unknown", msg, status)


_llm_cache: dict = {}
_sem = threading.BoundedSemaphore(max(1, S.llm_concurrency))


def _make_llm(max_tokens: int):
    """ChatGroq dengan model & parameter yang sudah ada; retry ditangani modul ini (max_retries=0) agar terkontrol."""
    if max_tokens not in _llm_cache:
        from langchain_groq import ChatGroq
        _llm_cache[max_tokens] = ChatGroq(api_key=S.groq_api_key, model=S.groq_model, temperature=0.1,
                                          max_tokens=max_tokens, timeout=S.llm_timeout, max_retries=0)
    return _llm_cache[max_tokens]


def _invoke(llm, payload: PromptPayload, max_tokens: int) -> tuple[str, str | None]:
    client = llm if llm is not None else _make_llm(max_tokens)
    resp = client.invoke(payload.messages())
    text = resp if isinstance(resp, str) else (getattr(resp, "content", "") or "")
    finish = None if isinstance(resp, str) else (getattr(resp, "response_metadata", {}) or {}).get("finish_reason")
    return str(text).strip(), finish


def _sleep_for(attempt: int, failure: LLMFailure) -> float:
    wait = failure.retry_after if failure.retry_after is not None else S.llm_backoff_base * (2 ** attempt)
    return wait + random.uniform(0, 0.5 + 0.25 * wait)


def generate(question: str, evidence: list[dict], llm=None, sleep=time.sleep, max_context_chars: int | None = None) -> dict:
    """Satu generasi LLM. Mengembalikan dict hasil + snapshot prompt FINAL yang benar-benar dikirim.

    status: generated | failed | skipped_no_evidence | disabled
    """
    t0 = time.perf_counter()
    result: dict = {"status": "skipped_no_evidence", "model": S.groq_model, "attempts": [], "answer": None,
                    "error_category": None, "error_message": None, "http_status": None, "eval_status": None,
                    "prompt": None, "context_sent": [], "duration_ms": 0.0}
    chosen = select_context(evidence, question=question, max_chars=max_context_chars)
    if not chosen:
        return result
    if llm is None and not available():
        result.update(status="disabled", error_category="disabled", error_message="USE_LLM=false atau GROQ_API_KEY kosong")
        return result

    payload = build_prompt(question, chosen)
    max_tokens, reduced = S.llm_max_tokens, False
    failure: LLMFailure | None = None
    attempts = max(1, S.llm_max_attempts)
    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            with _sem:
                text, finish = _invoke(llm, payload, max_tokens)
            if not text:
                # Model penalaran dapat menghabiskan max_tokens sebelum menulis jawaban: naikkan batas sekali.
                if finish == "length" and max_tokens < 4096 and llm is None:
                    result["attempts"].append({"attempt": attempt + 1, "ok": False, "category": "empty_output",
                                               "finish_reason": finish, "max_tokens": max_tokens,
                                               "ms": round((time.perf_counter() - started) * 1000, 1)})
                    max_tokens = min(max_tokens * 2, 4096)
                    continue
                raise LLMFailure("empty_output", f"model mengembalikan keluaran kosong (finish_reason={finish})")
            result["attempts"].append({"attempt": attempt + 1, "ok": True, "finish_reason": finish,
                                       "ms": round((time.perf_counter() - started) * 1000, 1)})
            # Guard NOT_FOUND palsu: konteks memuat bagian yang ditanyakan tetapi model menolak -> periksa ulang sekali.
            if (NOT_FOUND_RE.search(text) and not CITE_RE.search(text) and not result.get("recheck")
                    and context_has_answer(question, chosen)):
                result["recheck"] = True
                payload = PromptPayload(payload.question, payload.evidence, payload.context_text,
                                        payload.system, payload.user + RECHECK_NOTE, payload.valid_ids)
                result["attempts"][-1]["ok"] = False
                result["attempts"][-1]["category"] = "false_not_found_recheck"
                continue
            result.update(status="generated", answer=text, error_category=None)
            failure = None
            break
        except Exception as exc:  # noqa: BLE001
            failure = classify_exception(exc)
            result["attempts"].append({"attempt": attempt + 1, "ok": False, "category": failure.category,
                                       "http_status": failure.status_code, "retry_after": failure.retry_after,
                                       "ms": round((time.perf_counter() - started) * 1000, 1)})
            if failure.category == "context_too_large" and not reduced and len(chosen) > 1:
                chosen = chosen[: max(1, len(chosen) // 2)]       # kurangi berdasarkan peringkat relevansi
                payload, reduced = build_prompt(question, chosen), True
                continue
            if not failure.retryable or attempt == attempts - 1:
                break
            wait = _sleep_for(attempt, failure)
            if failure.retry_after is not None and failure.retry_after > S.llm_max_retry_wait:
                break    # batas harian/menit panjang: gagal cepat, jangan menahan request
            sleep(min(wait, S.llm_max_retry_wait + 1))
    if result["status"] != "generated" and failure is None:
        failure = LLMFailure("empty_output", "keluaran kosong setelah seluruh percobaan")
    result["prompt"] = payload.snapshot()
    result["context_sent"] = payload.evidence
    result["duration_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    if failure is not None:
        result.update(status="failed", error_category=failure.category, error_message=failure.message,
                      http_status=failure.status_code, eval_status=failure.eval_status)
    return result


# --------------------------------------------------------------------------------------
# Validasi jawaban & sitasi
# --------------------------------------------------------------------------------------
def _segments(answer: str) -> list[str]:
    text = re.sub(r"[*_`#]+", "", answer or "")
    out = []
    for line in text.splitlines():
        line = line.strip(" -•\t")
        if not line:
            continue
        # kalimat = teks sampai tanda akhir (diikuti spasi/akhir baris) + sitasi yang menyusul; desimal "1.5" tidak terpecah
        out += [m.group(0).strip() for m in re.finditer(r"\S.*?(?:[.!?](?=\s|$)(?:\s*(?:\[\d+\])+)*|$)", line) if m.group(0).strip()]
    return out


def factual_segments(answer: str) -> list[str]:
    segs = []
    for s in _segments(answer):
        body = CITE_RE.sub("", s).strip()
        if body.endswith(":") or len(body.split()) < 3 or NOT_FOUND_RE.search(body):
            continue
        segs.append(s)
    return segs


def _blocks(answer: str) -> list[list[str]]:
    """Pecah jawaban menjadi blok (dipisah baris kosong; daftar bernomor/bullet satu blok); tiap blok berisi segmen kalimat."""
    blocks: list[list[str]] = []
    cur: list[str] = []
    for line in (answer or "").splitlines():
        if not line.strip():
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)
    return [segs for segs in ([s for s in _segments("\n".join(b))] for b in blocks) if segs]


def cited_facts(answer: str) -> list[tuple[str, list[int]]]:
    """Kalimat faktual beserta nomor sumbernya. Sitasi pada akhir kelompok kalimat berlaku untuk kalimat-kalimat
    bersitasi-kosong yang mendahuluinya dalam blok yang sama (paragraf atau daftar). Kalimat tanpa sitasi di AKHIR blok
    (tidak ada sitasi sesudahnya) tetap dianggap tidak bersitasi (ids kosong)."""
    out: list[tuple[str, list[int]]] = []
    for segs in _blocks(answer):
        inherited: list[int] = []
        block: list[tuple[str, list[int]]] = []
        for s in reversed(segs):
            body = CITE_RE.sub("", s).strip()
            if body.endswith(":") or len(body.split()) < 3 or NOT_FOUND_RE.search(body):
                continue                              # bukan klaim faktual
            own = [int(m) for m in CITE_RE.findall(s)]
            if own:
                inherited = own
            block.append((s, own if own else list(inherited)))
        out += list(reversed(block))
    return out


def uncited_segments(answer: str) -> list[str]:
    return [s for s, ids in cited_facts(answer) if not ids]


def valid(out: str, evidence: list[dict]) -> bool:
    """Sitasi sah bila semua nomor dikenal dan tiap kelompok klaim faktual bersitasi (sitasi di akhir paragraf/butir
    menutupi kalimat-kalimat sebelumnya dalam blok itu); pernyataan 'belum ditemukan' boleh tanpa sitasi."""
    if not out or len(out.strip()) < 10:
        return False
    known = {int(e.get("n", i)) for i, e in enumerate(evidence, 1)}
    marks = [int(m) for m in CITE_RE.findall(out)]
    if any(m not in known for m in marks):
        return False
    return not uncited_segments(out)


_NUM_RE = re.compile(r"\d+(?:[.,:]\d+)*")


def _support(segment: str, chunk_texts: list[str]) -> bool:
    """Heuristik leksikal: istilah isi & angka pada kalimat harus muncul pada chunk yang dikutip."""
    body = CITE_RE.sub("", segment)
    pool_terms = {stem(t) for t in terms(" ".join(chunk_texts))}
    pool_text = " ".join(chunk_texts)
    seg_terms = {stem(t) for t in terms(body) if not t.isdigit()}
    nums = {n for n in _NUM_RE.findall(body)}
    if any(n not in pool_text for n in nums):
        return False
    if not seg_terms:
        return True
    return len(seg_terms & pool_terms) / len(seg_terms) >= 0.25


def validate_answer(answer: str, sent: list[dict]) -> dict:
    """Validasi menyeluruh terhadap chunk yang benar-benar dikirim. `sent` memakai nomor n dari prompt final."""
    by_n = {int(e["n"]): e for e in sent}
    all_marks = [int(m) for m in CITE_RE.findall(answer or "")]
    invalid = sorted({m for m in all_marks if m not in by_n})
    cited = list(dict.fromkeys(m for m in all_marks if m in by_n))
    facts_ids = cited_facts(answer)                  # sitasi di akhir blok berlaku untuk kalimat sebelumnya
    facts = [s for s, _ in facts_ids]
    uncited = [s for s, ids in facts_ids if not ids]
    supported = unsupported = 0
    unsupported_segments = []
    for s, raw_ids in facts_ids:
        ids = [i for i in raw_ids if i in by_n]
        if not ids:
            continue
        texts = []
        for i in ids:
            texts.append(by_n[i].get("text", ""))
            texts += [x.get("text", "") for x in by_n[i].get("neighbors", [])]
        if _support(s, texts):
            supported += 1
        else:
            unsupported += 1
            unsupported_segments.append(s[:200])
    judged = supported + unsupported
    not_found_stmt = bool(NOT_FOUND_RE.search(answer or ""))
    return {
        "cited_numbers": cited, "invalid_markers": invalid, "factual_claims": len(facts),
        "uncited_claims": len(uncited), "supported_claims": supported, "unsupported_claims": unsupported,
        "support_ratio": round(supported / judged, 3) if judged else None,
        "unsupported_segments": unsupported_segments, "not_found_statement": not_found_stmt,
        "semantic_support": "lexical_heuristic", "groundedness": "lexical_heuristic",
    }


def eval_status_for(answer: str | None, v: dict) -> str:
    """Status akhir: hanya dipanggil bila generasi berhasil."""
    if v["invalid_markers"]:
        return "INVALID_CITATION"
    if v["not_found_statement"] and not v["cited_numbers"]:
        return "NOT_FOUND"
    if not v["cited_numbers"]:
        return "INVALID_CITATION"          # klaim tanpa satu pun sitasi
    if v["support_ratio"] is not None and v["support_ratio"] < 0.34:
        return "UNSUPPORTED_ANSWER"
    if v["not_found_statement"]:
        return "PARTIAL"
    return "ANSWERED"


def finalize_citations(answer: str, sent: list[dict]) -> tuple[str, list[dict], dict]:
    """Buang nomor tak dikenal, buat daftar sitasi hanya dari sumber yang benar-benar dikutip (metadata dari retrieval),
    dan renumber berurutan. Tidak pernah menambahkan sitasi yang tidak ditulis model."""
    by_n = {int(e["n"]): e for e in sent}
    cited = list(dict.fromkeys(int(m) for m in CITE_RE.findall(answer) if int(m) in by_n))
    mapping = {old: new for new, old in enumerate(cited, 1)}
    text = CITE_RE.sub(lambda m: f"[{mapping[int(m.group(1))]}]" if int(m.group(1)) in mapping else "", answer)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    cites = []
    for old in cited:
        e = by_n[old]
        cites.append({
            "source_id": f"S{mapping[old]}", "n": mapping[old], "prompt_n": old,
            "doc_id": e.get("doc_id"), "title": e.get("title"), "chunk_id": e.get("chunk_id"),
            "chunk_ids": [e.get("chunk_id")] + [x["chunk_id"] for x in e.get("neighbors", [])],
            "breadcrumb": e.get("breadcrumb"), "page": e.get("page"), "url": e.get("url"),
            "file_name": e.get("file_name"), "section": e.get("section"),
            "section_title": e.get("section_title"), "section_path": e.get("section_path"),
            "paragraph_index": e.get("paragraph_index"), "source_reference": e.get("source_reference"),
            "snippet": (e.get("text") or "")[:300], "score": e.get("score"),
            "retrieval_methods": e.get("retrieval_methods", []), "retrieval_scores": e.get("retrieval_scores", {}),
            "graph_paths": e.get("graph_paths", []),
        })
    return text, cites, {str(k): v for k, v in mapping.items()}


def assess_confidence(question: str, answer: str, evidence: list[dict]) -> tuple[float | None, str]:
    """Tidak ada skor karangan: confidence diturunkan dari rasio dukungan leksikal (heuristik, belum terkalibrasi)."""
    if not answer or not evidence:
        return None, "no_evidence"
    v = validate_answer(answer, evidence)
    if v["support_ratio"] is None:
        return None, "not_assessable"
    return v["support_ratio"], "lexical_support_ratio_uncalibrated"