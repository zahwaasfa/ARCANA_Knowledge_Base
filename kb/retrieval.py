"""Orkestrasi retrieval GraphRAG ARCANA: pertanyaan -> rencana -> dokumen -> bagian graf -> (hibrida | ekspansi) -> jawaban terstruktur + sitasi."""
from __future__ import annotations
import re
import time
from . import search
from .config import S
from .chunker import lead
from .query_plan import OVERVIEW_SECTIONS, QueryPlan, analyze, fold, stem, terms
from .text import overlap

SECTION_LABEL = {
    "identitas": "Identitas dokumen", 
    "deskripsi": "Deskripsi", 
    "manfaat": "Manfaat", 
    "proses_bisnis": "Proses bisnis penyusunan",
    "regulasi": "Dasar hukum", 
    "data_sekunder": "Sumber data / rujukan", 
    "mitra": "Mitra dan kolaborasi", 
    "tag": "Kata kunci"
}

SEED_CAT_TO_INTENT = {
    "deskripsi": "deskripsi", 
    "manfaat": "manfaat", 
    "identitas": "identitas", 
    "proses": "proses", 
    "regulasi": "regulasi",
    "sumber": "sumber_data", 
    "mitra": "mitra", 
    "tag": "tag"
}


def strip_prefix(text: str, title: str) -> str:
    """Buang awalan template ('Manfaat dokumen "<judul>": ...') yang ditambahkan saat ingest; sisakan isinya."""
    t = " ".join((text or "").split())
    k = t.find(f'"{title}"')
    if 0 <= k <= 80:
        m = re.search(r":\s+", t[k + len(title) + 2:k + len(title) + 2 + 40])
        if m:
            return t[k + len(title) + 2 + m.end():].strip()
    return t


def sentences(text: str, min_len: int = 25) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", " ".join((text or "").split())) if p.strip()]
    out: list[str] = []
    for p in parts:
        if out and len(p) < min_len:
            out[-1] += " " + p
        else:
            out.append(p)
    return out


def best_passage(text: str, topic: list[str], n: int = 320) -> str:
    """Ambil kalimat (+ tetangganya) yang paling banyak memuat istilah pertanyaan."""
    ss = sentences(text, 1)
    if len(ss) <= 1 or not topic:
        return lead(text, n)
    qs = {stem(t) for t in topic if not t.isdigit()}
    sc = [sum(stem(w) in qs for w in terms(x)) for x in ss]
    i = max(range(len(ss)), key=lambda k: (sc[k], -k))
    out = ss[i]
    for j in (i + 1, i - 1):
        if len(out) < n * 0.6 and 0 <= j < len(ss):
            out = f"{out} {ss[j]}" if j > i else f"{ss[j]} {out}"
    return lead(out, n)


def bullets(items: list[str], cap: int | None = None, more: str = "lainnya") -> str:
    items = [i for i in items if i]
    shown = items[:cap] if cap else items
    s = "\n".join(f"- {i}" for i in shown)
    if cap and len(items) > cap:
        s += f"\n- _…dan {len(items) - cap} {more}_"
    return s


def _named_anchors(question: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b[A-Z][A-Z0-9]{1,}\b", question)))


def _anchor_forms(anchor: str) -> tuple[str, ...]:
    aliases = {
        "AI": ("AI", "artificial intelligence", "kecerdasan artifisial", "kecerdasan buatan"),
        "RAG": ("RAG", "retrieval augmented generation"),
        "UGM": ("UGM", "Universitas Gadjah Mada"),
        "NUS": ("NUS", "National University of Singapore"),
        "RUU": ("RUU", "Rancangan Undang-Undang"),
        "UU": ("UU", "Undang-Undang"),
        "NA": ("NA", "Naskah Akademik"),
    }
    return aliases.get(anchor.upper(), (anchor,))


def _filter_named_anchors(question: str, hits: list[dict]) -> list[dict]:
    anchors = _named_anchors(question)
    if not anchors or not hits:
        return hits
    out = []
    for hit in hits:
        haystack = " ".join(str(hit.get(key) or "") for key in ("title", "file_name", "breadcrumb", "section", "description", "text")).casefold()
        def contains(form: str) -> bool:
            if len(form) <= 4 and form.upper() == form:
                return bool(re.search(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])", haystack, re.I))
            return fold(form) in fold(haystack)
        if any(contains(form) for anchor in anchors for form in _anchor_forms(anchor)):
            out.append(hit)
    return out if out else hits


class CiteBook:
    """Penomoran sitasi [n] yang konsisten antara teks jawaban dan daftar sitasi."""

    def __init__(self):
        self.items: list[dict] = []
        self._idx: dict[str, int] = {}

    def add(self, ev: dict, snippet: str | None = None, score: float | None = None, extra_ids: list[str] | None = None, section: str | None = None) -> str:
        cid = ev["chunk_id"]
        if cid not in self._idx:
            source_no = len(self.items) + 1
            self.items.append({
                "doc_id": ev["doc_id"], 
                "title": ev["title"], 
                "chunk_id": cid,
                "source_id": f"S{source_no}",
                "breadcrumb": ev.get("breadcrumb") or ev["title"], 
                "page": ev.get("page"), 
                "section_title": ev.get("section_title"), "section_path": ev.get("section_path"),
                "paragraph_index": ev.get("paragraph_index"), "source_reference": ev.get("source_reference"),
                "list_group_id": ev.get("list_group_id"), "list_item_pos": ev.get("list_item_pos"),
                "list_size": ev.get("list_size"), "lead_in_id": ev.get("lead_in_id"),
                "url": ev.get("url") or None,
                "file_name": ev.get("file_name") or None, 
                "section": section or ev.get("section"),
                "retrieval_methods": ev.get("retrieval_methods") or ["graph_section"],
                "retrieval_scores": ev.get("retrieval_scores", {}),
                "graph_paths": ev.get("graph_paths", []),
                "snippet": snippet or lead(strip_prefix(ev["text"], ev["title"]), 300), 
                "score": round(float(score), 3) if score is not None else None,
                "chunk_ids": extra_ids or [cid], 
                "text": strip_prefix(ev["text"], ev["title"])
            })
            self._idx[cid] = len(self.items)
        return f"[{self._idx[cid]}]"


def pick_resolved(plan: QueryPlan, cands: list[dict]) -> list[dict]:
    if not cands:
        return []
    qn = [t for t in search.doc_terms(plan) if not t.isdigit()]
    top = cands[0]
    unique = len(cands) == 1 or cands[1]["rank"] < top["rank"] - 0.08
    confident = top["cover"] >= 0.4 and (len(qn) >= 1 or unique)
    if not confident:
        return []
    return [c for c in cands if c["cover"] >= 0.35 and c["rank"] >= top["rank"] - 0.1][:3]


def complete_list_groups(db, hits: list[dict], max_group: int = 40, window: int = 30) -> list[dict]:
    """Traversal pengantar <-> butir TANPA properti baru: ambil jendela paragraf di sekitar tiap hit, turunkan grup
    daftar (search.derive_list_groups), lalu lengkapi hit yang menjadi bagian daftar dengan pengantar + semua butirnya."""
    seeds = [h for h in hits if h.get("paragraph_index") is not None and _looks_listy(h)]
    if not seeds:
        return derive_labels(hits)
    windows = [{"doc_id": h["doc_id"], "lo": max(0, h["paragraph_index"] - window), "hi": h["paragraph_index"] + window} for h in seeds]
    derived = search.derive_list_groups(search.window_chunks(db, windows))
    wanted = {r["list_group_id"] for r in derived if r["chunk_id"] in {h["chunk_id"] for h in seeds} and r.get("list_group_id")}
    rows = [r for r in derived if r.get("list_group_id") in wanted]
    return merge_group_rows(derive_labels(hits), rows, max_group)


def _looks_listy(h: dict) -> bool:
    t = (h.get("text") or "").strip()
    return bool(h.get("list_group_id") or search._item_kind(h) or search._LEAD.search(t))


def derive_labels(hits: list[dict]) -> list[dict]:
    """Beri label daftar pada hit yang sudah berdekatan (tanpa DB), mis. hasil ekspansi bagian."""
    by_doc: dict[str, list[dict]] = {}
    for h in hits:
        by_doc.setdefault(h.get("doc_id"), []).append(h)
    for rows in by_doc.values():
        if any(r.get("paragraph_index") is not None for r in rows):
            search.derive_list_groups(rows)
    return hits


def merge_group_rows(hits: list[dict], rows: list[dict], max_group: int = 40) -> list[dict]:
    by_id = {h["chunk_id"]: h for h in hits}
    for r in rows:                                             # bawa label turunan ke hit yang sudah ada
        if r["chunk_id"] in by_id:
            by_id[r["chunk_id"]].update({k: r.get(k) for k in ("list_group_id", "lead_in_id", "list_item_pos", "list_size")})
    sizes: dict[str, int] = {}
    for r in rows:
        sizes[r.get("list_group_id")] = sizes.get(r.get("list_group_id"), 0) + 1
    for r in rows:
        if sizes.get(r.get("list_group_id"), 0) > max_group and r["chunk_id"] not in by_id and not r.get("list_item_pos") == 0:
            continue                                    # daftar sangat panjang: hanya pengantar + butir yang memang terambil
        by_id.setdefault(r["chunk_id"], {**r, "score": 0.0, "retrieval_methods": ["list_group_expansion"],
                                         "retrieval_scores": {}, "graph_paths": []})
        if "list_group_expansion" not in by_id[r["chunk_id"]].get("retrieval_methods", []) and r["chunk_id"] not in {h["chunk_id"] for h in hits}:
            by_id[r["chunk_id"]]["retrieval_methods"] = ["list_group_expansion"]
    return sorted(by_id.values(), key=lambda h: (h.get("doc_id") or "", h.get("paragraph_index") or 0, h.get("chunk_index", 0)))


def _expand_section_list(db, question: str, hits: list[dict], doc_ids: list[str]) -> list[dict]:
    """Untuk pertanyaan daftar/bagian: ambil seluruh bagian yang judulnya diminta (frasa + sinonim), lalu lengkapi daftar."""
    plan = analyze(question)
    if not (plan.list_request or plan.section_phrases):
        return complete_list_groups(db, hits)
    paths = search.match_headings(plan, search.section_heading_chunks(db, doc_ids))
    if paths:
        section_rows = search.section_path_chunks(db, doc_ids, paths)
        if section_rows:
            original = {h.get("chunk_id"): h for h in hits}
            ordered = []
            for row in section_rows:
                prior = original.get(row.get("chunk_id"), {})
                ordered.append({
                    **row, **{k: v for k, v in prior.items() if k in {"score", "retrieval_methods", "retrieval_scores", "graph_paths"}},
                    "retrieval_methods": list(dict.fromkeys((prior.get("retrieval_methods") or []) + ["section_heading_expansion"])),
                })
            # hit di luar bagian yang diminta tetap dipertahankan (di akhir) agar bukti lain tidak hilang
            in_section = {r["chunk_id"] for r in ordered}
            extra = [h for h in hits if h["chunk_id"] not in in_section and not plan.section_phrases]
            ordered = sorted(ordered, key=lambda h: (h.get("doc_id") or "", h.get("chunk_index", 0))) + extra
            return complete_list_groups(db, ordered)

    section_paths = list(dict.fromkeys(h.get("section_path") for h in hits if h.get("section_path")))
    if not section_paths:
        return complete_list_groups(db, hits)
    rows = search.section_path_chunks(db, doc_ids, section_paths)
    by_id = {h.get("chunk_id"): h for h in hits}
    for row in rows:
        by_id.setdefault(row.get("chunk_id"), {**row, "score": 0.0, "retrieval_methods": ["section_expansion"], "retrieval_scores": {}, "graph_paths": []})
    return complete_list_groups(db, sorted(by_id.values(), key=lambda h: (h.get("doc_id") or "", h.get("chunk_index", 0))))


def answer(db, question: str, top_k: int = 5, context_doc_ids: list[str] | None = None,
           corpus: str | None = None, document_id: str | None = None, generate_llm: bool = False,
           neighbor_window: int | None = None) -> dict:
    plan = analyze(question)
    trace: list[str] = []
    cb, res = CiteBook(), {"corpus": corpus or "all", "matched_question": None, "evidence": [], "suggestions": [], "context_doc_ids": []}

    if document_id:
        resolved = search.docs_by_ids(db, [document_id], corpus)
        cands = resolved
        trace.append("document_id_filter")
        for doc in resolved:
            doc.setdefault("cover", 1.0)
    else:
        cands, st = search.find_documents(db, plan, corpus=corpus) if search.doc_terms(plan) else ([], [])
        trace += st
        cands = _filter_named_anchors(question, cands)
        resolved = pick_resolved(plan, cands)
        if resolved:
            trace.append("doc_resolved:" + ",".join(d["doc_id"] for d in resolved))

    if document_id and not resolved:
        resolved = search.docs_by_ids(db, [document_id], None)

    if document_id and resolved:
        hits, stages = search.search_chunks(db, plan, top_k, [document_id], corpus, restrict_doc_ids=False)
        trace.extend(stages)
        hits = _filter_named_anchors(question, hits)
        hits = _expand_section_list(db, question, hits, [document_id])
        for hit in hits:
            cb.add(hit, snippet=best_passage(hit["text"], plan.topic), score=hit.get("score", 0.0))
        res.update(
            source="graph_chunks" if hits else "none", context_doc_ids=[document_id] if hits else [],
            citations=[{k: v for k, v in c.items() if k != "text"} for c in cb.items],
            plan={"intents": plan.intents, "sections": [], "topic": plan.topic, "year": plan.year,
                  "followup": False, "resolved": [document_id], "trace": trace})
    else:
        ctx = [d for d in (context_doc_ids or [])]
        weak = not cands or cands[0]["cover"] < 0.4
        if ctx and (plan.followup or (weak and plan.intents)) and not resolved:
            resolved = search.docs_by_ids(db, ctx[:1], corpus)
            if resolved:
                trace.append("konteks_percakapan")
                for d in resolved:
                    d.setdefault("cover", 1.0)

        if resolved:
            ids = [d["doc_id"] for d in resolved]
            chunk_hits, chunk_stages = search.search_chunks(
                db, plan, top_k, ids, corpus, restrict_doc_ids=S.graph_restrict_resolved
            )
            trace.extend(chunk_stages)
            chunk_hits = _filter_named_anchors(question, chunk_hits)
            chunk_hits = _expand_section_list(db, question, chunk_hits, ids)
            if chunk_hits:
                for hit in chunk_hits:
                    cb.add(hit, snippet=best_passage(hit["text"], plan.topic), score=hit.get("score", 0.0))
                res.update(source="graph_chunks", context_doc_ids=list(dict.fromkeys(hit["doc_id"] for hit in chunk_hits)))
            else:
                plan.reintent(resolved[0]["title"])
                if not plan.sections:
                    plan.intents = ["ringkasan"]; plan.sections = list(OVERVIEW_SECTIONS)
                ev = search.section_chunks(db, ids, plan.sections)
                for e in ev:
                    cb.add(e, snippet=best_passage(e["text"], plan.topic, 500))
                res.update(source="graph_sections", context_doc_ids=ids)
        else:
            res.update(_open_question(db, plan, cands, cb, top_k, trace, question, corpus))

    cites = cb.items
    evidence = [
        {
            "n": i,
            "source_id": c.get("source_id", f"S{i}"),
            "title": c["title"],
            "section": c.get("section"),
            "breadcrumb": c.get("breadcrumb"),
            "page": c.get("page"),
            "section_title": c.get("section_title"),
            "section_path": c.get("section_path"),
            "paragraph_index": c.get("paragraph_index"),
            "source_reference": c.get("source_reference"),
            "url": c.get("url"),
            "file_name": c.get("file_name"),
            "text": c["text"],
            "doc_id": c["doc_id"],
            "chunk_id": c["chunk_id"],
            "score": c.get("score"),
            "retrieval_methods": c.get("retrieval_methods", []),
            "retrieval_scores": c.get("retrieval_scores", {}),
            "graph_paths": c.get("graph_paths", []),
            "list_group_id": c.get("list_group_id"), "list_item_pos": c.get("list_item_pos"),
            "list_size": c.get("list_size"), "lead_in_id": c.get("lead_in_id"),
        } for i, c in enumerate(cites, 1)
    ]
    # Traversal graf: tambahkan saudara berurutan (SUB_CHUNK_OF) sebagai konteks pendamping tiap chunk terpilih.
    selected_ids = {e["chunk_id"] for e in evidence}
    window = S.graph_neighbor_window if neighbor_window is None else neighbor_window
    neighbors = search.neighbor_chunks(db, list(selected_ids), window)
    for e in evidence:
        e["neighbors"] = [x for x in neighbors.get(e["chunk_id"], []) if x["chunk_id"] not in selected_ids and x["text"]]

    # Catatan: generasi LLM TIDAK dilakukan di sini. Satu-satunya pemanggilan Groq ada di kb_api (rag.generate)
    # agar konteks yang dicatat identik dengan konteks yang dikirim dan tidak ada panggilan ganda.
    answer_text = ""

    res.update(
        answer=answer_text,
        citations=[{k: v for k, v in c.items() if k != "text"} for c in cites],
        evidence=evidence,
        plan={"intents": plan.intents, "sections": plan.sections, "topic": plan.topic, "year": plan.year, "followup": plan.followup,
              "resolved": [d["doc_id"] for d in resolved], "trace": trace}
    )
    return res


def answer_vector_only(db, question: str, top_k: int = 5, corpus: str | None = None,
                       document_id: str | None = None, context_doc_ids: list[str] | None = None,
                       generate_llm: bool = False, candidate_k: int | None = None) -> dict:
    followup_plan = analyze(question)
    prior_doc_id = (context_doc_ids or [None])[0] if followup_plan.followup else None
    scope_doc_id = document_id or prior_doc_id

    # Keep the legacy standalone Vector RAG candidate depth when candidate_k is
    # omitted. Hybrid orchestration passes a bounded depth to avoid reranking
    # hundreds of rows that will never enter the final answer context.
    hits = search.vector_only_chunks(db, question, candidate_k or max(top_k * 20, 100), corpus, scope_doc_id)
    hits = search.rerank_vector_hits(question, hits)
    hits = _filter_named_anchors(question, hits)
    hits = hits[:top_k]

    citations, evidence = [], []
    for hit in hits:
        text = (hit.get("text") or "").strip()
        passage = best_passage(text, terms(question), 520)
        index = len(evidence) + 1
        citations.append({
            "source_id": f"S{index}", "doc_id": hit["doc_id"], "title": hit["title"], "chunk_id": hit["chunk_id"],
            "breadcrumb": hit.get("breadcrumb"), "page": hit.get("page"), "url": hit.get("url"),
            "snippet": passage, "score": float(hit.get("score") or 0.0), "section": hit.get("section"),
            "section_title": hit.get("section_title"), "section_path": hit.get("section_path"),
            "paragraph_index": hit.get("paragraph_index"), "source_reference": hit.get("source_reference"),
            "file_name": hit.get("file_name"), "retrieval_methods": ["vector"],
            "retrieval_scores": {"vector": float(hit.get("score") or 0.0)},
        })
        evidence.append({
            "n": index, "source_id": f"S{index}", "title": hit["title"], "section": hit.get("section"),
            "section_title": hit.get("section_title"), "section_path": hit.get("section_path"),
            "paragraph_index": hit.get("paragraph_index"), "source_reference": hit.get("source_reference"),
            "breadcrumb": hit.get("breadcrumb"), "page": hit.get("page"), "url": hit.get("url"),
            "file_name": hit.get("file_name"), "text": text, "doc_id": hit["doc_id"], "chunk_id": hit["chunk_id"],
            "score": float(hit.get("score") or 0.0), "retrieval_methods": ["vector"],
            "retrieval_scores": {"vector": float(hit.get("score") or 0.0)}, "graph_paths": []
        })

    answer_text = ""   # generasi dilakukan satu kali di kb_api (rag.generate)

    best_score = max((float(hit.get("score") or 0.0) for hit in hits), default=0.0)
    return {
        "answer": answer_text, "source": "vector_chunks", "corpus": corpus,
        "context_doc_ids": list(dict.fromkeys(hit["doc_id"] for hit in hits)),
        "citations": citations, "evidence": evidence,
        "plan": {"trace": ["vector_only"], "resolved": [scope_doc_id] if scope_doc_id else [],
                 "best_similarity": round(best_score, 4), "evidence_sufficient": bool(hits)},
    }


def answer_hybrid(db, question: str, top_k: int = 5, context_doc_ids: list[str] | None = None,
                  corpus: str | None = None, document_id: str | None = None,
                  neighbor_window: int | None = None) -> dict:
    """Run both retrievers; keep Graph's exact section/list evidence authoritative when it matches."""
    outcomes: list[tuple[str, dict | None, str | None]] = []
    retriever_ms: dict[str, float] = {}
    started = time.perf_counter()
    try:
        outcomes.append(("graph", answer(db, question, top_k=max(top_k * 3, 15),
                                          context_doc_ids=context_doc_ids, corpus=corpus,
                                          document_id=document_id, neighbor_window=neighbor_window), None))
    except Exception as exc:  # each retrieval is an independent fallback path
        outcomes.append(("graph", None, f"{type(exc).__name__}: {exc}"))
    finally:
        retriever_ms["graph"] = round((time.perf_counter() - started) * 1000, 2)
    plan = analyze(question)
    graph_result = outcomes[0][1]
    structural = bool(plan.list_request or plan.section_phrases)

    # A section/list query has a structural answer in Graph RAG: the matched heading identifies the right
    # section and its paragraph/list links preserve every item. Resolve it FIRST; when it matches, Vector adds
    # nothing (and used to cost ~2 minutes), so it is skipped unless HYBRID_SKIP_VECTOR_ON_GRAPH_SECTION=false.
    targeted_rows: list[dict] = []
    if graph_result and structural:
        started = time.perf_counter()
        targeted_rows = _graph_targeted_section_rows(db, question, graph_result, corpus)
        retriever_ms["graph_section_expansion"] = round((time.perf_counter() - started) * 1000, 2)

    skip_vector = bool(targeted_rows) and S.hybrid_skip_vector_on_graph_section
    if skip_vector:
        retriever_ms["vector"] = 0.0
        outcomes.append(("vector", None, None))
    else:
        started = time.perf_counter()
        try:
            hybrid_top_k = max(top_k * 3, 15)
            outcomes.append(("vector", answer_vector_only(db, question, top_k=hybrid_top_k,
                                                           corpus=corpus, document_id=document_id,
                                                           context_doc_ids=context_doc_ids,
                                                           candidate_k=max(hybrid_top_k * 2, 30)), None))
        except Exception as exc:
            outcomes.append(("vector", None, f"{type(exc).__name__}: {exc}"))
        finally:
            retriever_ms["vector"] = round((time.perf_counter() - started) * 1000, 2)

    usable = [(name, result) for name, result, error in outcomes if result is not None]
    vector_result = outcomes[1][1]

    if graph_result and structural and targeted_rows:
        graph_rows = list(graph_result.get("evidence", []))
        if True:
            seen_ids: set[str] = set()
            selected = []
            for row in sorted(targeted_rows, key=lambda item: (
                    item.get("doc_id") or "", item.get("paragraph_index", item.get("chunk_index", 0)))):
                chunk_id = row.get("chunk_id")
                if not chunk_id or chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                item = dict(row)
                item["retrieval_methods"] = list(dict.fromkeys((item.get("retrieval_methods") or []) + ["graph"]))
                selected.append(item)
            for index, item in enumerate(selected, 1):
                item["n"] = index
            citations = [{k: v for k, v in item.items() if k not in {"text", "n", "neighbors"}} |
                         {"snippet": item.get("text", "")[:520]} for item in selected]
            methods = [name for name, result in usable if result.get("evidence")]
            failures = {name: error for name, result, error in outcomes if error}
            vector_note = ("vector_skipped_graph_section_match" if skip_vector else
                           "vector_ran_as_secondary_retriever" if vector_result else "vector_unavailable")
            return {
                "answer": "", "source": "hybrid_graph_vector", "corpus": corpus,
                "context_doc_ids": list(dict.fromkeys(item.get("doc_id") for item in selected if item.get("doc_id"))),
                "citations": citations, "evidence": selected,
                "plan": {**graph_result.get("plan", {}),
                         "trace": ["hybrid_graph_section_priority", *graph_result.get("plan", {}).get("trace", []), vector_note],
                         "retrieval_failures": failures},
                "retrieval": {"methods": methods, "selection_strategy": "graph_section_priority",
                              "vector_skipped": skip_vector,
                              "vector_skip_reason": "graph_section_match" if skip_vector else None,
                              "graph_candidates": len(graph_rows),
                              "vector_candidates": len(vector_result.get("evidence", [])) if vector_result else 0,
                              "selected_chunks": len(selected), "retriever_ms": retriever_ms,
                              "failures": failures},
            }

    lists = [result.get("citations", []) for _, result in usable]
    methods = [name for name, _ in usable]
    if lists:
        weights = [2.5 if name == "graph" else 1.0 for name in methods]
        fused = search.rrf(lists, key="chunk_id", methods=methods, weights=weights)
    else:
        fused = []

    evidence_by_id: dict[str, dict] = {}
    for _, result in usable:
        for item in result.get("evidence", []):
            evidence_by_id.setdefault(item.get("chunk_id"), item)
    ordered = []
    for item in fused:
        evidence = evidence_by_id.get(item.get("chunk_id"))
        if not evidence:
            continue
        row = dict(evidence)
        row.update(score=item.get("rrf"), retrieval_methods=item.get("retrieval_methods", []),
                   retrieval_scores=item.get("retrieval_scores", {}), graph_paths=item.get("graph_paths", []))
        ordered.append(row)
    # Avoid duplicate passages and keep the top evidence from the fused ranking.
    seen_text: set[str] = set()
    deduped = []
    for row in ordered:
        normalized = " ".join((row.get("text") or "").lower().split())
        if (not normalized or normalized in seen_text or
                ("section_heading_expansion" not in row.get("retrieval_methods", []) and
                 not _answer_bearing_evidence(question, row))):
            continue
        seen_text.add(normalized)
        deduped.append(row)
        if len(deduped) >= top_k:
            break
    for index, item in enumerate(deduped, 1):
        item["n"] = index
    citations = [{k: v for k, v in item.items() if k not in {"text", "n", "neighbors"}} |
                 {"snippet": item.get("text", "")[:520]} for item in deduped]
    available_methods = [name for name, result in usable if result.get("evidence")]
    failed = {name: error for name, result, error in outcomes if error}
    return {
        "answer": "", "source": "hybrid_graph_vector" if deduped else "none", "corpus": corpus,
        "context_doc_ids": list(dict.fromkeys(item.get("doc_id") for item in deduped if item.get("doc_id"))),
        "citations": citations, "evidence": deduped,
        "plan": {"trace": ["hybrid_rrf", *[f"{name}_available" for name in available_methods]],
                 "resolved": [], "retrieval_failures": failed},
        "retrieval": {"methods": available_methods, "graph_candidates": len(outcomes[0][1].get("evidence", [])) if outcomes[0][1] else 0,
                      "vector_candidates": len(outcomes[1][1].get("evidence", [])) if outcomes[1][1] else 0,
                      "fused_candidates": len(fused), "selected_chunks": len(deduped),
                      "retriever_ms": retriever_ms, "failures": failed},
    }


def _graph_targeted_section_rows(db, question: str, graph_result: dict,
                                 corpus: str | None = None) -> list[dict]:
    """Re-read the exact Graph heading and its complete list, excluding neighboring sections."""
    plan = analyze(question)
    graph_plan = graph_result.get("plan", {})
    doc_ids = list(dict.fromkeys(graph_plan.get("resolved", []) or graph_result.get("context_doc_ids", [])))
    if not doc_ids:
        return []
    headings = search.section_heading_chunks(db, doc_ids)
    paths = search.match_headings(plan, headings)
    if not paths:
        paths = list(dict.fromkeys(
            row.get("section_path") for row in graph_result.get("evidence", [])
            if row.get("section_path") and "section_heading_expansion" in row.get("retrieval_methods", [])
        ))
    rows = search.section_path_chunks(db, doc_ids, paths) if paths else []
    if not rows:
        return []
    rows = complete_list_groups(db, rows)
    # List-window recovery can touch neighboring paragraphs; retain only the matched heading subtree.
    rows = [row for row in rows if any(
        row.get("section_path") == path or (row.get("section_path") or "").startswith(path + " > ")
        for path in paths
    )]
    rows.sort(key=lambda row: (row.get("doc_id") or "", row.get("paragraph_index", row.get("chunk_index", 0))))
    for row in rows:
        row["retrieval_methods"] = list(dict.fromkeys(
            (row.get("retrieval_methods") or []) + ["section_heading_expansion", "graph"]
        ))
        row.setdefault("retrieval_scores", {})
        row.setdefault("graph_paths", [])
    return rows


def _answer_bearing_evidence(question: str, row: dict) -> bool:
    """Discard title/tag/table-of-contents fragments that do not support this query."""
    text = " ".join(str(row.get("text") or "").split())
    if len(text) < 45:
        return False
    metadata_sections = {"tag", "identitas", "deskripsi", "mitra", "regulasi", "sumber_data"}
    if row.get("section") in metadata_sections and len(text) < 180:
        return False
    plan = analyze(question)
    topic = {stem(token) for token in (plan.topic or terms(question)) if len(token) > 2 and not token.isdigit()}
    content = {stem(token) for token in terms(text)}
    if topic and len(topic & content) / len(topic) < 0.12:
        return False
    return True


def _open_question(db, plan: QueryPlan, cands: list[dict], cb: CiteBook, top_k: int, trace: list[str], question: str, corpus: str | None = None) -> dict:
    out = {"source": "none", "context_doc_ids": []}
    hits, st = search.search_chunks(db, plan, top_k, [d["doc_id"] for d in cands[:5]] if cands else None,
                                    corpus, restrict_doc_ids=False)
    hits = _filter_named_anchors(question, hits)
    trace += st
    if hits:
        ids = list(dict.fromkeys(h["doc_id"] for h in hits))
        hits = _expand_section_list(db, question, hits, ids)
        for hit in hits:
            cb.add(hit, snippet=best_passage(hit["text"], plan.topic), score=hit.get("score", 0.0))
        out.update(source="chunks", context_doc_ids=ids[:3], expansion=[], suggestions=[f"Jelaskan isi {hits[0]['title']}"])
        return out

    ents = search.entity_docs(db, plan, corpus=corpus)
    if ents:
        trace.append("entity_docs")
        ids = list(dict.fromkeys(e["doc_id"] for e in ents))[:5]
        chunk_hits, _ = search.search_chunks(db, plan, top_k, ids, corpus, restrict_doc_ids=False)
        for hit in chunk_hits:
            cb.add(hit, snippet=best_passage(hit["text"], plan.topic), score=hit.get("score", 0.0))
        out.update(source="entity_chunks", context_doc_ids=ids[:3])
        return out

    return out
