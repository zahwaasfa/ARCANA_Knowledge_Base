"""Regresi: pertanyaan daftar ("apa saja identifikasi masalah ...") pada naskah akademik/regulasi/laporan."""
import re
from types import SimpleNamespace
import pytest
from kb import rag, retrieval, search
from kb.chunker import build_chunks
from kb.extract import text_to_blocks
from kb.query_plan import analyze

SAMPLE = """B. Identifikasi Masalah
Ada empat pokok masalah yang menjadi panduan penyusunan Naskah
Akademik ini, yaitu:
1. Bagaimana permasalahan yang dihadapi terkait pokok substansi?
2. Mengapa perlu adanya Undang-Undang tentang Perubahan UU IKN?
3. Apa yang menjadi pertimbangan atau landasan filosofis?
4. Apa saja sasaran yang akan diwujudkan?
C. Tujuan dan Kegunaan
Berdasarkan latar belakang, Naskah Akademik ini
bertujuan untuk:
1. Merumuskan permasalahan yang dihadapi;
2. Merumuskan permasalahan hukum."""


def _chunks():
    return [c for c in build_chunks("D", "NA IKN", text_to_blocks(SAMPLE, 11)) if c.is_leaf]


def test_lettered_heading_with_minor_words_is_heading_not_list_item():
    blocks = text_to_blocks(SAMPLE, 11)
    assert any(b["kind"] == "h" and b["text"] == "C. Tujuan dan Kegunaan" for b in blocks)


def test_list_group_links_lead_in_and_items_in_own_section():
    leaves = _chunks()
    ident = [c for c in leaves if c.section_path.endswith("B. Identifikasi Masalah")]
    assert [c.list_item_pos for c in ident] == [0, 1, 2, 3, 4]
    assert len({c.list_group_id for c in ident}) == 1 and ident[0].list_size == 4
    assert all(c.lead_in_id == ident[0].chunk_id for c in ident[1:])
    assert "Ada empat pokok masalah" in ident[1].list_context
    tujuan = [c for c in leaves if c.section_path.endswith("C. Tujuan dan Kegunaan")]
    assert len(tujuan) == 3 and tujuan[0].list_group_id != ident[0].list_group_id


def test_query_plan_separates_document_identity_from_section_request():
    p = analyze("Apa saja identifikasi masalah dari Naskah Akademik RUU Nomor 3 Tahun 2022?")
    assert p.list_request and "identifikasi masalah" in p.section_phrases
    assert p.reg_refs == ["nomor 3 tahun 2022"]
    assert {"naskah", "akademik", "rancangan", "undang"} <= set(p.doc_topic)
    assert "identifikasi" not in p.doc_topic and "masalah" not in p.doc_topic


def test_document_resolved_by_regulation_number_and_alias():
    plan = analyze("Apa saja identifikasi masalah dari Naskah Akademik RUU Nomor 3 Tahun 2022?")
    rows = [
        {"doc_id": "a", "title": "Naskah Akademik Rancangan Undang-Undang tentang Perubahan Undang-Undang Nomor 3 Tahun 2022 tentang Ibu Kota Negara", "file_name": "x", "tags_text": "RUU IKN"},
        {"doc_id": "b", "title": "Laporan Identifikasi Masalah Perencanaan Makro", "file_name": "y", "tags_text": ""},
    ]
    ranked = search.rank_documents(plan, rows)
    assert ranked[0]["doc_id"] == "a" and ranked[0]["ref_match"]
    assert [d["doc_id"] for d in retrieval.pick_resolved(plan, ranked)][:1] == ["a"]


def test_acronym_filter_does_not_drop_chunks_that_spell_out_the_acronym():
    q = "Apa saja identifikasi masalah dari Naskah Akademik RUU Nomor 3 Tahun 2022?"
    good = {"chunk_id": "g", "title": "Rancangan Undang-Undang", "text": "Ada empat pokok masalah ... Rancangan Undang-Undang tentang Perubahan UU IKN?"}
    other = {"chunk_id": "o", "title": "t", "text": "asas pembentukan RUU Perubahan UU IKN"}
    assert {h["chunk_id"] for h in retrieval._filter_named_anchors(q, [good, other])} == {"g", "o"}


def test_match_headings_uses_phrase_and_synonyms():
    plan = analyze("Apa saja identifikasi masalah dari Naskah Akademik RUU Nomor 3 Tahun 2022?")
    heads = [{"doc_id": "d", "title": "Naskah Akademik RUU", "section_path": p} for p in
             ("PENDAHULUAN > A. Latar Belakang", "PENDAHULUAN > B. Identifikasi Masalah", "PENDAHULUAN > C. Tujuan dan Kegunaan")]
    assert search.match_headings(plan, heads) == ["PENDAHULUAN > B. Identifikasi Masalah"]
    heads2 = [{"doc_id": "d", "title": "Kajian", "section_path": "BAB I > 1.2 Rumusan Masalah"}]
    assert search.match_headings(plan, heads2) == ["BAB I > 1.2 Rumusan Masalah"]


def _ev(leaves, **extra):
    return [{"n": i, "chunk_id": c.chunk_id, "doc_id": c.doc_id, "title": "NA IKN", "breadcrumb": c.breadcrumb,
             "section_path": c.section_path, "text": c.text, "paragraph_index": c.paragraph_index,
             "list_group_id": c.list_group_id, "list_item_pos": c.list_item_pos, "list_size": c.list_size,
             "neighbors": [], **extra} for i, c in enumerate(leaves, 1)]


def test_context_keeps_whole_list_in_document_order_and_labels_items():
    leaves = _chunks()
    ev = list(reversed(_ev(leaves)))                     # urutan retrieval acak
    q = "Apa saja identifikasi masalah dari Naskah Akademik RUU Nomor 3 Tahun 2022?"
    sel = rag.select_context(ev, question=q)
    idx = [e["paragraph_index"] for e in sel]
    assert idx == sorted(idx) and len(sel) == len(leaves)
    prompt = rag.build_prompt(q, sel).context_text
    assert "(kalimat pengantar daftar)" in prompt and "(butir 4 dari 4 pada daftar yang sama)" in prompt


def test_context_budget_drops_other_groups_not_items_of_the_requested_list():
    leaves = _chunks()
    q = "Apa saja identifikasi masalah dari Naskah Akademik?"
    ev = _ev(leaves)
    need = sum(len(e["text"]) + 240 for e in ev if "Identifikasi" in e["section_path"])
    sel = rag.select_context(ev, question=q, max_chars=need + 10)
    assert [e["list_item_pos"] for e in sel if "Identifikasi" in e["section_path"]] == [0, 1, 2, 3, 4]
    assert not any("Tujuan" in e["section_path"] for e in sel)


def test_merge_group_rows_completes_partial_hits_in_order():
    rows = [{"chunk_id": f"c{i}", "doc_id": "d", "paragraph_index": i, "list_group_id": "g", "list_item_pos": i, "text": "x"} for i in range(5)]
    hits = [dict(rows[3], score=1.0, retrieval_methods=["fulltext"])]
    out = retrieval.merge_group_rows(hits, rows)
    assert [h["chunk_id"] for h in out] == ["c0", "c1", "c2", "c3", "c4"]
    assert out[3]["retrieval_methods"] == ["fulltext"]


class _LLM:
    def __init__(self, replies): self.replies, self.prompts = list(replies), []
    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        return SimpleNamespace(content=self.replies.pop(0), response_metadata={})


def test_false_not_found_is_rechecked_when_context_contains_requested_section():
    pytest.importorskip("langchain_core")
    q = "Apa saja identifikasi masalah dari Naskah Akademik RUU Nomor 3 Tahun 2022?"
    llm = _LLM(["Informasi tersebut belum ditemukan dalam sumber yang diperiksa.",
                "Ada empat pokok masalah. [1] 1. Bagaimana permasalahan yang dihadapi? [2]"])
    out = rag.generate(q, _ev(_chunks()), llm=llm, sleep=lambda s: None)
    assert out["status"] == "generated" and out.get("recheck") and "PEMERIKSAAN ULANG" in llm.prompts[1]
    assert "belum ditemukan" not in out["answer"]


def test_real_not_found_is_not_forced():
    pytest.importorskip("langchain_core")
    ev = [{"n": 1, "chunk_id": "c", "doc_id": "d", "title": "Laporan Lain", "breadcrumb": "Laporan Lain > Anggaran",
           "text": "Realisasi anggaran triwulan dua.", "neighbors": []}]
    llm = _LLM(["Informasi tersebut belum ditemukan dalam sumber yang diperiksa."])
    out = rag.generate("Apa saja identifikasi masalah dari Naskah Akademik RUU Nomor 3 Tahun 2022?", ev, llm=llm, sleep=lambda s: None)
    assert not out.get("recheck") and len(llm.prompts) == 1


# ---- data LAMA (tanpa ingest ulang): heading "C." tergabung ke paragraf, tanpa properti daftar ----
def _legacy_rows():
    texts = [("B. Identifikasi Masalah", "Ada empat pokok masalah yang menjadi panduan, yaitu:", None),
             ("B. Identifikasi Masalah", "1. Bagaimana permasalahan?", "1."), ("B. Identifikasi Masalah", "2. Mengapa perlu?", "2."),
             ("B. Identifikasi Masalah", "3. Apa pertimbangan?", "3."), ("B. Identifikasi Masalah", "4. Apa sasaran?", "4."),
             ("B. Identifikasi Masalah", "C. Tujuan dan Kegunaan Berdasarkan latar belakang, Naskah ini bertujuan untuk:", "C."),
             ("B. Identifikasi Masalah", "1. Merumuskan masalah;", "1."), ("B. Identifikasi Masalah", "2. Merumuskan hukum.", "2.")]
    return [{"chunk_id": f"c{i}", "doc_id": "d", "paragraph_index": i, "chunk_index": i, "text": t, "item_no": it,
             "section_path": f"PENDAHULUAN > {sec}"} for i, (sec, t, it) in enumerate(texts)]


def test_derive_list_groups_on_legacy_chunks_splits_at_letter_heading():
    rows = search.derive_list_groups(_legacy_rows())
    g1 = [r for r in rows if r["list_group_id"] == rows[0]["list_group_id"]]
    assert [r["list_item_pos"] for r in g1] == [0, 1, 2, 3, 4] and rows[5]["list_group_id"] != rows[0]["list_group_id"]
    assert rows[5]["list_item_pos"] == 0 and rows[6]["lead_in_id"] == "c5"


def test_trim_section_rows_cuts_merged_letter_heading():
    rows = _legacy_rows()
    out = search.trim_section_rows(rows, ["PENDAHULUAN > B. Identifikasi Masalah"])
    assert [r["chunk_id"] for r in out] == ["c0", "c1", "c2", "c3", "c4"]


def test_complete_list_groups_without_new_properties():
    rows = _legacy_rows()
    class DB:
        def read(self, q, **p): return [dict(r) for r in rows]
    hits = [dict(rows[3], score=1.0, retrieval_methods=["fulltext"])]
    out = retrieval.complete_list_groups(DB(), hits)
    assert [h["chunk_id"] for h in out] == ["c0", "c1", "c2", "c3", "c4"] and out[3]["list_item_pos"] == 3
