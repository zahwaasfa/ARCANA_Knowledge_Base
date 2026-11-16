"""Vector candidate limit is bounded (never the index size) and final citations map to the right chunks."""
import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kb import rag, search
from kb.config import S


class RecordingDB:
    def __init__(self):
        self.queries = []

    def read(self, query, **params):
        self.queries.append((query, params))
        return []


def test_vector_candidate_limit_is_bounded_and_independent_of_index_size(monkeypatch):
    monkeypatch.setattr(S.__class__, "embeddings", True, raising=False)
    monkeypatch.setitem(sys.modules, "kb.embeddings", types.SimpleNamespace(embed=lambda xs: [[0.1] * 4]))
    db = RecordingDB()
    search.vector_only_chunks(db, "pertanyaan", top_k=30)
    limits = [p["candidate_limit"] for q, p in db.queries if "candidate_limit" in p]
    assert limits and all(l <= S.vector_max_candidates for l in limits)
    assert not any("count(c)" in q for q, _ in db.queries), "no per-request count over all embedded chunks"
    assert search.vector_candidate_limit(10**6) == S.vector_max_candidates
    assert search.vector_candidate_limit(5) < 100


def test_scoped_vector_retries_globally_with_bounded_limit(monkeypatch):
    monkeypatch.setattr(S.__class__, "embeddings", True, raising=False)
    monkeypatch.setitem(sys.modules, "kb.embeddings", types.SimpleNamespace(embed=lambda xs: [[0.1] * 4]))
    db = RecordingDB()
    search.vector_only_chunks(db, "q", top_k=5, doc_id="DOC-041")
    assert len(db.queries) == 2 and db.queries[1][1]["doc_id"] is None
    assert all(p["candidate_limit"] <= S.vector_max_candidates for _, p in db.queries)


def test_final_citations_point_to_cited_chunks_not_prompt_order():
    sent = [{"n": i, "doc_id": "DOC-041", "title": "NA IKN", "chunk_id": f"DOC-041_p{i}", "page": 11,
             "section_title": "B. Identifikasi Masalah", "text": f"butir {i}", "neighbors": []} for i in range(1, 6)]
    # model cites prompt slots 2..5 (slot 1 is the unnumbered lead-in)
    text, cites, mapping = rag.finalize_citations("A [2]. B [3]. C [4]. D [5].", sent)
    assert text == "A [1]. B [2]. C [3]. D [4]."
    assert [c["chunk_id"] for c in cites] == [f"DOC-041_p{i}" for i in (2, 3, 4, 5)]
    assert [c["prompt_n"] for c in cites] == [2, 3, 4, 5]
    assert all(c["page"] == 11 and c["title"] == "NA IKN" and c["section_title"] for c in cites)
    assert mapping == {"2": 1, "3": 2, "4": 3, "5": 4}


def test_ikn_identifikasi_masalah_returns_four_items_from_doc041_page11_without_vector(monkeypatch):
    from kb import retrieval
    q = "Apa saja identifikasi masalah dalam Naskah Akademik RUU Nomor 3 Tahun 2022 tentang Ibu Kota Negara?"
    path = "Isi Dokumen > PENDAHULUAN > B. Identifikasi Masalah"
    rows = [{"chunk_id": f"DOC-041_p{42 + i:07d}", "doc_id": "DOC-041", "title": "NA IKN", "page": 11, "paragraph_index": 42 + i,
             "chunk_index": 70 + i, "section_title": "B. Identifikasi Masalah", "section_path": path,
             "text": "Ada empat pokok masalah yang menjadi panduan penyusunan Naskah Akademik ini, yaitu:" if i == 0 else f"{i}. Butir masalah {i} IKN"}
            for i in range(5)]
    monkeypatch.setattr(retrieval, "answer", lambda *a, **k: {"evidence": rows[:1], "citations": [], "context_doc_ids": ["DOC-041"],
                                                              "plan": {"resolved": ["DOC-041"], "trace": []}})
    monkeypatch.setattr(retrieval, "answer_vector_only", lambda *a, **k: (_ for _ in ()).throw(AssertionError("vector must be skipped")))
    monkeypatch.setattr(search, "section_heading_chunks", lambda *a, **k: [{"doc_id": "DOC-041", "section_path": path}])
    monkeypatch.setattr(search, "match_headings", lambda *a, **k: [path])
    monkeypatch.setattr(search, "section_path_chunks", lambda *a, **k: [dict(r) for r in rows])
    monkeypatch.setattr(retrieval, "complete_list_groups", lambda db, r: r)
    res = retrieval.answer_hybrid(object(), q, top_k=5)
    assert res["retrieval"]["selection_strategy"] == "graph_section_priority" and res["retrieval"]["vector_skipped"]
    assert [e["doc_id"] for e in res["evidence"]] == ["DOC-041"] * 5 and {e["page"] for e in res["evidence"]} == {11}
    assert sum(e["text"][0].isdigit() for e in res["evidence"]) == 4


def test_openapi_schema_builds_and_keeps_contract():
    from kb_api import main
    spec = main.app.openapi()
    for path in ("/kb/ask", "/kb/retrieve/cross-document", "/kb/health", "/kb/questions", "/kb/entities", "/kb/feedback"):
        assert path in spec["paths"]
    assert "/kb/query" not in spec["paths"]
