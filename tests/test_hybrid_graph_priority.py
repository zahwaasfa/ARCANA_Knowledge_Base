from kb import retrieval, search


QUESTION = "Apa saja identifikasi masalah dari Naskah Akademik RUU Nomor 3 Tahun 2022?"


def _list_row(index: int, text: str) -> dict:
    return {
        "n": index + 1,
        "chunk_id": f"graph-{index}",
        "doc_id": "DOC-041",
        "title": "Naskah Akademik RUU Nomor 3 Tahun 2022",
        "section": "identifikasi_masalah",
        "section_title": "B. Identifikasi Masalah",
        "section_path": "PENDAHULUAN > B. Identifikasi Masalah",
        "paragraph_index": index,
        "chunk_index": index,
        "list_group_id": "problems",
        "list_item_pos": index,
        "list_size": 4,
        "lead_in_id": "graph-0",
        "text": text,
        "retrieval_methods": ["section_heading_expansion", "fulltext"],
        "retrieval_scores": {"fulltext": 10.0 - index},
        "graph_paths": [{"entity_kind": "Document", "entity": "DOC-041"}],
        "neighbors": [],
    }


def test_hybrid_keeps_complete_graph_section_ahead_of_vector_distractors(monkeypatch):
    graph_rows = [
        _list_row(0, "Ada empat pokok masalah yang menjadi panduan penyusunan Naskah Akademik ini, yaitu:"),
        _list_row(1, "1. Bagaimana permasalahan terkait substansi Kewenangan Khusus, keuangan, wilayah, jabatan, ruang, pertanahan, perumahan, pemantauan, serta jaminan keberlanjutan IKN?"),
        _list_row(2, "2. Mengapa diperlukan Undang-Undang tentang Perubahan UU IKN sebagai dasar pemecahan masalah tersebut?"),
        _list_row(3, "3. Apa pertimbangan atau landasan filosofis, sosiologis, dan yuridis pembentukan Rancangan Undang-Undang Perubahan UU IKN?"),
        _list_row(4, "4. Apa sasaran, jangkauan, arah pengaturan, serta ruang lingkup Rancangan Undang-Undang Perubahan UU IKN?"),
    ]
    vector_rows = [{**_list_row(i, f"Vector distraction about IKN financial details, unrelated section {i}."),
                    "chunk_id": f"vector-{i}", "section_title": "C. Ruang Lingkup Materi",
                    "section_path": "BAB VI > C. Ruang Lingkup Materi",
                    "retrieval_methods": ["vector"], "retrieval_scores": {"vector": 0.99}}
                   for i in range(5)]
    monkeypatch.setattr(retrieval, "answer", lambda *a, **k: {
        "evidence": graph_rows, "citations": [{key: value for key, value in row.items() if key != "text"} for row in graph_rows],
        "context_doc_ids": ["DOC-041"], "plan": {"resolved": ["DOC-041"], "trace": ["doc_resolved:DOC-041"]},
    })
    vector_calls = []
    monkeypatch.setattr(retrieval, "answer_vector_only", lambda *a, **k: vector_calls.append(1) or {
        "evidence": vector_rows, "citations": [{key: value for key, value in row.items() if key != "text"} for row in vector_rows],
    })
    heading_path = "PENDAHULUAN > B. Identifikasi Masalah"
    monkeypatch.setattr(search, "section_heading_chunks", lambda *a, **k: [
        {"doc_id": "DOC-041", "title": "Naskah Akademik RUU", "section_path": heading_path}])
    monkeypatch.setattr(search, "section_path_chunks", lambda *a, **k: [
        {**row, "section_path": heading_path} for row in graph_rows])
    monkeypatch.setattr(retrieval, "complete_list_groups", lambda db, rows: rows)
    monkeypatch.setattr(search, "rrf", lambda *a, **k: (_ for _ in ()).throw(AssertionError("list query must use exact Graph section")))

    result = retrieval.answer_hybrid(object(), QUESTION, top_k=1)

    assert result["plan"]["trace"][0] == "hybrid_graph_section_priority"
    assert result["retrieval"]["selection_strategy"] == "graph_section_priority"
    # Graph section match is strong -> the expensive Vector retriever must NOT run.
    assert vector_calls == []
    assert result["retrieval"]["methods"] == ["graph"]
    assert result["retrieval"]["vector_skipped"] is True
    assert result["retrieval"]["vector_candidates"] == 0
    assert result["plan"]["trace"][-1] == "vector_skipped_graph_section_match"
    assert [row["chunk_id"] for row in result["evidence"]] == [f"graph-{i}" for i in range(5)]
    assert all("graph" in row["retrieval_methods"] for row in result["evidence"])
    assert not any(row["chunk_id"].startswith("vector-") for row in result["evidence"])
    assert len(result["citations"]) == 5


def test_hybrid_rrf_weights_graph_more_than_vector_for_open_questions(monkeypatch):
    captured = {}
    graph = {"evidence": [{"chunk_id": "g", "doc_id": "d", "title": "Kebijakan AI",
                           "text": "Kebijakan AI dibuat karena pegawai membutuhkan pedoman tata kelola yang jelas.",
                           "retrieval_methods": ["fulltext"]}],
             "citations": [{"chunk_id": "g", "doc_id": "d", "title": "Kebijakan AI"}],
             "context_doc_ids": ["d"], "plan": {"trace": []}}
    vector = {"evidence": [{"chunk_id": "v", "doc_id": "d", "title": "Kebijakan AI",
                             "text": "Kebijakan AI dibuat untuk mendukung tata kelola yang konsisten.",
                             "retrieval_methods": ["vector"]}],
              "citations": [{"chunk_id": "v", "doc_id": "d", "title": "Kebijakan AI"}]}
    monkeypatch.setattr(retrieval, "answer", lambda *a, **k: graph)
    monkeypatch.setattr(retrieval, "answer_vector_only", lambda *a, **k: vector)

    def fake_rrf(lists, **kwargs):
        captured["weights"] = kwargs.get("weights")
        return [{**lists[0][0], "rrf": 1.0, "retrieval_methods": ["graph"]},
                {**lists[1][0], "rrf": 0.5, "retrieval_methods": ["vector"]}]
    monkeypatch.setattr(search, "rrf", fake_rrf)

    result = retrieval.answer_hybrid(object(), "Mengapa kebijakan AI dibuat?", top_k=2)

    assert captured["weights"] == [2.5, 1.0]
    assert result["evidence"][0]["chunk_id"] == "g"


def test_vector_still_runs_when_graph_has_no_section_match(monkeypatch):
    calls = []
    monkeypatch.setattr(retrieval, "answer", lambda *a, **k: {"evidence": [], "citations": [], "plan": {"trace": []}})
    monkeypatch.setattr(retrieval, "answer_vector_only", lambda *a, **k: calls.append(k) or {"evidence": [], "citations": []})
    retrieval.answer_hybrid(object(), "Mengapa kebijakan AI dibuat?", top_k=5)
    assert len(calls) == 1 and calls[0]["candidate_k"] == 30   # bounded row limit, not an index-wide count
