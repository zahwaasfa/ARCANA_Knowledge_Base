import unittest
from unittest.mock import patch

from kb import cross_document as cd


def chunk(cid, doc, text, **kw):
    return {"chunk_id": cid, "doc_id": doc, "title": kw.pop("title", doc), "text": text, "score": 1.0,
            "section": kw.pop("section", None), "section_title": kw.pop("section_title", None),
            "retrieval_methods": kw.pop("retrieval_methods", ["fulltext"]), **kw}


Q = "Apa saja identifikasi masalah dalam Naskah Akademik RUU Nomor 3 Tahun 2022 tentang Ibu Kota Negara?"
IKN_TITLE = "Naskah Akademik RUU tentang Perubahan UU Nomor 3 Tahun 2022 tentang Ibu Kota Negara"
GOOD = chunk("c-good", "DOC-041", "1. Bagaimana permasalahan pokok substansi Kewenangan Khusus pengelolaan keuangan IKN dalam praktik?",
             section_title="B. Identifikasi Masalah", title=IKN_TITLE, page=11)
META = [
    chunk("c-tag", "DOC-041", 'Kata kunci dokumen "Naskah Akademik": naskah akademik, ruu, ikn', section="tag", title=IKN_TITLE),
    chunk("c-desc", "DOC-041", 'Deskripsi dokumen "Naskah Akademik RUU IKN": Naskah Akademik RUU IKN tentang Ibu Kota Negara', section="deskripsi", title=IKN_TITLE),
    chunk("c-mitra", "DOC-041", 'Mitra (Internal UKE) dalam penyusunan "Naskah Akademik": Biro Hukum untuk penyusunan', section="mitra", title=IKN_TITLE),
]
NOISE = [chunk("c-page", "DOC-041", "Halaman 12"), chunk("c-ocr", "DOC-041", "l a y a n a n m e d i s k e l a s d u n i a x y z q w e r t")]
UNRELATED = chunk("c-eco", "DOC-003", "KEK Sanur menghadirkan Bali International Hospital dengan realisasi investasi Rp4,42 triliun untuk pariwisata kesehatan.",
                  retrieval_methods=["graph_entity_document_chunks"], title="Laporan Perkembangan Ekonomi Indonesia")


class CrossDocumentTests(unittest.TestCase):
    def run_retrieve(self, hits, raw_links=(), linked_hits=None, top_k=8):
        def fake_search(db, plan, k, ids, **kw):
            return ((linked_hits or []) if ids and kw.get("restrict_doc_ids") and ids != ["DOC-041"] else hits, ["fulltext"])
        with patch.object(cd.search, "find_documents", return_value=([], [])), \
             patch.object(cd.search, "search_chunks", side_effect=fake_search), \
             patch.object(cd, "find_links", return_value=list(raw_links)):
            return cd.retrieve(object(), Q, top_k)

    def test_drops_metadata_noise_and_unrelated_documents(self):
        result = self.run_retrieve([*META, *NOISE, UNRELATED, GOOD])
        self.assertEqual([e["chunk_id"] for e in result["evidence"]], ["c-good"])
        self.assertEqual(result["document_ids"], ["DOC-041"])
        self.assertEqual(result["graph_links"], [])
        dropped = result["retrieval"]["dropped"]
        self.assertEqual(dropped.get("metadata_chunk"), 3)
        self.assertGreaterEqual(dropped.get("noise", 0), 2)
        self.assertGreaterEqual(dropped.get("graph_entity_only_weak", 0) + dropped.get("low_relevance", 0), 1)

    def test_generic_tags_and_years_never_create_links(self):
        raw = [{"source_doc_id": "DOC-041", "source_title": "A", "target_doc_id": "DOC-001", "target_title": "B",
                "kind": "Tag", "name": n, "doc_frequency": 3} for n in ("laporan", "2025", "Laporan")]
        result = self.run_retrieve([GOOD], raw)
        self.assertEqual(result["graph_links"], [])
        self.assertEqual(result["document_ids"], ["DOC-041"])

    def test_specific_shared_entity_makes_explicit_link_with_relevant_target_evidence(self):
        linked = chunk("c-lnk", "DOC-050", "Identifikasi masalah pengelolaan keuangan IKN dalam kajian naskah akademik ibu kota negara ini.",
                       section_title="Pengelolaan Keuangan IKN", title="Kajian Keuangan IKN")
        raw = [{"source_doc_id": "DOC-041", "source_title": "NA IKN", "target_doc_id": "DOC-050", "target_title": "Kajian Keuangan IKN",
                "kind": "Tag", "name": "keuangan ikn", "doc_frequency": 2}]
        result = self.run_retrieve([GOOD], raw, linked_hits=[linked])
        self.assertEqual(len(result["graph_links"]), 1)
        link = result["graph_links"][0]
        self.assertEqual((link["source_doc_id"], link["target_doc_id"]), ("DOC-041", "DOC-050"))
        self.assertEqual(link["relation"], "SHARED_ENTITY")
        self.assertEqual(link["shared"][0]["name"], "keuangan ikn")
        self.assertIn("keuangan ikn", link["reason"])
        self.assertEqual(link["evidence_chunk_ids"], ["c-lnk"])
        self.assertEqual(result["document_ids"], ["DOC-041", "DOC-050"])
        self.assertEqual([s["role"] for s in result["source_documents"]], ["primary", "linked"])

    def test_link_dropped_when_target_has_no_relevant_evidence(self):
        raw = [{"source_doc_id": "DOC-041", "source_title": "NA IKN", "target_doc_id": "DOC-003", "target_title": "Ekonomi",
                "kind": "Tag", "name": "ikn", "doc_frequency": 2}]
        result = self.run_retrieve([GOOD], raw, linked_hits=[UNRELATED])
        self.assertEqual(result["graph_links"], [])
        self.assertNotIn("DOC-003", result["document_ids"])

    def test_citations_match_evidence_order(self):
        result = self.run_retrieve([GOOD])
        self.assertEqual(result["citations"][0]["source_id"], "S1")
        self.assertEqual(result["citations"][0]["chunk_id"], result["evidence"][0]["chunk_id"])
        self.assertEqual(result["citations"][0]["page"], 11)


if __name__ == "__main__":
    unittest.main()
