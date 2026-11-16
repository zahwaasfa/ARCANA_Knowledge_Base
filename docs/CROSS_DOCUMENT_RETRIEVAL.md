# Cross-document retrieval

`kb/cross_document.py` is a domain-independent orchestration layer using the shared `kb.search.search_chunks` implementation. By default it searches across all indexed corpora; explicit corpus or document-ID filters narrow the search. It deduplicates chunk IDs, expands from initial documents through graph links, searches linked documents, and returns source IDs, evidence, graph-link metadata, and a trace. The runtime endpoint is `POST /kb/retrieve/cross-document`.

## Perubahan F4-fix
- Evidence disaring sebelum dikembalikan: chunk metadata katalog (`tag/deskripsi/mitra/identitas/...`), header/footer, nomor halaman, OCR terfragmentasi, dan chunk berkoherensi rendah dibuang (jumlahnya di `retrieval.dropped`). Chunk yang hanya datang dari jalur entitas graf wajib memuat >=50% istilah topik.
- Pertanyaan daftar/section memakai ekspansi section Graph yang sama dengan `/kb/ask` (lead-in + semua butir).
- `graph_links` hanya memuat relasi `SHARED_ENTITY` dengan `source_doc_id`, `target_doc_id`, `shared` (kind, name, doc_frequency) dan `reason`. Tag generik (`laporan`, tahun, dll.), entitas ber-document-frequency tinggi (`XDOC_LINK_MAX_DOC_FREQ`), entitas yang tidak dibahas di evidence/pertanyaan, dan dokumen target tanpa evidence relevan tidak dijadikan relasi.
- `search.expand_graph` (dipakai jalur graph /kb/ask) juga tidak lagi memakai tag generik/tahun.

Current limitations from source inspection: the implemented document-to-document expansion in `kb/search.py` uses shared `Tag` nodes; this is keyword/tag graph traversal, not demonstrated semantic entity resolution. The supplied graph schema has generic `Organization`, `Location`, `Reference`, `Category`, and `Tag` nodes, but no canonical entity-resolution/disambiguation service or conflict detector. `document_type` is reported as unsupported by the current schema rather than silently applied. Multi-hop reasoning beyond the existing tag expansion is not claimed.

## Independent fixture

`evaluation/cross_document/scenarios.json` is explicitly synthetic and exists to exercise multi-document/multi-hop-shaped contracts. Its facts are not ARCANA source facts and must not be treated as benchmark results. Real evidence-grounded scenarios require ingesting the listed actual source documents and verifying each quote/citation against them.
