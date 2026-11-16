# Menjalankan ulang service dengan aman (tanpa menghapus data Neo4j)

Gunakan HANYA perintah berikut. Jangan memakai `docker compose down -v`, `docker volume rm`, `docker system prune --volumes`,
dan jangan menjalankan `run_ingest --force` (menimpa data).

```bash
docker compose build kb-api
docker compose up -d --no-deps --force-recreate kb-api   # hanya API; volume Neo4j tidak disentuh
docker compose ps
curl -s localhost:8000/kb/health | head -c 400
```
Env baru (opsional, ada default): `VECTOR_MAX_CANDIDATES=300`, `VECTOR_CANDIDATE_MULTIPLIER=4`,
`VECTOR_SCOPED_CANDIDATE_MULTIPLIER=12`, `HYBRID_SKIP_VECTOR_ON_GRAPH_SECTION=true`, `XDOC_MIN_COVERAGE=0.34`, `XDOC_LINK_MAX_DOC_FREQ=6`.
Perubahan hanya kode; skema, data, dan embedding tidak diubah (tidak perlu ingest ulang).

Tes lokal (tanpa Neo4j/LLM): `pip install -r requirements.txt pytest && python -m pytest -q tests`.
Verifikasi live setelah restart: ulangi pertanyaan IKN pada `/kb/ask` dan `/kb/retrieve/cross-document`; cek `timings_ms.retrieval`,
`retrieval.vector_skipped`, `retrieval.dropped`, dan `graph_links`.
