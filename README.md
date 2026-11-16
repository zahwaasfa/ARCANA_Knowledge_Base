# ARCANA Knowledge Graph External Knowledge Base

General-purpose external knowledge service for ARCANA: document ingestion, generic Neo4j knowledge graph, optional vector indexing, Graph/Vector retrieval, grounded answer generation, and evidence citations. Existing Bappenas and Notetaker records are retained as domain data and evaluation material; Notetaker is not the service identity.

## Implemented runtime

- FastAPI service (`kb_api.main:app`), OpenAPI at `/docs`, health at `/kb/health`.
- Hybrid answer endpoint: `POST /kb/ask` (Graph + Vector via reciprocal rank fusion).
- Catalog and operations endpoints: `GET /kb/questions`, `GET /kb/entities`, `POST /kb/feedback`, and `GET /kb/health`.
- Public KB requests search across all ingested documents; callers do not select a corpus. Health lists every document and its stored corpus as diagnostic metadata.
- Independent cross-document evidence endpoint: `POST /kb/retrieve/cross-document`.
- Ingestion supports PDF, DOCX, PPTX, XLSX, CSV, TXT, and Markdown extraction. Actual domain catalogs and metadata adaptation remain configured in current source; see limitations below.
- Neo4j schema and indexes: `schema/`; graph contract: `docs/GRAPH_CONTRACT.md`.

## Quick start

1. Copy `.env.example` to `.env`; set Neo4j connection and credentials. Configure Groq only if answer generation is enabled.
2. Start local Neo4j if needed: `docker compose --profile local-db up -d neo4j`.
3. Initialize schema and ingest configured data: `docker compose run --rm loader`.
4. Start API: `docker compose up -d kb-api`; open `http://localhost:8000/docs`. Development auth is disabled by default; set `KB_API_AUTH_REQUIRED=true` and `KB_API_KEY` to require a key (also provide it to Streamlit).

