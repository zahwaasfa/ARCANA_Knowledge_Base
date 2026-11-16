"""Konfigurasi terpusat dari environment/.env (tidak ada kredensial di kode)."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent


def _s(k: str, d: str = "", *alt: str) -> str:
    for key in (k, *alt):
        v = os.getenv(key)
        if v not in (None, ""):
            return v.split("#")[0].strip()
    return d


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    neo4j_uri: str = _s("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = _s("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = _s("NEO4J_PASSWORD", "")
    neo4j_database: str | None = _s("NEO4J_DATABASE") or None
    docs_dir: Path = ROOT / "data" / "bappenas" / "documents"
    metadata_json: Path = ROOT / "data" / "bappenas" / "metadata_bappenas.json"
    seed_json: Path = ROOT / "data" / "bappenas" / "seed_qa.json"
    notetaker_markdown_dir: Path = ROOT / "data" / "notetaker" / "markdown"
    notetaker_json_dir: Path = ROOT / "data" / "notetaker" / "json"
    schema_dir: Path = ROOT / "schema"
    reports_dir: Path = ROOT / "reports"
    max_chunk_chars: int = int(_s("MAX_CHUNK_CHARS", "1200"))
    fanout: int = int(_s("CHUNK_FANOUT", "8"))
    max_chunks_per_doc: int = int(_s("MAX_CHUNKS_PER_DOC", "3000"))
    max_table_rows: int = (int(_s("MAX_TABLE_ROWS", "20000")) or 10**9)
    crosslink_topk: int = int(_s("CROSSLINK_TOPK", "3", "CROSS_DOC_CHUNK_LINKS"))
    crosslink_min_sim: float = float(_s("CROSSLINK_MIN_SIM", "0.30"))
    embeddings: bool = _s("EMBEDDINGS", "false").lower() == "true"
    embedding_model: str = _s("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "EMBEDDING_MODEL")
    embedding_dim: int = int(_s("EMBEDDING_DIM", "384"))
    groq_api_key: str = _s("GROQ_API_KEY")
    groq_model: str = _s("GROQ_MODEL_NAME", "openai/gpt-oss-120b")
    use_llm: bool = _s("USE_LLM", "true").lower() == "true"
    top_k: int = int(_s("TOP_K_CHUNKS", "5"))
    # --- Ketahanan pemanggilan LLM (model Groq tidak diubah; hanya perilaku pemanggilan) ---
    llm_timeout: float = float(_s("LLM_TIMEOUT_SECONDS", "60"))
    llm_max_attempts: int = int(_s("LLM_MAX_ATTEMPTS", "3"))
    llm_backoff_base: float = float(_s("LLM_BACKOFF_BASE_SECONDS", "2"))
    llm_max_retry_wait: float = float(_s("LLM_MAX_RETRY_WAIT_SECONDS", "30"))
    llm_concurrency: int = int(_s("LLM_CONCURRENCY", "2"))
    llm_max_context_chars: int = int(_s("LLM_MAX_CONTEXT_CHARS", "12000"))
    llm_max_tokens: int = int(_s("LLM_MAX_TOKENS", "1500"))
    # Graph RAG murni graf + teks; vector hanya untuk mode vector (hindari pencampuran konteks).
    graph_use_vector: bool = _s("GRAPH_RAG_USE_VECTOR", "false").lower() == "true"
    graph_neighbor_window: int = int(_s("GRAPH_NEIGHBOR_WINDOW", "1"))
    # --- Optimasi Graph RAG (hasil audit; semua dapat dimatikan lewat env untuk ablasi) ---
    llm_strict_grounding: bool = _s("LLM_STRICT_GROUNDING", "true").lower() == "true"   # aturan 9 pada prompt (kedua metode)
    graph_per_doc_cap: int = int(_s("GRAPH_PER_DOC_CAP", "3"))          # cap keragaman lintasan-1 (lintasan-2 mengisi sisa slot)
    graph_entity_weight: float = float(_s("GRAPH_ENTITY_RRF_WEIGHT", "0.5"))  # bobot RRF jalur entitas graf (1.0 = perilaku lama)
    graph_min_cov: float = float(_s("GRAPH_MIN_COVERAGE", "0.2"))       # chunk jalur-entitas-saja wajib memuat >= x istilah topik (0 = off)
    graph_restrict_resolved: bool = _s("GRAPH_RESTRICT_TO_RESOLVED", "true").lower() == "true"
    # --- Batas kandidat Vector (tidak lagi dihitung dari total seluruh embedding) ---
    vector_max_candidates: int = int(_s("VECTOR_MAX_CANDIDATES", "300"))        # batas atas keras queryNodes
    vector_candidate_multiplier: int = int(_s("VECTOR_CANDIDATE_MULTIPLIER", "4"))
    vector_scoped_multiplier: int = int(_s("VECTOR_SCOPED_CANDIDATE_MULTIPLIER", "12"))  # post-filter dokumen butuh kolam lebih besar
    hybrid_skip_vector_on_graph_section: bool = _s("HYBRID_SKIP_VECTOR_ON_GRAPH_SECTION", "true").lower() == "true"
    # --- Cross-document ---
    xdoc_min_coverage: float = float(_s("XDOC_MIN_COVERAGE", "0.34"))
    xdoc_link_max_doc_freq: int = int(_s("XDOC_LINK_MAX_DOC_FREQ", "6"))
    api_url: str = _s("KB_API_URL", "http://localhost:8000")
    api_key: str = _s("KB_API_KEY")
    api_auth_required: bool = _s("KB_API_AUTH_REQUIRED", "false").lower() == "true"
    cors_origins: tuple[str, ...] = tuple(x.strip() for x in _s("KB_CORS_ORIGINS", "http://localhost:8501,http://localhost:8000").split(",") if x.strip())


S = Settings()
