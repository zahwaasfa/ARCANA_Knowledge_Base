"""Cross-document retrieval over the shared KB search layer.

Differences from the first version (which returned tag-only links and unfiltered chunks):
  * Evidence must answer the question: catalog metadata (tag/deskripsi/mitra/...), headers/footers, OCR noise and
    chunks that only share generic words are dropped before ranking is returned.
  * Structural questions (list / named section) re-use the exact Graph section expansion used by /kb/ask, so list
    items and their lead-in stay together.
  * A cross-document link is returned only when (1) the shared entity is specific (not a generic tag / a year and
    low document frequency), (2) the entity is actually discussed in the retrieved evidence or the question, and
    (3) the linked document itself has relevant evidence. Every link states source, target, shared entities and a reason.
"""
from __future__ import annotations

import re

from . import retrieval, search
from .config import S
from .query_plan import analyze, stem, terms

GENERIC_TAGS = search.GENERIC_TAGS
METADATA_SECTIONS = {"tag", "identitas", "deskripsi", "manfaat", "mitra", "regulasi", "data_sekunder",
                     "sumber_data", "proses_bisnis"}
_PAGE_ONLY = re.compile(r"^\W*(?:halaman|hal\.?|page)?\s*\d{1,4}(?:\s*(?:/|dari|of)\s*\d{1,4})?\W*$", re.I)
_YEAR = re.compile(r"^\d{4}$")


def is_generic_entity(name: str) -> bool:
    n = " ".join((name or "").lower().split())
    return len(n) < 3 or bool(_YEAR.match(n)) or n.isdigit() or n in GENERIC_TAGS


def noise_reason(row: dict) -> str | None:
    """Return why a chunk is header/footer/OCR noise, else None."""
    text = " ".join(str(row.get("text") or "").split())
    if _PAGE_ONLY.match(text):
        return "page_number"
    if len(text) < 40 and not re.search(r"[:;]$", text) and not re.match(r"^\s*(?:\d+|[a-zA-Z])[.)]\s", text):
        return "too_short"
    title = " ".join(str(row.get("title") or "").lower().split())
    if title and text.lower() == title:
        return "title_repeat"
    letters = sum(ch.isalpha() for ch in text)
    if letters / max(len(text), 1) < 0.5:
        return "low_alpha_ratio"
    toks = text.split()
    if len(toks) >= 8 and sum(len(t) == 1 for t in toks) / len(toks) > 0.35:
        return "ocr_fragmented"
    return None


def _stems(text: str) -> set[str]:
    return {stem(t) for t in terms(text or "") if len(t) > 2 and not t.isdigit()}


def _topic_stems(plan, question: str) -> set[str]:
    return {stem(t) for t in (plan.topic or terms(question)) if len(t) > 2 and not t.isdigit()}


def classify(row: dict, topic: set[str], wanted_sections: set[str]) -> tuple[bool, str, float]:
    """(keep, reason, content_coverage). Coverage uses chunk text + its section title only (not doc title)."""
    if row.get("section") in METADATA_SECTIONS and row.get("section") not in wanted_sections:
        return False, "metadata_chunk", 0.0
    why = noise_reason(row)
    if why:
        return False, f"noise:{why}", 0.0
    if not topic:
        return True, "no_topic_terms", 1.0
    content = _stems(f"{row.get('text') or ''} {row.get('section_title') or ''}")
    cov = len(topic & content) / len(topic)
    methods = set(row.get("retrieval_methods") or [])
    graph_only = methods == {"graph_entity_document_chunks"}
    if graph_only and cov < 0.5:
        return False, "graph_entity_only_weak", cov
    with_title = len(topic & (content | _stems(str(row.get("title") or "")))) / len(topic)
    if cov >= S.xdoc_min_coverage or (with_title >= 0.5 and cov >= 0.2):
        return True, "relevant", cov
    return False, "low_relevance", cov


def filter_evidence(rows: list[dict], topic: set[str], wanted_sections: set[str] | None = None,
                    dropped: dict | None = None) -> list[dict]:
    out, seen = [], set()
    for row in rows:
        keep, reason, cov = classify(row, topic, wanted_sections or set())
        if not keep:
            if dropped is not None:
                key = reason.split(":")[0]
                dropped[key] = dropped.get(key, 0) + 1
            continue
        norm = " ".join(str(row.get("text") or "").lower().split())
        if norm in seen:
            continue
        seen.add(norm)
        out.append({**row, "content_coverage": round(cov, 3)})
    return out


def find_links(db, doc_ids: list[str], corpus: str | None = None, limit: int = 60) -> list[dict]:
    """Raw shared-entity candidates (Tag / Reference) between documents, with document frequency of the entity."""
    if not doc_ids:
        return []
    return search._safe(lambda: db.read(
        """MATCH (d:Document) WHERE d.doc_id IN $ids
           MATCH (d)-[:TAGGED_WITH|USES_REFERENCE]->(x)<-[:TAGGED_WITH|USES_REFERENCE]-(o:Document)
           WHERE o.doc_id <> d.doc_id AND ($corpus IS NULL OR coalesce(o.corpus,o.domain,'bappenas')=$corpus)
           WITH d, o, x, size([(x)<-[:TAGGED_WITH|USES_REFERENCE]-(z:Document) | 1]) AS df
           WHERE df <= $max_df
           RETURN d.doc_id AS source_doc_id, d.title AS source_title, o.doc_id AS target_doc_id, o.title AS target_title,
                  labels(x)[0] AS kind, coalesce(x.name, x.title) AS name, df AS doc_frequency
           LIMIT $limit""", ids=doc_ids, corpus=corpus, max_df=S.xdoc_link_max_doc_freq, limit=limit), default=[]) or []


def build_links(raw: list[dict], topic: set[str], evidence_by_doc: dict[str, str]) -> list[dict]:
    """Group raw shared-entity rows into explicit source->target relations; drop generic/ungrounded ones."""
    grouped: dict[tuple[str, str], dict] = {}
    for r in raw:
        name = r.get("name") or ""
        if is_generic_entity(name):
            continue
        ent = _stems(name)
        grounded = bool(ent and ent <= (topic | _stems(evidence_by_doc.get(r["source_doc_id"], ""))))
        if not grounded:
            continue
        key = tuple(sorted((r["source_doc_id"], r["target_doc_id"])))
        g = grouped.setdefault(key, {"source_doc_id": r["source_doc_id"], "source_title": r.get("source_title"),
                                     "target_doc_id": r["target_doc_id"], "target_title": r.get("target_title"),
                                     "shared": []})
        item = {"kind": r.get("kind"), "name": name, "doc_frequency": r.get("doc_frequency")}
        if item not in g["shared"]:
            g["shared"].append(item)
    links = []
    for g in grouped.values():
        names = ", ".join(f"{s['kind']} '{s['name']}'" for s in g["shared"][:5])
        links.append({**g, "doc_id": g["target_doc_id"], "title": g["target_title"], "relation": "SHARED_ENTITY",
                      "hops": 1, "via": "+".join(sorted({str(s["kind"]).lower() for s in g["shared"]})),
                      "reason": f"Kedua dokumen berbagi entitas spesifik ({names}) yang dibahas dalam evidence/pertanyaan.",
                      "score": float(len(g["shared"]))})
    return sorted(links, key=lambda l: l["score"], reverse=True)


def retrieve(db, question: str, top_k: int = 8, *, corpus: str | None = None,
             document_ids: list[str] | None = None, document_type: str | None = None) -> dict:
    plan = analyze(question)
    topic = _topic_stems(plan, question)
    wanted = set(plan.sections or [])
    pool = max(top_k * 3, 20)
    dropped: dict[str, int] = {}
    trace: list[str] = []
    scope = list(dict.fromkeys(document_ids or []))

    primary_ids = list(scope)
    if not scope and search.doc_terms(plan):
        cands, st = search.find_documents(db, plan, corpus=corpus)
        trace += st
        resolved = retrieval.pick_resolved(plan, retrieval._filter_named_anchors(question, cands))
        primary_ids = [d["doc_id"] for d in resolved]
        if primary_ids:
            trace.append("doc_resolved:" + ",".join(primary_ids))

    raw, stages = search.search_chunks(db, plan, pool, primary_ids or None, corpus=corpus, restrict_doc_ids=bool(scope))
    trace += stages
    for hit in raw:
        hit.setdefault("retrieval_methods", ["graph_search"])
    candidates = filter_evidence(raw, topic, wanted, dropped)

    structural: list[dict] = []
    if primary_ids and (plan.list_request or plan.section_phrases):
        rows = retrieval._graph_targeted_section_rows(db, question, {"plan": {"resolved": primary_ids}, "evidence": raw}, corpus)
        structural = filter_evidence([{**r, "retrieval_methods": list(dict.fromkeys((r.get("retrieval_methods") or []) + ["section_heading_expansion"]))}
                                      for r in rows], set(), wanted, dropped)
        if structural:
            trace.append(f"section_expansion_rows={len(structural)}")

    seen: set[str] = set()
    primary: list[dict] = []
    for row in structural + candidates:
        cid = str(row.get("chunk_id") or "")
        if cid and cid not in seen:
            seen.add(cid)
            primary.append(row)
    keep_n = max(top_k, len(structural)) if structural else top_k
    keep_n = min(keep_n, 40)
    primary_doc_ids = list(dict.fromkeys(r["doc_id"] for r in primary[:keep_n]))

    # --- substantive cross-document links (shared specific entity + relevant evidence in target) ---
    evidence_text: dict[str, str] = {}
    for r in primary[:keep_n]:
        evidence_text[r["doc_id"]] = evidence_text.get(r["doc_id"], "") + " " + str(r.get("text") or "")
    raw_links = find_links(db, primary_doc_ids, corpus)
    links = build_links(raw_links, topic, evidence_text)
    trace.append(f"link_candidates={len(raw_links)}")
    target_ids = [i for i in dict.fromkeys(l["target_doc_id"] for l in links) if i not in primary_doc_ids][:5]
    linked_evidence: list[dict] = []
    kept_links: list[dict] = []
    if target_ids:
        t_hits, t_stages = search.search_chunks(db, plan, pool, target_ids, corpus=corpus, restrict_doc_ids=True)
        trace += t_stages
        good = filter_evidence(t_hits, topic, wanted, dropped)
        good_docs = {r["doc_id"] for r in good}
        for l in links:
            t = l["target_doc_id"]
            if t in good_docs or t in primary_doc_ids:
                ids = [r["chunk_id"] for r in good if r["doc_id"] == t][:3]
                kept_links.append({**l, "evidence_chunk_ids": ids})
        for r in good:
            if r["chunk_id"] not in seen:
                linked_evidence.append({**r, "retrieval_methods": list(dict.fromkeys((r.get("retrieval_methods") or []) + ["linked_document"]))})
    else:
        kept_links = [l for l in links if l["target_doc_id"] in primary_doc_ids]
    trace.append(f"cross_document_links={len(kept_links)}")

    reserve = min(len(linked_evidence), max(1, top_k // 4)) if linked_evidence else 0
    evidence = (primary[:max(keep_n - reserve, 1)] + linked_evidence[:reserve])[:max(keep_n, 1)]
    for i, h in enumerate(evidence, 1):
        h["n"] = i
        h["source_id"] = f"S{i}"
    docs = list(dict.fromkeys(h["doc_id"] for h in evidence))
    linked_set = {l["target_doc_id"] for l in kept_links}
    kept_links = [l for l in kept_links if l["target_doc_id"] in docs or l["source_doc_id"] in docs]
    filter_status = "unsupported_by_current_schema" if document_type else "not_requested"
    methods = sorted({m for h in evidence for m in (h.get("retrieval_methods") or [])})
    return {
        "question": question,
        "retrieval_mode": "cross_document_graph",
        "source_documents": [{"doc_id": d, "title": next((h.get("title") for h in evidence if h.get("doc_id") == d), None),
                              "role": "linked" if d in linked_set and d not in primary_doc_ids else "primary"} for d in docs],
        "document_ids": docs,
        "evidence": evidence,
        "citations": [{"doc_id": h.get("doc_id"), "title": h.get("title"), "chunk_id": h.get("chunk_id"),
                       "page": h.get("page"), "section_title": h.get("section_title"), "section_path": h.get("section_path"),
                       "paragraph_index": h.get("paragraph_index"), "file_name": h.get("file_name"),
                       "snippet": (h.get("text") or "")[:500], "score": h.get("score"), "source_id": h["source_id"]}
                      for h in evidence],
        "graph_links": kept_links,
        "retrieval": {"methods": methods, "primary_documents": primary_doc_ids, "linked_documents": sorted(linked_set),
                      "dropped": dropped,
                      "explanation": ("Evidence dipilih dari fulltext + graph (entitas/section) lalu disaring: metadata katalog, "
                                      "header/footer, noise OCR, dan chunk berkoherensi rendah dibuang. Relasi lintas dokumen hanya "
                                      "dari entitas spesifik bersama (bukan tag generik/tahun) yang terbukti relevan.")},
        "trace": trace,
        "filters": {"corpus": corpus, "document_ids": document_ids or [],
                    "document_type": document_type, "document_type_status": filter_status},
        "status": "ok" if evidence else "no_evidence",
    }
