"""Simulasi end-to-end OFFLINE (tanpa Neo4j/Groq) atas PDF sebenarnya: ekstraksi -> chunking -> resolusi dokumen ->
retrieval -> context assembly -> prompt. Dipakai untuk bukti sebelum/sesudah perbaikan.

Pakai:  python scripts/simulate_retrieval.py [--root <repo>] [--question "..."]
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
ap.add_argument("--question", default="Apa saja identifikasi masalah dari Naskah Akademik RUU Nomor 3 Tahun 2022?")
ap.add_argument("--file", default="B.03*")
ap.add_argument("--legacy-data", default="", help="root repo LAMA: chunk dibangun dgn ekstraktor/chunker lama = data yang sudah ada di Neo4j (tanpa ingest ulang)")
a = ap.parse_args()
ROOT = Path(a.root)
sys.path.insert(0, str(ROOT))

from kb import search, retrieval, rag                     # noqa: E402
from kb.extract import extract_blocks                      # noqa: E402
from kb.chunker import build_chunks                        # noqa: E402

meta = json.load(open(ROOT / "data/bappenas/metadata_bappenas.json"))["metadata_bappenas"]
catalog, DOC = [], None
def _walk(o):
    if isinstance(o, dict):
        if o.get("Judul"):
            yield o
        else:
            for v in o.values():
                yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


for i, m in enumerate(_walk(meta)):
    row = {"doc_id": f"D{i:02d}", "title": m["Judul"], "year": m.get("Tahun Penyusunan"), "description": m.get("Deskripsi"),
           "benefit": None, "url": None, "file_name": m.get("Nama File"), "has_file": True, "category": None, "subcategory": None,
           "output_type": None, "knowledge_type": None, "access_rights": None, "publish_date": None,
           "tags_text": re.sub(r"_x000D_|\n", " ", str(m.get("Tag") or "")), "corpus": "bappenas"}
    catalog.append(row)
    if "Naskah Akademik" in m["Judul"] and "Ibu Kota" in m["Judul"]:
        DOC = row
pdf = list((ROOT / "data/bappenas/documents").glob(a.file))[0]
if a.legacy_data:
    import importlib.util
    def _load(name, path):
        pkg = importlib.util.spec_from_file_location("legacy_kb", Path(a.legacy_data) / "kb/__init__.py", submodule_search_locations=[str(Path(a.legacy_data) / "kb")])
        m = importlib.util.module_from_spec(pkg); sys.modules["legacy_kb"] = m; pkg.loader.exec_module(m)
        sp = importlib.util.spec_from_file_location(f"legacy_kb.{name}", path); mod = importlib.util.module_from_spec(sp)
        sys.modules[f"legacy_kb.{name}"] = mod; sp.loader.exec_module(mod); return mod
    _ex = _load("extract", Path(a.legacy_data) / "kb/extract.py"); _ch = _load("chunker", Path(a.legacy_data) / "kb/chunker.py")
    chunks = _ch.build_chunks(DOC["doc_id"], DOC["title"], _ex.extract_blocks(pdf))
else:
    chunks = build_chunks(DOC["doc_id"], DOC["title"], extract_blocks(pdf))
leaves = [c for c in chunks if c.is_leaf]


def row(c):
    d = {k: getattr(c, k, None) for k in ("chunk_id", "doc_id", "section", "section_title", "section_path", "parent_section_id",
         "paragraph_index", "source_reference", "item_no", "text", "level", "is_leaf", "breadcrumb", "chunk_index",
         "list_group_id", "lead_in_id", "list_item_pos", "list_size", "list_context")}
    d.update(page=c.page, title=DOC["title"], url=None, file_name=DOC["file_name"])
    return d


class FakeDB:
    def read(self, q, **p):
        if "RETURN DISTINCT c.doc_id AS doc_id, c.section_title" in q:
            seen, out = set(), []
            for c in leaves:
                if c.section_path and c.section_path not in seen:
                    seen.add(c.section_path); out.append({"doc_id": c.doc_id, "section_title": c.section_title,
                                                         "section_path": c.section_path, "parent_section_id": c.parent_section_id, "title": DOC["title"]})
            return out
        if "$section_paths" in q or "section_paths" in p:
            return [row(c) for c in leaves if any(c.section_path == s or c.section_path.startswith(s + " > ") for s in p["section_paths"])]
        if "$windows" in q or "windows" in p:
            return [row(c) for c in leaves if any(c.doc_id == w["doc_id"] and w["lo"] <= c.paragraph_index <= w["hi"] for w in p["windows"])]
        if "toLower(c.text) CONTAINS t" in q:           # CHUNK_LEXICAL
            out = []
            for c in leaves:
                hit = [t for t in p["terms"] if t in c.text.lower()]
                if hit:
                    out.append({**row(c), "score": float(len(hit))})
            return sorted(out, key=lambda r: -r["score"])[:p["limit"]]
        return []


# patch hanya lapisan Neo4j: full-text -> leksikal sederhana; katalog dokumen dari metadata
def fake_ft(db, kind, cypher, q, **p):
    if kind != "chunk":
        return []
    ws = {w for w in re.findall(r"[a-z0-9]{4,}", q.lower())}
    out = []
    for c in leaves:
        hit = [w for w in ws if w in c.text.lower() or w in c.breadcrumb.lower() or w in (getattr(c, "list_context", "") or "").lower()]
        if hit:
            out.append({**row(c), "score": float(len(hit))})
    return sorted(out, key=lambda r: -r["score"])[:p.get("limit", 40)]


search._ft = fake_ft
search.doc_catalog = lambda db, corpus=None, ttl=600: catalog
db = FakeDB()

res = retrieval.answer(db, a.question, top_k=5)
ev = res["evidence"]
print("== PLAN ==")
print(json.dumps({k: res["plan"].get(k) for k in ("resolved", "intents", "trace")}, ensure_ascii=False, indent=1))
print("source:", res["source"], "| evidence:", len(ev))
for e in ev[:12]:
    print(f'  [{e["n"]}] {e["chunk_id"]} | {e.get("section_path")} | pos={e.get("list_item_pos")} | {e["text"][:70]!r}')

sel = rag.select_context(ev, question=a.question)
payload = rag.build_prompt(a.question, sel)
print("== CONTEXT FINAL ==", len(sel), "chunk,", len(payload.context_text), "karakter")
txt = payload.context_text
for k in range(1, 5):
    print(f"  butir {k} ada di context:", bool(re.search(rf"\n?ISI SUMBER: {k}\. ", txt)))
print("  pengantar 'Ada empat pokok masalah' ada:", "Ada empat pokok masalah" in txt)
print("  dokumen benar:", all(e["doc_id"] == DOC["doc_id"] for e in sel) and bool(sel))
if "--show" in sys.argv:
    print(payload.context_text)
