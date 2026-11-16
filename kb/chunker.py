"""Hierarchical chunk tree with source paragraphs as the only retrievable leaves."""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    parent_id: str | None
    level: int
    chunk_index: int
    text: str
    breadcrumb: str
    page: int | None
    section: str | None
    item_no: int | str | None
    is_leaf: bool
    token_count: int
    next_id: str | None = None
    document_title: str = ""
    paragraph_index: int = 0
    section_title: str | None = None
    section_path: str = ""
    parent_section_id: str | None = None
    source_reference: str = ""
    # --- relasi daftar bernomor (pengantar <-> butir) ---
    list_group_id: str | None = None      # sama untuk pengantar + semua butirnya
    lead_in_id: str | None = None         # chunk_id paragraf pengantar (untuk butir)
    list_item_pos: int = 0                # 0 = pengantar, 1..n = urutan butir
    list_size: int = 0                    # jumlah butir pada daftar
    list_context: str = ""                # judul bagian + pengantar (menjadi bagian teks yang diindeks)


@dataclass
class _Tree:
    heading: str = ""
    level: int = 0
    paragraphs: list[dict] = field(default_factory=list)
    children: list["_Tree"] = field(default_factory=list)


def lead(text: str, n: int = 350) -> str:
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    cut = text[:n]
    m = max(cut.rfind(". "), cut.rfind("; "))
    return cut[:m + 1] if m > n * 0.5 else cut[:max(cut.rfind(" "), 1)] + " …"


def _section_id(doc_id: str, path: list[str]) -> str | None:
    if not path:
        return None
    slug = re.sub(r"[^\w-]+", "_", "/".join(path), flags=re.UNICODE).strip("_")
    return f"{doc_id}_section_{slug[:120]}"


def _paragraph_item(text: str, explicit):
    if explicit is not None:
        return explicit
    match = re.match(r"^\s*(?:[-•●▪*]|(\d+[.)])|([a-zA-Z][.)]))\s+", text)
    return next((g for g in match.groups() if g), None) if match else None


_LEAD_IN = re.compile(r"(:\s*$|\b(yaitu|yakni|sebagai berikut|berikut|antara lain|meliputi|terdiri atas|terdiri dari)\s*[:.]?\s*$)", re.I)


def _is_item(p: dict) -> bool:
    return _paragraph_item(p["text"], p.get("item")) is not None


def annotate_lists(paragraphs: list[dict]) -> None:
    """Tandai paragraf pengantar dan butir daftar berurutan (in-place, generik untuk semua dokumen).

    Pengantar = paragraf non-butir yang berakhir ':' / 'yaitu' / 'sebagai berikut' dan langsung diikuti butir.
    Butir tanpa pengantar eksplisit tetap dikelompokkan (run butir berurutan) agar urutan/jumlahnya terjaga.
    """
    n, i, gid = len(paragraphs), 0, 0
    while i < n:
        p = paragraphs[i]
        lead = None
        j = i
        if not _is_item(p) and _LEAD_IN.search(p["text"].strip()) and i + 1 < n and _is_item(paragraphs[i + 1]):
            lead, j = i, i + 1
        elif _is_item(p):
            j = i
        else:
            i += 1
            continue
        k = j
        while k < n and _is_item(paragraphs[k]) and paragraphs[k].get("kind", "p") == "p":
            k += 1
        if k == j:
            i += 1
            continue
        gid += 1
        key = f"g{gid}"
        if lead is not None:
            paragraphs[lead].update(_lg=key, _lpos=0, _lsize=k - j, _lead=True)
        for pos, idx in enumerate(range(j, k), 1):
            paragraphs[idx].update(_lg=key, _lpos=pos, _lsize=k - j, _lead=False,
                                   _lead_text=paragraphs[lead]["text"] if lead is not None else "")
        i = k


def build_chunks(doc_id: str, title: str, blocks: list[dict], max_chars: int = 1200,
                 fanout: int = 8, max_chunks: int = 3000) -> list[Chunk]:
    """Build section/group parents recursively and one intact leaf per source paragraph.

    ``max_chars`` and ``max_chunks`` never merge, split, or discard source paragraphs.
    ``fanout`` controls only how many direct child nodes each hierarchy node holds.
    Parent nodes carry structure; only paragraph leaves are sent to semantic retrieval.
    """
    del max_chars, max_chunks  # kept for call-site compatibility; paragraphs are never truncated
    fan = max(2, int(fanout))
    root = _Tree()
    stack: list[_Tree] = [root]
    for block in blocks:
        if block.get("kind") == "h":
            level = int(block.get("level") or 1)
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()
            node = _Tree((block.get("text") or "").strip(), level)
            stack[-1].children.append(node)
            stack.append(node)
        else:
            text = (block.get("text") or "").strip()
            if text:
                stack[-1].paragraphs.append({**block, "text": text})

    out: list[Chunk] = []
    paragraph_index = 0
    lead_ids: dict[str, str] = {}

    def emit(node: _Tree, parent: Chunk | None, path: list[str]) -> Chunk:
        nonlocal paragraph_index
        title_path = path + ([node.heading] if node.heading else [])
        heading = node.heading or ("Bagian Awal" if parent is None else "Kelompok isi")
        cid = f"{doc_id}_c{len(out):06d}"
        chunk = Chunk(
            chunk_id=cid, doc_id=doc_id, parent_id=parent.chunk_id if parent else None,
            level=(parent.level + 1) if parent else 0, chunk_index=len(out),
            text=heading, breadcrumb=" > ".join([title, *title_path]), page=None,
            section=heading if node.heading else (parent.section if parent else None), item_no=None,
            is_leaf=False, token_count=int(len(re.findall(r"\w+", heading)) * 1.3),
            document_title=title, section_title=title_path[-1] if title_path else None,
            section_path=" > ".join(title_path), parent_section_id=_section_id(doc_id, title_path),
            source_reference=title,
        )
        out.append(chunk)
        annotate_lists(node.paragraphs)
        entries: list[tuple[str, object]] = [("p", p) for p in node.paragraphs]
        entries += [("s", child) for child in node.children]
        _emit_entries(entries, chunk, title_path)
        return chunk

    def _emit_entries(entries, parent, path):
        nonlocal paragraph_index
        # Entries in an overflow group are recursively partitioned, retaining the
        # section nodes and paragraph leaves as their original typed children.
        if len(entries) > fan:
            for i in range(0, len(entries), fan):
                subset = entries[i:i + fan]
                g = Chunk(
                    chunk_id=f"{doc_id}_c{len(out):06d}", doc_id=doc_id,
                    parent_id=parent.chunk_id, level=parent.level + 1,
                    chunk_index=len(out), text=f"Kelompok {i // fan + 1}",
                    breadcrumb=parent.breadcrumb, page=None, section=parent.section,
                    item_no=None, is_leaf=False, token_count=2, document_title=title,
                    section_title=parent.section_title, section_path=parent.section_path,
                    parent_section_id=parent.parent_section_id, source_reference=title,
                )
                out.append(g)
                _emit_entries(subset, g, path)
            return
        for kind, value in entries:
            if kind == "p":
                p = value
                paragraph_index += 1
                ptext, page = p["text"], p.get("page")
                cid = f"{doc_id}_p{paragraph_index:07d}"
                lg = p.get("_lg")
                group_id = f"{doc_id}_list_{parent.chunk_id}_{lg}" if lg else None
                if group_id and p.get("_lead"):
                    lead_ids[group_id] = cid
                heading_ctx = parent.section_title or ""
                lead_text = p.get("_lead_text") or (ptext if p.get("_lead") else "")
                list_ctx = " ".join(x for x in (heading_ctx, lead_text) if x).strip() if group_id else ""
                out.append(Chunk(
                    chunk_id=cid, doc_id=doc_id,
                    parent_id=parent.chunk_id, level=parent.level + 1,
                    chunk_index=len(out), text=ptext, breadcrumb=parent.breadcrumb,
                    page=page, section=p.get("section") or parent.section,
                    item_no=_paragraph_item(ptext, p.get("item")), is_leaf=True,
                    token_count=int(len(re.findall(r"\w+", ptext)) * 1.3),
                    document_title=title, paragraph_index=paragraph_index,
                    section_title=parent.section_title, section_path=parent.section_path,
                    parent_section_id=parent.parent_section_id,
                    source_reference=f"{title}" + (f", halaman {page}" if page is not None else ""),
                    list_group_id=group_id, lead_in_id=lead_ids.get(group_id) if group_id and not p.get("_lead") else None,
                    list_item_pos=int(p.get("_lpos") or 0), list_size=int(p.get("_lsize") or 0),
                    list_context=list_ctx,
                ))
            elif kind == "s":
                emit(value, parent, path)

    emit(root, None, [])

    siblings: dict[str | None, list[Chunk]] = {}
    for chunk in out:
        siblings.setdefault(chunk.parent_id, []).append(chunk)
    for group in siblings.values():
        for current, following in zip(group, group[1:]):
            current.next_id = following.chunk_id
    return out
