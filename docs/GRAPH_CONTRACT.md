# Kontrak Graf ARCANA Bappenas (sesuai spesifikasi node & relasi)

```
(:Regulation/:Reference) <-[:USES_REFERENCE]- (:Document) -[:PUBLISHED_BY]-> (:Organization)
(:Document) -[:PRODUCED_BY]-> (:UnitKerja UKE_2) -[:PART_OF]-> (:UnitKerja UKE_1) -[:BELONGS_TO_ORG]-> (:Organization)
(:Document) -[:UNDER_DIVISION]-> (:UnitKerja UKE_1)       (:Document) -[:HAS_PARTNER]-> (:UnitKerja|:Organization)
(:Document) -[:CONTAINS_STEP]-> (:ProcessStep)             (:Document) -[:TAGGED_WITH]-> (:Tag) <-[:TAGGED_WITH]- (:Chunk)
(:Document) -[:IN_CATEGORY]-> (:Category) <-[:SUBCATEGORY_OF]- (:SubCategory) <-[:IN_SUB_CATEGORY]- (:Document)
(:Document) -[:LOCATED_IN]-> (:Location)

(:Document) -[:CONTAINS_CHUNK {sequence}]-> (:Chunk root/group) <-[:SUB_CHUNK_OF {order, level_diff:1}]- (:Chunk section/group/paragraph) <- ...
(:Chunk) -[:REFERENCES_CHUNK {similarity, type}]-> (:Chunk)      lintas dokumen: semantic_similarity | policy_evidence

(:Question) -[:ANSWERED_BY]-> (:Answer) -[:CITES]-> (:Chunk);  (:Question) -[:ASKED_ABOUT]-> (:Document)   -- dataset seed
```

## Paragraph leaves dan hierarki
- Setiap paragraf sumber, butir daftar, atau baris tabel menjadi satu chunk daun. Daun tidak digabung, dipotong, atau dibuang oleh batas kapasitas chunk.
- Heading, section, dan fanout grouping menjadi chunk parent struktural; parent terhubung rekursif lewat `SUB_CHUNK_OF` dan tidak masuk semantic retrieval/citation.
- `CHUNK_FANOUT` membatasi jumlah anak langsung per parent. Section besar membentuk lapisan group tambahan sampai semua paragraf terhubung ke root dokumen.
- Daun memiliki `chunk_index` global dan `paragraph_index` global untuk urutan; `next_chunk_id` menunjuk sibling berikutnya, sedangkan `SUB_CHUNK_OF` menghubungkan hierarki. `section_title`, `section_path`, `parent_section_id`, `page_number`, `item_no`, dan `source_reference` menjaga provenance.
- ID daun deterministik `DOC-001_p0000001`; ID parent deterministik dari urutan pembuatan.
- Pertanyaan daftar memperluas hasil relevan ke semua daun dalam section yang terpilih; urutan konteks mengikuti `chunk_index`.
- `Document.corpus` dan `Chunk.corpus` membedakan corpus `bappenas` dari `notetaker` tanpa memisahkan graph Neo4j.

## Sumber isi
1. Metadata katalog (selalu): Ringkasan, Manfaat, Proses Bisnis (per langkah), Regulasi, Data Sekunder, Mitra, Kata Kunci → chunk dengan `section` baku (dipakai sitasi).
2. Berkas fisik Bappenas (bila ada di `data/bappenas/documents/`, dicocokkan dengan kolom *Nama File*): PDF/DOCX/PPTX/XLSX/CSV/TXT/MD → paragraph chunks dengan hierarki heading asli di metadata. Markdown notetaker berada di `data/notetaker/markdown/`; metadata JSON pasangannya berada di `data/notetaker/json/`.

## Ekstensi di luar spesifikasi (minimal)
`SUBCATEGORY_OF` (SubCategory→Category), label `Question/Answer` + `CITES/ANSWERED_BY/ASKED_ABOUT` untuk dataset seed, properti `Document.has_physical_file`.
