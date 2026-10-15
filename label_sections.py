"""
Docwise — Step 2 of the embedding comparison: label every frozen chunk with the
ground-truth section(s) it contains.

Why
---
The 200 questions name *sections* as ground truth, but retrieval returns *chunks*.
This script decides, once and before any experiment runs, which section(s) each
chunk belongs to. The labels depend only on the frozen chunk text and the source
section text — never on an embedding model — so they apply identically to
Configuration A and Configuration B.

Method (deterministic, no ML)
-----------------------------
1. Rebuild each page's extracted text from its overlapping chunks
   (chunk_text uses 700-char windows with a 580-char step), which gives every
   chunk an exact character span in the extracted document.
2. Build the source document from sections.jsonl (heading + body, in order),
   remembering which section every source character belongs to.
3. Align extracted text to source text (difflib.SequenceMatcher, whitespace and
   zero-width joiners removed). Characters that align map directly to a source
   position; characters that don't (OCR/text-layer errors) are placed by linear
   interpolation between the nearest aligned characters. Order is preserved, so
   this is robust even for noisy text.
4. For each chunk, count how many of its characters fall in each section.
   A chunk gets a section label if that section covers
       >= MIN_SHARE of the chunk's characters   OR   >= MIN_CHARS characters.
   The section with the largest share is always included.
5. Chunks are flagged for manual review if alignment coverage is low or a
   section's share lies close to the threshold.

These thresholds are fixed here, before any retrieval result exists.

Usage (from the backend folder)
-------------------------------
    python label_sections.py
    python label_sections.py --chunks evaluation/frozen/chunks.jsonl \
        --sections evaluation/dataset/sections.jsonl \
        --questions evaluation/dataset/questions.jsonl --out evaluation/labels
"""

import argparse
import difflib
import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict

CHUNK_SIZE, OVERLAP = 700, 120
STEP = CHUNK_SIZE - OVERLAP
MIN_SHARE = 0.30          # section covers >= 30% of the chunk ...
MIN_CHARS = 150           # ... or >= 150 non-whitespace characters of it
REVIEW_COVERAGE = 0.50    # flag if < 50% of the chunk aligned directly
REVIEW_BAND = (0.15, 0.40)  # flag if any section share falls in this band
DOCS = ["B1", "B2", "U1", "U2", "I1", "I2", "E1", "E2"]
DROP = {"\u200c", "\u200d", "\ufeff"}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def keep(ch):
    return not ch.isspace() and ch not in DROP


def squeeze(text):
    """NFC, then drop whitespace/joiners; returns (string, index map to original)."""
    text = unicodedata.normalize("NFC", text)
    out, idx = [], []
    for i, ch in enumerate(text):
        if keep(ch):
            out.append(ch)
            idx.append(i)
    return "".join(out), idx


def rebuild_pages(chunks):
    """(doc, page) -> page text, and chunk_id -> (start, end) within that page."""
    by_page = defaultdict(list)
    for c in chunks:
        by_page[(c["doc_code"], c["page"])].append(c)
    pages, spans = {}, {}
    for key, cs in by_page.items():
        cs.sort(key=lambda c: c["chunk_in_page"])
        text = ""
        for k, c in enumerate(cs):
            start = k * STEP
            if k > 0 and text[start:start + OVERLAP] != c["text"][:OVERLAP]:
                raise SystemExit(f"overlap mismatch at {c['chunk_id']} — chunks are not from chunk_text()")
            text = text[:start] + c["text"]
            spans[c["chunk_id"]] = (start, start + len(c["text"]))
        pages[key] = text
    return pages, spans


def label_doc(doc, chunks, pages, spans, sections):
    doc_pages = sorted(p for (d, p) in pages if d == doc)
    # extracted document text with page offsets
    ext, page_off = "", {}
    for p in doc_pages:
        page_off[p] = len(ext)
        ext += pages[(doc, p)] + "\n"
    # source document with a section id per character
    src, sec_of = "", []
    secs = sorted((s for s in sections if s["doc_code"] == doc), key=lambda s: s["section_index"])
    for s in secs:
        block = unicodedata.normalize("NFC", s["section"] + "\n" + s["body_text"] + "\n")
        src += block
        sec_of += [s["section_index"]] * len(block)
    names = {s["section_index"]: s["section"] for s in secs}

    e_sq, e_idx = squeeze(ext)          # squeezed extracted -> positions in ext
    s_sq, s_idx = squeeze(src)          # squeezed source    -> positions in src
    ext_pos_to_sq = {pos: k for k, pos in enumerate(e_idx)}

    sm = difflib.SequenceMatcher(None, e_sq, s_sq, autojunk=False)
    mapped = [None] * len(e_sq)
    for a, b, n in sm.get_matching_blocks():
        for t in range(n):
            mapped[a + t] = b + t
    anchors = [i for i, v in enumerate(mapped) if v is not None]
    direct = [v is not None for v in mapped]
    # interpolate unaligned characters between neighbouring anchors
    if not anchors:
        raise SystemExit(f"{doc}: no alignment at all")
    first, last = anchors[0], anchors[-1]
    for i in range(0, first):
        mapped[i] = mapped[first]
    for i in range(last + 1, len(mapped)):
        mapped[i] = mapped[last]
    for a0, a1 in zip(anchors, anchors[1:]):
        if a1 - a0 > 1:
            v0, v1 = mapped[a0], mapped[a1]
            for i in range(a0 + 1, a1):
                mapped[i] = round(v0 + (v1 - v0) * (i - a0) / (a1 - a0))

    doc_ratio = round(sm.ratio(), 3)
    out = []
    for c in chunks:
        if c["doc_code"] != doc:
            continue
        s0, s1 = spans[c["chunk_id"]]
        off = page_off[c["page"]]
        sq_positions = [ext_pos_to_sq[p] for p in range(off + s0, off + s1) if p in ext_pos_to_sq]
        counts = Counter(sec_of[s_idx[mapped[k]]] for k in sq_positions)
        total = sum(counts.values()) or 1
        shares = {names[i]: round(n / total, 3) for i, n in counts.most_common()}
        chars = {names[i]: n for i, n in counts.items()}
        top = max(shares, key=shares.get)
        labels = [name for name, sh in shares.items() if sh >= MIN_SHARE or chars[name] >= MIN_CHARS]
        if top not in labels:
            labels.insert(0, top)
        coverage = round(sum(direct[k] for k in sq_positions) / total, 3)
        reasons = []
        if coverage < REVIEW_COVERAGE:
            reasons.append(f"low alignment ({coverage:.2f})")
        border = [n for n, sh in shares.items() if REVIEW_BAND[0] <= sh < REVIEW_BAND[1]]
        if border:
            reasons.append("near threshold: " + "; ".join(border))
        out.append({
            "chunk_id": c["chunk_id"], "doc_code": doc, "page": c["page"],
            "auto_labels": labels, "section_shares": shares,
            "alignment_coverage": coverage, "needs_review": bool(reasons),
            "review_reason": " | ".join(reasons), "text": c["text"],
        })
    return out, doc_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="evaluation/frozen/chunks.jsonl")
    ap.add_argument("--sections", default="evaluation/dataset/sections.jsonl")
    ap.add_argument("--questions", default="evaluation/dataset/questions.jsonl")
    ap.add_argument("--out", default="evaluation/labels")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    chunks = read_jsonl(args.chunks)
    sections = read_jsonl(args.sections)
    questions = read_jsonl(args.questions)
    pages, spans = rebuild_pages(chunks)

    labels, doc_ratio = [], {}
    for d in DOCS:
        rows, ratio = label_doc(d, chunks, pages, spans, sections)
        labels += rows
        doc_ratio[d] = ratio

    with open(os.path.join(args.out, "chunk_labels_auto.jsonl"), "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps({k: v for k, v in r.items() if k != "text"}, ensure_ascii=False) + "\n")

    # provisional relevance check against the 200 questions
    by_doc_label = defaultdict(set)
    for r in labels:
        for s in r["auto_labels"]:
            by_doc_label[(r["doc_code"], s)].add(r["chunk_id"])
    q_stats, zero, partial = [], [], []
    for q in questions:
        per_sec = {s: len(by_doc_label[(q["doc_code"], s)]) for s in q["gt_section_list"]}
        rel = set().union(*(by_doc_label[(q["doc_code"], s)] for s in q["gt_section_list"]))
        q_stats.append(len(rel))
        if not rel:
            zero.append(q["question_id"])
        elif any(n == 0 for n in per_sec.values()):
            partial.append(q["question_id"])

    report = {
        "rule": {"min_share": MIN_SHARE, "min_chars": MIN_CHARS,
                 "review_if_coverage_below": REVIEW_COVERAGE, "review_if_share_in": REVIEW_BAND},
        "inputs_sha256": {"chunks": sha256(args.chunks), "sections": sha256(args.sections),
                          "questions": sha256(args.questions)},
        "doc_alignment_ratio": doc_ratio,
        "chunks": len(labels),
        "chunks_with_2plus_labels": sum(len(r["auto_labels"]) > 1 for r in labels),
        "chunks_needing_review": sum(r["needs_review"] for r in labels),
        "needs_review_by_doc": dict(Counter(r["doc_code"] for r in labels if r["needs_review"])),
        "relevant_chunks_per_question": {"min": min(q_stats), "max": max(q_stats),
                                         "mean": round(sum(q_stats) / len(q_stats), 2)},
        "questions_with_no_relevant_chunk": zero,
        "questions_with_a_gt_section_missing": partial,
    }
    with open(os.path.join(args.out, "label_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return labels, report


if __name__ == "__main__":
    main()
