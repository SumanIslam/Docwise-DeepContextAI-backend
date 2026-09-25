"""
Docwise — Step 1 of the embedding comparison: FREEZE the evaluation chunks.

What this script does
---------------------
Runs the *current* Docwise extraction + chunking pipeline ONCE on the 8
evaluation PDFs and writes the result to disk, so that both embedding
configurations (A: all-MiniLM-L6-v2, B: paraphrase-multilingual-MiniLM-L12-v2)
embed exactly the same chunk texts, and relevance labels are made only once.

What it does NOT do
-------------------
* It does not modify main.py.
* It does not import main.py (importing it would load the models, open
  ./qdrant_data and create docwise.db). Instead it reads main.py's source and
  executes ONLY these functions, unchanged:
      _is_unicode_bangla, _is_garbled, _fix_unicode, _ocr_page_image,
      _ocr_embedded_images, _classify_pdf, extract_pages, chunk_text
  The SHA-256 of main.py is recorded, so the frozen output is tied to the
  exact code version that produced it.
* It does not touch Qdrant, SQLite, the uploads folder or the Groq API.

Usage (from the backend folder, where main.py lives)
----------------------------------------------------
    python freeze_chunks.py --pdf-dir eval_pdfs --out eval_frozen

Optional:
    --check-determinism   run extraction a second time and compare (slow: OCRs twice)
    --no-token-stats      skip tokenizer statistics (no model download needed)
    --tesseract-cmd PATH  override the Tesseract binary path
    --force               allow overwriting an existing output folder

Outputs (in --out)
------------------
    chunks.jsonl             one line per chunk: stable chunk_id, doc, page, text, ...
    pages.jsonl              the raw extracted/OCR text of every page
    extraction_report.json   PDF types, counts, timings, environment, checksums
"""

import argparse
import ast
import datetime
import hashlib
import io
import json
import os
import platform
import re
import sys
import time
import unicodedata
from importlib import metadata

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pypdf import PdfReader

DOC_CODES = ["B1", "B2", "U1", "U2", "I1", "I2", "E1", "E2"]
EXPECTED_TYPE = {"B": "bijoy", "U": "unicode_bangla", "I": "image_only", "E": "english"}
FUNCS_FROM_MAIN = [
    "_is_unicode_bangla", "_is_garbled", "_fix_unicode", "_ocr_page_image",
    "_ocr_embedded_images", "_classify_pdf", "_broken_bangla_score",
    "extract_pages", "chunk_text",
]
EMBED_MODELS = {
    "A_baseline": "all-MiniLM-L6-v2",
    "B_multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
}
# Same limits as _process_pdf_background in main.py (inline there, so mirrored here):
MIN_CHUNK_CHARS = 10     # `if len(chunk.strip()) < 10: continue`
MAX_CHUNK_CHARS = 4000   # `chunk_texts.append(chunk[:4000])`


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha1_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def pkg_version(name):
    try:
        return metadata.version(name)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Load main.py's extraction functions WITHOUT running main.py's top level
# ──────────────────────────────────────────────────────────────────────────────
def load_pipeline(main_path, tesseract_cmd):
    import pytesseract

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    elif os.name == "nt":  # same default as main.py
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Same capability checks as main.py — but here a missing piece is fatal,
    # because freezing without OCR or Bengali data would produce invalid chunks.
    import fitz  # noqa: F401  (PyMuPDF)
    import PIL   # noqa: F401
    tess_version = str(pytesseract.get_tesseract_version())
    langs = pytesseract.get_languages()
    if "ben" not in langs:
        sys.exit(f"[freeze] ABORT: Tesseract 'ben' language data not installed (found: {langs}).")

    src = open(main_path, encoding="utf-8").read()
    tree = ast.parse(src)
    wanted = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS_FROM_MAIN]
    found = {n.name for n in wanted}
    missing = [n for n in FUNCS_FROM_MAIN if n not in found]
    if missing:
        sys.exit(f"[freeze] ABORT: functions not found in main.py: {missing}")

    import regex
    namespace = {
        "__name__": "docwise_main_extract",
        "PdfReader": PdfReader, "io": io, "os": os, "re": re, "_regex": regex,
        "unicodedata": unicodedata, "_CAN_OCR": True, "_HAS_BEN_LANG": True,
        # Experiment C switches — taken from the command line, never from .env,
        # so a freeze is always reproducible from the recorded settings.
        "UNICODE_OCR_FALLBACK": True, "BROKEN_TEXT_THRESHOLD": 10.0,
    }
    code = compile(ast.Module(body=wanted, type_ignores=[]), main_path, "exec")
    exec(code, namespace)

    tessdata = {}
    try:
        prefix = os.environ.get("TESSDATA_PREFIX") or os.path.join(
            os.path.dirname(pytesseract.pytesseract.tesseract_cmd), "tessdata")
        ben = os.path.join(prefix, "ben.traineddata")
        if os.path.exists(ben):
            tessdata = {"ben_traineddata_path": ben,
                        "ben_traineddata_bytes": os.path.getsize(ben),
                        "ben_traineddata_sha256": sha256_file(ben)}
    except Exception:
        pass

    env = {
        "tesseract_version": tess_version,
        "tesseract_cmd": pytesseract.pytesseract.tesseract_cmd,
        "tesseract_languages": sorted(langs),
        **tessdata,
    }
    return namespace, env


def find_pdfs(pdf_dir):
    pdfs = {}
    for name in sorted(os.listdir(pdf_dir)):
        if not name.lower().endswith(".pdf") or "evaluation" in name.lower():
            continue
        m = re.search(r"(?<![A-Z0-9])(B1|B2|U1|U2|I1|I2|E1|E2)(?![0-9])", name)
        if m:
            if m.group(1) in pdfs:
                sys.exit(f"[freeze] ABORT: two PDFs for {m.group(1)}: {pdfs[m.group(1)]} and {name}")
            pdfs[m.group(1)] = os.path.join(pdf_dir, name)
    missing = [d for d in DOC_CODES if d not in pdfs]
    if missing:
        sys.exit(f"[freeze] ABORT: no PDF found for {missing} in {pdf_dir}")
    return pdfs


def extract_all(ns, pdfs):
    """Returns {doc: {"pdf_type", "pages": [{"page", "text"}], "seconds"}}"""
    out = {}
    for doc in DOC_CODES:
        path = pdfs[doc]
        t0 = time.perf_counter()
        pdf_type = ns["_classify_pdf"](path)
        pages = ns["extract_pages"](path)   # classifies again internally — same result
        out[doc] = {"pdf_type": pdf_type, "pages": pages,
                    "seconds": round(time.perf_counter() - t0, 2)}
    return out


def build_chunks(ns, doc, pages):
    chunks = []
    for item in pages:
        idx = 0
        for chunk in ns["chunk_text"](item["text"]):
            if len(chunk.strip()) < MIN_CHUNK_CHARS:
                continue
            text = chunk[:MAX_CHUNK_CHARS]
            idx += 1
            chunks.append({
                "chunk_id": f"{doc}-p{item['page']:02d}-c{idx:02d}",
                "doc_code": doc,
                "page": item["page"],
                "chunk_in_page": idx,
                "char_len": len(text),
                "text_sha1": sha1_text(text),
                "text": text,
            })
    return chunks


def token_stats(chunks):
    from sentence_transformers import SentenceTransformer
    stats = {}
    for key, model_name in EMBED_MODELS.items():
        m = SentenceTransformer(model_name)
        max_len = int(m.max_seq_length)
        tok = m.tokenizer
        truncated = 0
        for c in chunks:
            n = len(tok(c["text"], add_special_tokens=True, truncation=False)["input_ids"])
            c.setdefault("tokens", {})[key] = n
            if n > max_len:
                truncated += 1
        stats[key] = {"model": model_name, "max_seq_length": max_len,
                      "chunks_truncated": truncated, "chunks_total": len(chunks)}
        del m
    return stats


def main():
    ap = argparse.ArgumentParser(description="Freeze Docwise evaluation chunks.")
    ap.add_argument("--main", default="main.py")
    ap.add_argument("--pdf-dir", default="eval_pdfs")
    ap.add_argument("--out", default="eval_frozen")
    ap.add_argument("--tesseract-cmd", default=None)
    ap.add_argument("--check-determinism", action="store_true")
    ap.add_argument("--no-token-stats", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-ocr-fallback", action="store_true",
                    help="reproduce the original extraction: keep broken Bengali text layers")
    ap.add_argument("--broken-threshold", type=float, default=10.0,
                    help="Bengali orthography errors per 1000 characters above which a page is re-read with OCR")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.out, "chunks.jsonl")) and not args.force:
        sys.exit(f"[freeze] ABORT: {args.out} already contains frozen chunks. "
                 f"Frozen data must not be overwritten silently; use --force deliberately.")
    os.makedirs(args.out, exist_ok=True)

    ns, tess_env = load_pipeline(args.main, args.tesseract_cmd)
    ns["UNICODE_OCR_FALLBACK"] = not args.no_ocr_fallback
    ns["BROKEN_TEXT_THRESHOLD"] = args.broken_threshold
    pdfs = find_pdfs(args.pdf_dir)
    print(f"[freeze] main.py sha256 = {sha256_file(args.main)[:16]}...")
    print(f"[freeze] Tesseract {tess_env['tesseract_version']}  langs={tess_env['tesseract_languages']}")

    run1 = extract_all(ns, pdfs)

    determinism = None
    if args.check_determinism:
        print("[freeze] second extraction run for determinism check ...")
        run2 = extract_all(ns, pdfs)
        determinism = {}
        for doc in DOC_CODES:
            a = [(p["page"], sha1_text(p["text"])) for p in run1[doc]["pages"]]
            b = [(p["page"], sha1_text(p["text"])) for p in run2[doc]["pages"]]
            determinism[doc] = (a == b) and run1[doc]["pdf_type"] == run2[doc]["pdf_type"]

    all_chunks, per_doc = [], {}
    with open(os.path.join(args.out, "pages.jsonl"), "w", encoding="utf-8") as fp:
        for doc in DOC_CODES:
            r = run1[doc]
            for p in r["pages"]:
                fp.write(json.dumps({"doc_code": doc, "page": p["page"], "pdf_type": r["pdf_type"],
                                     "char_len": len(p["text"]), "text_sha1": sha1_text(p["text"]),
                                     "text": p["text"]}, ensure_ascii=False) + "\n")
            chunks = build_chunks(ns, doc, r["pages"])
            for c in chunks:
                c["pdf_type"] = r["pdf_type"]
                c["source_file"] = os.path.basename(pdfs[doc])
            all_chunks.extend(chunks)
            total_pages = len(PdfReader(pdfs[doc]).pages)
            expected = EXPECTED_TYPE[doc[0]]
            per_doc[doc] = {
                "source_file": os.path.basename(pdfs[doc]),
                "pdf_sha256": sha256_file(pdfs[doc]),
                "pdf_type_detected": r["pdf_type"],
                "pdf_type_expected": expected,
                "type_matches_expected": r["pdf_type"] == expected,
                "pages_total": total_pages,
                "pages_with_text": len(r["pages"]),
                "chunks": len(chunks),
                "extract_seconds": r["seconds"],
            }

    tstats = None
    if not args.no_token_stats:
        print("[freeze] computing token statistics for both embedding models ...")
        tstats = token_stats(all_chunks)

    ids = [c["chunk_id"] for c in all_chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk_id"
    with open(os.path.join(args.out, "chunks.jsonl"), "w", encoding="utf-8") as fc:
        for c in all_chunks:
            fc.write(json.dumps(c, ensure_ascii=False) + "\n")

    report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Frozen chunk set for the Docwise embedding-model comparison (A vs B).",
        "pipeline": {
            "main_py_sha256": sha256_file(args.main),
            "functions_used_from_main_py": FUNCS_FROM_MAIN,
            "chunk_size": 700, "chunk_overlap": 120,
            "unicode_ocr_fallback": not args.no_ocr_fallback,
            "broken_text_threshold": args.broken_threshold,
            "min_chunk_chars": MIN_CHUNK_CHARS, "max_chunk_chars": MAX_CHUNK_CHARS,
        },
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            **{p: pkg_version(p) for p in ["pypdf", "PyMuPDF", "pytesseract", "Pillow", "ftfy",
                                            "sentence-transformers", "transformers", "torch",
                                            "qdrant-client", "rank-bm25"]},
            **tess_env,
        },
        "documents": per_doc,
        "total_chunks": len(all_chunks),
        "all_types_match_expected": all(d["type_matches_expected"] for d in per_doc.values()),
        "determinism_check": determinism,
        "token_stats": tstats,
        "output_sha256": {
            "chunks.jsonl": sha256_file(os.path.join(args.out, "chunks.jsonl")),
            "pages.jsonl": sha256_file(os.path.join(args.out, "pages.jsonl")),
        },
    }
    with open(os.path.join(args.out, "extraction_report.json"), "w", encoding="utf-8") as fr:
        json.dump(report, fr, ensure_ascii=False, indent=2)

    # ── ASCII-only console summary (safe on any Windows console) ──
    print("\n doc  detected         expected         ok   pages  chunks  sec")
    for doc in DOC_CODES:
        d = per_doc[doc]
        print(f" {doc}   {d['pdf_type_detected']:<16} {d['pdf_type_expected']:<16} "
              f"{'yes' if d['type_matches_expected'] else 'NO ':<4} "
              f"{d['pages_with_text']}/{d['pages_total']:<4} {d['chunks']:<7} {d['extract_seconds']}")
    print(f"\n total chunks: {len(all_chunks)}  (hybrid_search pool limit is 200)")
    if len(all_chunks) > 200:
        print(" WARNING: more than 200 chunks — the BM25 pool would depend on the dense model.")
    if not report["all_types_match_expected"]:
        print(" WARNING: at least one PDF was classified differently from its intended condition.")
    if determinism is not None:
        print(f" determinism: {determinism}")
    if tstats:
        for k, s in tstats.items():
            print(f" {k}: max_seq_length={s['max_seq_length']}  truncated {s['chunks_truncated']}/{s['chunks_total']}")
    print(f"\n written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()