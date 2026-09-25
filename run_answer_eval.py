"""
Docwise — answer-quality evaluation (step 1 of 2: generate answers to grade).

The retrieval experiments measure whether a relevant passage is FOUND. This
script measures what the user actually sees: it runs the DEPLOYED pipeline
end to end (multilingual-e5-small embeddings, dense-only retrieval, no
reranker, OCR-repaired chunks), sends the retrieved context to the same LLM
with the same prompt as the app, and writes every answer to a workbook for
manual grading.

Sample
------
48 questions: 6 per document, one of each question type, chosen deterministically
(the first question of each type by ID), so all four document conditions, both
query languages and all six question types are represented.

Output (in --out)
-----------------
    answers.jsonl        question, retrieved chunks, context, answer, timings
    grading.xlsx         one row per question, with empty columns to fill in
    run_manifest.json    model, prompt, settings, checksums

Then grade grading.xlsx by hand and run:
    python score_answers.py --workbook evaluation/answers/<run>/grading.xlsx

Requirements
------------
    pip install openpyxl          (grading workbook only)
    .env must contain GROQ_API_KEY

Usage (from the backend folder)
-------------------------------
    python run_answer_eval.py
"""

import argparse
import datetime
import json
import os
import platform
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from run_experiment import (CONFIGS, DOCS, EVAL_CID, load_embedder, load_main_functions,
                            pkg, read_jsonl, sha256)

load_dotenv()

EMBED_CONFIG = "D_e5_bengali"
COLLECTION_SUFFIX = "_ocr"          # the OCR-repaired index built in Experiment C
TOP_N = 5                           # main.py: rerank(..., top_n=5)
CONTEXT_CHARS = 600                 # main.py: CONTEXT_CHARS
TYPES = ["Direct / Factual", "Semantic / Paraphrased", "Conceptual",
         "Cross-lingual", "Multi-hop / Comparative", "Challenging / Indirect"]

# Copied verbatim from main.py's search_stream (non-overview branch). The
# script verifies that these strings are still present in main.py, so the
# evaluation cannot silently drift from the deployed prompt.
PROMPT_BN = ("You are a precise document assistant. "
             "Answer ONLY from the provided context. "
             "Respond in Bengali (Bangla). "
             "If the answer is not in the context, say so in Bengali. "
             "Use **bold** for key terms, bullet lists for multiple points. "
             "Do NOT mention filenames or page numbers.")
PROMPT_EN = ("You are a precise document assistant. "
             "Answer ONLY from the provided context. "
             "Respond in English. "
             "If the answer is not in the context, say so clearly. "
             "Use **bold** for key terms, bullet lists for multiple points, "
             "code blocks for code. "
             "Do NOT mention filenames or page numbers. Be direct and concise.")


def pick_sample(questions):
    by_doc = defaultdict(list)
    for q in questions:
        by_doc[q["doc_code"]].append(q)
    sample = []
    for doc in DOCS:
        qs = sorted(by_doc[doc], key=lambda q: int(q["question_id"].split("Q")[1]))
        for t in TYPES:
            first = next((q for q in qs if q["question_type"] == t), None)
            if first:
                sample.append(first)
    return sample


def retrieve(client, collection, model, question):
    flt = Filter(must=[FieldCondition(key="chat_id", match=MatchValue(value=EVAL_CID)),
                       FieldCondition(key="doc_id", match=MatchAny(any=DOCS))])
    qv = model.encode(question).tolist()
    pool = client.query_points(collection, query=qv, limit=200, query_filter=flt).points
    return [h for h in pool if h.score > 0.02][:TOP_N]


def generate_answer(groq_client, model_name, system_prompt, context, question):
    resp = groq_client.chat.completions.create(
        model=model_name, temperature=0.2, max_tokens=768,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}],
    )
    return resp.choices[0].message.content


def write_workbook(path, rows):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "How to grade"
    guide = [
        ("Docwise — answer quality grading", True),
        ("", False),
        ("Grade every row on the 'Grading' sheet. Fill ONLY the yellow columns (J, K, L, M).", False),
        ("", False),
        ("J  answer_correct — does the answer actually answer the question?", True),
        ("     1    = correct and complete", False),
        ("     0.5  = partly correct, or correct but missing an important part", False),
        ("     0    = wrong, or says the information is not in the documents when it is", False),
        ("     NA   = the question is ambiguous; exclude the row", False),
        ("", False),
        ("K  grounded — is every claim supported by the retrieved context (column I)?", True),
        ("     yes     = everything comes from the context", False),
        ("     partial = mostly, but something was added that is not in the context", False),
        ("     no      = invented content (hallucination)", False),
        ("", False),
        ("L  language_ok — is the answer in the same language as the question? (yes / no)", True),
        ("", False),
        ("M  notes — anything worth recording (optional)", True),
        ("", False),
        ("Method", True),
        ("Read the question (F), then the answer (H), then check it against the context (I).", False),
        ("An answer can be grounded but wrong (it used the wrong passage), or correct but not", False),
        ("grounded (the model knew it from elsewhere). Those two columns are judged separately.", False),
        ("Do not adjust the answers; only grade them.", False),
    ]
    for i, (text, bold) in enumerate(guide, 1):
        c = ws.cell(i, 1, text)
        c.font = Font(name="Arial", bold=bold, size=12 if i == 1 else 10)
    ws.column_dimensions["A"].width = 110

    gs = wb.create_sheet("Grading")
    head = ["sample_id", "question_id", "doc", "doc_type", "question_type", "question",
            "ground_truth_sections", "answer", "retrieved_context", "answer_correct",
            "grounded", "language_ok", "notes", "query_language", "retrieved_pages"]
    HF = PatternFill("solid", fgColor="305496")
    YF = PatternFill("solid", fgColor="FFF2CC")
    thin = Side(style="thin", color="BFBFBF")
    BD = Border(top=thin, bottom=thin, left=thin, right=thin)
    wrap = Alignment(wrap_text=True, vertical="top")
    for j, h in enumerate(head, 1):
        c = gs.cell(1, j, h)
        c.font = Font(name="Arial", bold=True, color="FFFFFF")
        c.fill = HF
        c.alignment = wrap
        c.border = BD
    for i, r in enumerate(rows, 2):
        vals = [r["sample_id"], r["question_id"], r["doc_code"], r["doc_type"], r["question_type"],
                r["question"], "; ".join(r["gt_sections"]), r["answer"], r["context"],
                "", "", "", "", r["query_language"], ", ".join(str(p) for p in r["pages"])]
        for j, v in enumerate(vals, 1):
            c = gs.cell(i, j, v)
            c.alignment = wrap
            c.border = BD
            c.font = Font(name="Nirmala UI", size=10) if j in (6, 7, 8, 9, 13) else Font(name="Arial", size=10)
            if head[j - 1] in ("answer_correct", "grounded", "language_ok", "notes"):
                c.fill = YF
    dv1 = DataValidation(type="list", formula1='"1,0.5,0,NA"', allow_blank=True)
    dv2 = DataValidation(type="list", formula1='"yes,partial,no"', allow_blank=True)
    dv3 = DataValidation(type="list", formula1='"yes,no"', allow_blank=True)
    gs.add_data_validation(dv1); gs.add_data_validation(dv2); gs.add_data_validation(dv3)
    n = len(rows) + 1
    dv1.add(f"J2:J{n}"); dv2.add(f"K2:K{n}"); dv3.add(f"L2:L{n}")
    for col, w in zip("ABCDEFGHIJKLMNO", [10, 11, 6, 15, 18, 45, 28, 70, 70, 13, 11, 12, 26, 8, 14]):
        gs.column_dimensions[col].width = w
    gs.freeze_panes = "B2"
    gs.auto_filter.ref = f"A1:O{n}"
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="main.py")
    ap.add_argument("--questions", default="evaluation/dataset/questions.jsonl")
    ap.add_argument("--chunks", default="evaluation/frozen_ocr/chunks.jsonl")
    ap.add_argument("--qdrant-path", default="evaluation/qdrant_eval")
    ap.add_argument("--out", default="evaluation/answers")
    args = ap.parse_args()

    # main.py writes these prompts as several adjacent string literals, so check
    # for distinctive fragments rather than the joined text.
    src = open(args.main, encoding="utf-8").read()
    for frag in ('"You are a precise document assistant. "',
                 '"Answer ONLY from the provided context. "',
                 '"If the answer is not in the context, say so in Bengali. "',
                 '"Do NOT mention filenames or page numbers. Be direct and concise."'):
        if frag not in src:
            sys.exit(f"[A] ABORT: main.py no longer contains the prompt fragment {frag} — "
                     f"update PROMPT_BN/PROMPT_EN in this script to match the app.")
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("[A] ABORT: GROQ_API_KEY not found (.env).")
    from groq import Groq
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    llm = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    questions = read_jsonl(args.questions)
    chunks = {c["chunk_id"]: c for c in read_jsonl(args.chunks)}
    sample = pick_sample(questions)
    main_fns = load_main_functions(args.main)
    detect = main_fns.get("_detect_response_lang")

    cfg = CONFIGS[EMBED_CONFIG]
    collection = cfg["collection"] + COLLECTION_SUFFIX
    model, _ = load_embedder(cfg)
    client = QdrantClient(path=args.qdrant_path)
    if collection not in [c.name for c in client.get_collections().collections]:
        sys.exit(f"[A] ABORT: collection {collection} not found — run Experiment C first.")

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    print(f"[A] generating answers for {len(sample)} questions with {llm} ...")
    for i, q in enumerate(sample, 1):
        top = retrieve(client, collection, model, q["question"])
        context = "".join(f"[Page {h.payload['page']}]\n{h.payload['text'][:CONTEXT_CHARS]}\n\n"
                          for h in top)
        lang = detect(context, q["question"]) if detect else ("bangla" if q["query_language"] == "bn" else "english")
        prompt = PROMPT_BN if lang == "bangla" else PROMPT_EN
        t0 = time.perf_counter()
        try:
            answer = generate_answer(groq_client, llm, prompt, context, q["question"])
        except Exception as e:
            answer = f"[ERROR] {e}"
        rows.append({
            "sample_id": f"S{i:02d}", "question_id": q["question_id"], "doc_code": q["doc_code"],
            "doc_type": q["doc_type"], "question_type": q["question_type"],
            "query_language": q["query_language"], "question": q["question"],
            "gt_sections": q["gt_section_list"], "chunk_ids": [h.payload["chunk_id"] for h in top],
            "pages": [h.payload["page"] for h in top],
            "retrieved_docs": sorted({h.payload["doc_id"] for h in top}),
            "context": context, "answer": answer, "response_language": lang,
            "seconds": round(time.perf_counter() - t0, 2),
        })
        if i % 10 == 0:
            print(f"[A]   {i}/{len(sample)}")
    client.close()

    with open(os.path.join(out_dir, "answers.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    try:
        write_workbook(os.path.join(out_dir, "grading.xlsx"), rows)
    except ImportError:
        print("[A] openpyxl is not installed — run: pip install openpyxl")

    manifest = {
        "run_id": run_id, "evaluation": "answer quality (manual grading)",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "pipeline": "deployed: multilingual-e5-small, dense-only, no reranker, OCR-repaired chunks",
        "llm": llm, "temperature": 0.2, "max_tokens": 768, "top_n": TOP_N,
        "context_chars_per_chunk": CONTEXT_CHARS, "sample_size": len(rows),
        "sampling": "first question of each of the 6 types in each of the 8 documents",
        "inputs_sha256": {"main.py": sha256(args.main), "chunks": sha256(args.chunks),
                          "questions": sha256(args.questions)},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        **{p: pkg(p) for p in ["sentence-transformers", "qdrant-client", "groq"]}},
        "mean_seconds_per_answer": round(sum(r["seconds"] for r in rows) / len(rows), 2),
        "answers_with_error": [r["sample_id"] for r in rows if r["answer"].startswith("[ERROR]")],
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    own = sum(q["doc_code"] in r["retrieved_docs"] for q, r in zip(sample, rows))
    print(f"\n[A] {len(rows)} answers written; mean {manifest['mean_seconds_per_answer']}s each")
    print(f"[A] the question's own document appeared in the retrieved passages for {own}/{len(rows)}")
    print(f"[A] grade this file by hand: {os.path.abspath(os.path.join(out_dir, 'grading.xlsx'))}")


if __name__ == "__main__":
    main()
