"""
Docwise — Experiment C: does re-reading broken Bengali text layers with OCR help?

Background
----------
U1 and U2 are Unicode Bengali PDFs exported from Word. Their text layer is
scrambled (agreement with the source text 0.62-0.72, versus 0.91-0.96 for OCR of
the scanned documents), and retrieval on them is far worse than on every other
document type. main.py now detects such pages with a Bengali orthography check
and re-reads them with OCR.

This experiment compares the two chunk sets under the DEPLOYED configuration
(embedding model multilingual-e5-small, dense-only retrieval, the settings
Experiment E selected):

    before  evaluation/frozen/chunks.jsonl       broken text layer kept
    after   evaluation/frozen_ocr/chunks.jsonl   broken pages re-read with OCR

The 200 questions, the section-level ground truth and the retrieval code are
unchanged. Only U1/U2's text differs, so the other 150 questions act as a
control: their numbers should stay the same.

Preparation
-----------
    python freeze_chunks.py --pdf-dir evaluation/pdfs --out evaluation/frozen_ocr --check-determinism
    python label_sections.py --chunks evaluation/frozen_ocr/chunks.jsonl --out evaluation/labels_ocr

Then
----
    python run_experiment_c.py
"""

import argparse
import csv
import datetime
import json
import os
import platform
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

import run_experiment as rx
from run_experiment import (CONFIGS, DOCS, EVAL_CID, TOP_K, aggregate,
                            ensure_index, load_embedder, paired_stats, per_question_metrics,
                            pkg, read_jsonl, sha256)

rx.STAGES = ["dense"]          # the deployed pipeline is dense-only
EMBED_CONFIG = "D_e5_bengali"
AFFECTED_DOCS = ["U1", "U2"]   # the only documents whose text changes


def qrels_from_labels(labels_path, questions, out_path):
    """Relevant chunks = chunks labelled with any of a question's ground-truth sections."""
    by_doc_section = defaultdict(set)
    for r in read_jsonl(labels_path):
        for s in r.get("final_labels", r["auto_labels"]):
            by_doc_section[(r["doc_code"], s)].add(r["chunk_id"])
    rows = []
    for q in questions:
        for s in q["gt_section_list"]:
            for c in sorted(by_doc_section[(q["doc_code"], s)]):
                rows.append((q["question_id"], c, s))
    merged = defaultdict(list)
    for qid, c, s in rows:
        merged[(qid, c)].append(s)
    with open(out_path, "w", encoding="utf-8") as f:
        for (qid, c), secs in merged.items():
            f.write(json.dumps({"question_id": qid, "chunk_id": c, "relevant": 1,
                                "matched_sections": sorted(set(secs))}, ensure_ascii=False) + "\n")
    empty = [q["question_id"] for q in questions
             if not any(k[0] == q["question_id"] for k in merged)]
    if empty:
        sys.exit(f"[C] ABORT: no relevant chunk for {empty[:5]} ... — check the labels")
    return out_path


def evaluate(client, cfg, model, chunks, chunks_sha, questions, qrels_path, reindex):
    rel = defaultdict(dict)
    for r in read_jsonl(qrels_path):
        rel[r["question_id"]][r["chunk_id"]] = r["matched_sections"]
    ensure_index(client, cfg, model, chunks, chunks_sha, reindex)
    flt = Filter(must=[FieldCondition(key="chat_id", match=MatchValue(value=EVAL_CID)),
                       FieldCondition(key="doc_id", match=MatchAny(any=DOCS))])
    rows, dump = [], []
    for q in questions:
        qv = model.encode(q["question"]).tolist()
        pool = client.query_points(cfg["collection"], query=qv, limit=max(TOP_K, 200),
                                   query_filter=flt).points
        ranked = [(h.payload["chunk_id"], float(h.score)) for h in pool if h.score > 0.02]
        qrel = rel[q["question_id"]]
        rows.append({"question_id": q["question_id"],
                     "dense": per_question_metrics(ranked, qrel, q["gt_section_list"], "dense")})
        dump.append({"question_id": q["question_id"], "top": [
            {"rank": i + 1, "chunk_id": c, "score": round(s, 6), "relevant": int(c in qrel)}
            for i, (c, s) in enumerate(ranked[:TOP_K])]})
    return rows, dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="evaluation/dataset/questions.jsonl")
    ap.add_argument("--chunks-before", default="evaluation/frozen/chunks.jsonl")
    ap.add_argument("--qrels-before", default="evaluation/labels/qrels.jsonl")
    ap.add_argument("--chunks-after", default="evaluation/frozen_ocr/chunks.jsonl")
    ap.add_argument("--labels-after", default="evaluation/labels_ocr/chunk_labels_auto.jsonl")
    ap.add_argument("--qdrant-path", default="evaluation/qdrant_eval")
    ap.add_argument("--results", default="evaluation/results_ocr")
    ap.add_argument("--reindex", action="store_true")
    args = ap.parse_args()

    questions = read_jsonl(args.questions)
    before = read_jsonl(args.chunks_before)
    after = read_jsonl(args.chunks_after)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.results, run_id)
    os.makedirs(out_dir, exist_ok=True)

    unchanged = ({c["chunk_id"]: c["text_sha1"] for c in before if c["doc_code"] not in AFFECTED_DOCS}
                 == {c["chunk_id"]: c["text_sha1"] for c in after if c["doc_code"] not in AFFECTED_DOCS})
    print(f"[C] chunks: before={len(before)} after={len(after)}; "
          f"documents other than {AFFECTED_DOCS} unchanged: {unchanged}")
    if not unchanged:
        print("[C] WARNING: text changed outside U1/U2 — the control group is no longer clean.")

    qrels_after = qrels_from_labels(args.labels_after, questions,
                                    os.path.join(out_dir, "qrels_after.jsonl"))
    cfg = CONFIGS[EMBED_CONFIG]
    model, _ = load_embedder(cfg)
    client = QdrantClient(path=args.qdrant_path)

    sets = {
        "before_text_layer": (before, args.chunks_before, args.qrels_before, cfg["collection"]),
        "after_ocr_fallback": (after, args.chunks_after, qrels_after, cfg["collection"] + "_ocr"),
    }
    per_set, summary = {}, {}
    for name, (chunks, path, qrels, coll) in sets.items():
        print(f"[C] === {name} ===")
        rows, dump = evaluate(client, {**cfg, "collection": coll}, model, chunks,
                              sha256(path), questions, qrels, args.reindex)
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "per_question.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(d, "rankings.jsonl"), "w", encoding="utf-8") as f:
            for r in dump:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        metrics = aggregate(rows, questions)
        with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        per_set[name] = {r["question_id"]: r for r in rows}
        summary[name] = metrics
    client.close()

    # paired comparison: the 50 affected questions, and the 150 controls
    rng = np.random.default_rng(12345)
    qmap = {q["question_id"]: q for q in questions}
    groups = [("affected (U1/U2)", [q["question_id"] for q in questions if q["doc_code"] in AFFECTED_DOCS]),
              ("control (other 6 docs)", [q["question_id"] for q in questions if q["doc_code"] not in AFFECTED_DOCS]),
              ("overall", [q["question_id"] for q in questions])]
    for g in ["query_language", "question_type"]:
        b = defaultdict(list)
        for q in questions:
            if q["doc_code"] in AFFECTED_DOCS:
                b[q[g]].append(q["question_id"])
        groups += [(f"affected / {g}={v}", ids) for v, ids in sorted(b.items())]
    rows = []
    A, B = per_set["before_text_layer"], per_set["after_ocr_fallback"]
    for met in ["R@1", "R@5", "R@10", "MRR@10"]:
        for gname, ids in groups:
            xs = [A[i]["dense"][met] for i in ids]
            ys = [B[i]["dense"][met] for i in ids]
            diff, lo, hi, p, test = paired_stats(xs, ys, met != "MRR@10", rng)
            rows.append({"group": gname, "n": len(ids), "metric": met,
                         "mean_before": round(float(np.mean(xs)), 4),
                         "mean_after": round(float(np.mean(ys)), 4),
                         "difference": round(diff, 4), "ci95_low": round(lo, 4),
                         "ci95_high": round(hi, 4), "p_value": round(p, 5), "test": test})
    with open(os.path.join(out_dir, "comparison.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    manifest = {
        "run_id": run_id, "experiment": "C — broken Bengali text layer vs OCR fallback",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "pipeline": "deployed configuration: dense-only retrieval, no BM25, no reranker",
        "embedding_config": cfg, "affected_documents": AFFECTED_DOCS,
        "control_chunks_identical": unchanged,
        "inputs_sha256": {"chunks_before": sha256(args.chunks_before),
                          "chunks_after": sha256(args.chunks_after),
                          "qrels_before": sha256(args.qrels_before),
                          "labels_after": sha256(args.labels_after),
                          "questions": sha256(args.questions)},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        **{p: pkg(p) for p in ["sentence-transformers", "transformers", "torch",
                                                "qdrant-client", "numpy", "scipy"]}},
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n[C] U1/U2 only (50 questions, dense retrieval)")
    print(f" {'set':<22}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR@10':>8}")
    for name, m in summary.items():
        vals = {}
        for met in ["R@1", "R@5", "R@10", "MRR@10"]:
            ids = [q["question_id"] for q in questions if q["doc_code"] in AFFECTED_DOCS]
            vals[met] = np.mean([per_set[name][i]["dense"][met] for i in ids])
        print(f" {name:<22}" + "".join(f"{vals[m]:>7.3f}" if m != "MRR@10" else f"{vals[m]:>8.3f}"
                                       for m in ["R@1", "R@5", "R@10", "MRR@10"]))
    print("\n[C] control documents (150 questions — should not change)")
    for name in summary:
        ids = [q["question_id"] for q in questions if q["doc_code"] not in AFFECTED_DOCS]
        v = np.mean([per_set[name][i]["dense"]["R@5"] for i in ids])
        print(f" {name:<22}R@5 {v:.3f}")
    print(f"\n[C] results: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
