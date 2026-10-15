"""
Docwise — Experiment E: which retrieval pipeline should Docwise actually use?

Experiment D showed that with a Bengali-capable embedding model the dense
ranking alone is very strong (Recall@10 = 0.865), while the full pipeline
delivers only 0.28 to the LLM. This experiment holds the embedding model fixed
(D: intfloat/multilingual-e5-small, the same frozen chunks and the same Qdrant
collection) and varies ONLY what happens after the dense search.

Variants
--------
  V1 dense_only                 dense ranking, no BM25, no reranker
  V2 hybrid_rrf                 dense + BM25 fused with RRF            (current)
  V3 hybrid_weighted            RRF with dense weighted 0.8 / BM25 0.2
  V4 dense_msmarco              dense top-40, re-ordered by the English cross-encoder
  V5 hybrid_rrf_msmarco         the CURRENT PRODUCTION PIPELINE (reference)
  V6 hybrid_weighted_msmarco    V3 + the English cross-encoder
  V7 dense_bge                  dense top-40 + multilingual reranker   (optional)
  V8 hybrid_weighted_bge        V3 + multilingual reranker             (optional)

V7/V8 need --with-multilingual-reranker: BAAI/bge-reranker-v2-m3 is ~25x larger
than the current reranker and slow on CPU, so those two variants also use a
smaller candidate list (--bge-candidates, default 20).

The 0.8/0.2 weights and the candidate sizes are fixed here, before any result is
seen. Reranker scores are cached per (question, chunk) pair, so variants that
share candidates do not pay for the same pair twice.

Everything else is unchanged: frozen chunks, 200 questions, reviewed qrels,
search over all 8 documents, top_k=40, RRF k=60, dense floor 0.02.

Usage (from the backend folder, after Experiment D has been run)
----------------------------------------------------------------
    python run_pipeline_experiment.py
    python run_pipeline_experiment.py --with-multilingual-reranker
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
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
from rank_bm25 import BM25Okapi

import run_experiment as rx
from run_experiment import (CONFIGS, DOCS, EVAL_CID, RERANK_MODEL, TOP_K, aggregate,
                            load_embedder, load_main_functions, paired_stats,
                            per_question_metrics, pkg, read_jsonl, sha256)

# Each variant produces ONE ranking, not the five pipeline stages of the A/B/D
# harness; aggregate() groups and averages exactly as before over that one stage.
rx.STAGES = ["final"]

EMBED_CONFIG = "D_e5_bengali"          # fixed for this experiment
BGE_MODEL = "BAAI/bge-reranker-v2-m3"
DENSE_WEIGHT, BM25_WEIGHT = 0.8, 0.2
REFERENCE = "V5_hybrid_rrf_msmarco"    # the current production pipeline
PRODUCTION_TOP_N = 5
MSMARCO_FLOOR = -6.0                   # the floor main.py applies (ms-marco only)

VARIANTS = {
    "V1_dense_only":            {"retrieval": "dense",           "rerank": None},
    "V2_hybrid_rrf":            {"retrieval": "hybrid_rrf",      "rerank": None},
    "V3_hybrid_weighted":       {"retrieval": "hybrid_weighted", "rerank": None},
    "V4_dense_msmarco":         {"retrieval": "dense",           "rerank": "msmarco"},
    "V5_hybrid_rrf_msmarco":    {"retrieval": "hybrid_rrf",      "rerank": "msmarco"},
    "V6_hybrid_weighted_msmarco": {"retrieval": "hybrid_weighted", "rerank": "msmarco"},
    "V7_dense_bge":             {"retrieval": "dense",           "rerank": "bge"},
    "V8_hybrid_weighted_bge":   {"retrieval": "hybrid_weighted", "rerank": "bge"},
}


# ──────────────────────────────────────────────────────────────────────────────
def retrieve_all(client, collection, model, tokenize, questions, n_chunks):
    """One pass over the questions: dense, BM25 and both fusions per question."""
    flt = Filter(must=[FieldCondition(key="chat_id", match=MatchValue(value=EVAL_CID)),
                       FieldCondition(key="doc_id", match=MatchAny(any=DOCS))])
    out, texts = {}, {}
    t0 = time.perf_counter()
    for q in questions:
        query = q["question"]
        qv = model.encode(query).tolist()
        pool = client.query_points(collection, query=qv, limit=max(TOP_K, 200),
                                   query_filter=flt).points
        if len(pool) != n_chunks:
            sys.exit(f"[E] ABORT: pool has {len(pool)} chunks, expected {n_chunks}")
        for h in pool:
            texts[h.payload["chunk_id"]] = h.payload["text"]
        dense = [h for h in pool if h.score > 0.02]
        bm25 = BM25Okapi([tokenize(h.payload["text"]) for h in pool])
        sc = bm25.get_scores(tokenize(query))
        RRF_K = 60
        d_rank = {h.id: r + 1 for r, h in enumerate(dense)}
        b_order = sorted(range(len(pool)), key=lambda i: sc[i], reverse=True)
        b_rank = {pool[i].id: r + 1 for r, i in enumerate(b_order) if sc[i] > 0}

        def fuse(wd, wb):
            def s(h):
                v = 0.0
                if h.id in d_rank: v += wd / (RRF_K + d_rank[h.id])
                if h.id in b_rank: v += wb / (RRF_K + b_rank[h.id])
                return v
            keep = [h for h in pool if h.id in d_rank or h.id in b_rank]
            return [(h.payload["chunk_id"], s(h))
                    for h in sorted(keep, key=s, reverse=True)[:TOP_K]]

        out[q["question_id"]] = {
            "dense": [(h.payload["chunk_id"], float(h.score)) for h in dense],
            "hybrid_rrf": fuse(1.0, 1.0),
            "hybrid_weighted": fuse(DENSE_WEIGHT, BM25_WEIGHT),
        }
    print(f"[E] retrieval for {len(questions)} questions in {time.perf_counter() - t0:.1f}s")
    return out, texts


class RerankCache:
    """Scores (question, chunk) pairs once and reuses them across variants."""

    def __init__(self, model_name, max_length=512):
        from sentence_transformers import CrossEncoder
        print(f"[E] loading reranker {model_name} ...")
        self.ce = CrossEncoder(model_name, max_length=max_length)
        self.cache = {}
        self.calls = 0

    def score(self, qid, query, chunk_ids, texts):
        todo = [c for c in chunk_ids if (qid, c) not in self.cache]
        if todo:
            pairs = [(query, texts[c][:512]) for c in todo]
            for c, s in zip(todo, self.ce.predict(pairs)):
                self.cache[(qid, c)] = float(s)
            self.calls += len(todo)
        return [(c, self.cache[(qid, c)]) for c in chunk_ids]


def rank_for_variant(vname, v, qid, query, stages, texts, rerankers, bge_candidates):
    base = stages[v["retrieval"]]
    if v["rerank"] is None:
        return base
    cand = [c for c, _ in base][:TOP_K if v["rerank"] == "msmarco" else bge_candidates]
    scored = rerankers[v["rerank"]].score(qid, query, cand, texts)
    return sorted(scored, key=lambda x: x[1], reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="main.py")
    ap.add_argument("--chunks", default="evaluation/frozen/chunks.jsonl")
    ap.add_argument("--questions", default="evaluation/dataset/questions.jsonl")
    ap.add_argument("--qrels", default="evaluation/labels/qrels.jsonl")
    ap.add_argument("--qdrant-path", default="evaluation/qdrant_eval")
    ap.add_argument("--results", default="evaluation/results_pipeline")
    ap.add_argument("--with-multilingual-reranker", action="store_true",
                    help="also run V7/V8 with BAAI/bge-reranker-v2-m3 (slow on CPU)")
    ap.add_argument("--bge-candidates", type=int, default=20)
    args = ap.parse_args()

    chunks = read_jsonl(args.chunks)
    questions = read_jsonl(args.questions)
    rel = defaultdict(dict)
    for r in read_jsonl(args.qrels):
        rel[r["question_id"]][r["chunk_id"]] = r["matched_sections"]

    variants = {k: v for k, v in VARIANTS.items()
                if v["rerank"] != "bge" or args.with_multilingual_reranker}
    main_fns = load_main_functions(args.main)
    cfg = CONFIGS[EMBED_CONFIG]
    model, _ = load_embedder(cfg)
    client = QdrantClient(path=args.qdrant_path)
    if cfg["collection"] not in [c.name for c in client.get_collections().collections]:
        sys.exit(f"[E] ABORT: collection {cfg['collection']} not found — run Experiment D first.")

    stages, texts = retrieve_all(client, cfg["collection"], model, main_fns["tokenize"],
                                 questions, len(chunks))
    client.close()

    rerankers = {"msmarco": RerankCache(RERANK_MODEL)}
    if args.with_multilingual_reranker:
        print(f"[E] NOTE: {BGE_MODEL} is large; V7/V8 on CPU can take an hour or more.")
        rerankers["bge"] = RerankCache(BGE_MODEL)

    rows_by_variant, rankings, timing, floor_empty = {}, {}, {}, {}
    for vname, v in variants.items():
        print(f"[E] === {vname} ===")
        t0 = time.perf_counter()
        rows, rank_dump, empty = [], [], 0
        for q in questions:
            qid = q["question_id"]
            ranked = rank_for_variant(vname, v, qid, q["question"], stages[qid], texts,
                                      rerankers, args.bge_candidates)
            rows.append({"question_id": qid,
                         "final": per_question_metrics(ranked, rel[qid], q["gt_section_list"], "final")})
            rank_dump.append({"question_id": qid, "top": [
                {"rank": i + 1, "chunk_id": c, "score": round(float(s), 6),
                 "relevant": int(c in rel[qid])} for i, (c, s) in enumerate(ranked[:TOP_K])]})
            if v["rerank"] == "msmarco" and not [c for c, s in ranked if s > MSMARCO_FLOOR][:PRODUCTION_TOP_N]:
                empty += 1
        rows_by_variant[vname] = rows
        rankings[vname] = rank_dump
        timing[vname] = round(time.perf_counter() - t0, 1)
        floor_empty[vname] = empty if v["rerank"] == "msmarco" else None

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.results, run_id)
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for vname, rows in rows_by_variant.items():
        vdir = os.path.join(out_dir, vname)
        os.makedirs(vdir, exist_ok=True)
        with open(os.path.join(vdir, "per_question.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(vdir, "rankings.jsonl"), "w", encoding="utf-8") as f:
            for r in rankings[vname]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        metrics = aggregate(rows, questions)          # same grouping as Experiment A/B/D
        with open(os.path.join(vdir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        summary[vname] = metrics["overall"]["all"]["final"]

    # paired comparison of every variant against the current production pipeline
    import csv
    rng = np.random.default_rng(12345)
    qmap = {q["question_id"]: q for q in questions}
    ref_rows = {r["question_id"]: r for r in rows_by_variant[REFERENCE]}
    comp = []
    for vname, rows in rows_by_variant.items():
        if vname == REFERENCE:
            continue
        cur = {r["question_id"]: r for r in rows}
        groups = [("overall", "all", list(cur))]
        for g in ["doc_type", "query_language", "cross_lingual_direction", "question_type"]:
            b = defaultdict(list)
            for qid in cur:
                b[qmap[qid][g]].append(qid)
            groups += [(g, val, ids) for val, ids in sorted(b.items())]
        for met in ["R@1", "R@5", "R@10", "MRR@10"]:
            for g, val, ids in groups:
                xs = [ref_rows[i]["final"][met] for i in ids]
                ys = [cur[i]["final"][met] for i in ids]
                diff, lo, hi, p, test = paired_stats(xs, ys, met != "MRR@10", rng)
                comp.append({"reference": REFERENCE, "variant": vname, "metric": met,
                             "group": g, "value": val, "n": len(ids),
                             "mean_reference": round(float(np.mean(xs)), 4),
                             "mean_variant": round(float(np.mean(ys)), 4),
                             "difference": round(diff, 4), "ci95_low": round(lo, 4),
                             "ci95_high": round(hi, 4), "p_value": round(p, 5), "test": test})
    with open(os.path.join(out_dir, "comparison.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(comp[0].keys()))
        w.writeheader()
        w.writerows(comp)

    manifest = {
        "run_id": run_id, "experiment": "E — retrieval pipeline variants",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "embedding_config": {**cfg, "fixed_for_this_experiment": True},
        "settings": {"dense_weight": DENSE_WEIGHT, "bm25_weight": BM25_WEIGHT, "rrf_k": 60,
                     "top_k": TOP_K, "bge_candidates": args.bge_candidates,
                     "msmarco_floor_for_production": MSMARCO_FLOOR,
                     "production_top_n": PRODUCTION_TOP_N, "reference_variant": REFERENCE},
        "inputs_sha256": {"main.py": sha256(args.main), "chunks": sha256(args.chunks),
                          "questions": sha256(args.questions), "qrels": sha256(args.qrels)},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        **{p: pkg(p) for p in ["sentence-transformers", "transformers", "torch",
                                                "qdrant-client", "rank-bm25", "numpy", "scipy"]}},
        "variants": {k: {**v, "seconds": timing[k],
                         "questions_emptied_by_floor": floor_empty[k]} for k, v in variants.items()},
        "reranker_pairs_scored": {k: r.calls for k, r in rerankers.items()},
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n[E] OVERALL (200 questions, embedding model fixed = multilingual-e5-small)")
    print(f" {'variant':<30}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR@10':>8}{'sec':>7}")
    for vname, m in summary.items():
        g = lambda k: f"{m[k]['mean']:.3f}"
        star = "  <- current" if vname == REFERENCE else ""
        print(f" {vname:<30}{g('R@1'):>7}{g('R@5'):>7}{g('R@10'):>7}{g('MRR@10'):>8}"
              f"{timing[vname]:>7}{star}")
    print(f"\n[E] results: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
