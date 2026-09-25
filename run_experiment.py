"""
Docwise — Step 3: run the embedding-model comparison (retrieval only, no LLM).

Configurations
--------------
  A_baseline         all-MiniLM-L6-v2                         (default max_seq_length 256)  primary
  B_multilingual     paraphrase-multilingual-MiniLM-L12-v2    (default max_seq_length 128)  primary
  B256_multilingual  paraphrase-multilingual-MiniLM-L12-v2    (max_seq_length raised to 256) supplementary

Everything else is identical and taken from main.py itself:
  * tokenize(), hybrid_search() and rerank() are loaded from main.py's source
    (not re-typed), so the experiment runs the real Docwise retrieval code;
  * the frozen 122 chunks, 200 questions and the reviewed relevance file (qrels);
  * the reranker cross-encoder/ms-marco-MiniLM-L-6-v2, top_k=40, RRF k=60, floors.

Each configuration gets its own Qdrant collection inside evaluation/qdrant_eval/,
a separate local store, so the running app and ./qdrant_data are never touched.

Stages measured for every question (search scope: all 8 documents)
-------------------------------------------------------------------
  bm25        lexical ranking alone (diagnostic; identical for every configuration)
  dense       embedding ranking alone         <- the purest measure of the model
  hybrid      RRF fusion of dense + BM25, top 40
  rerank      the 40 hybrid candidates re-ordered by the cross-encoder
  production  what Docwise actually sends to the LLM (score > -6, top 5)

Metrics: Recall@1/5/10 (at least one relevant chunk in the top k), MRR@10,
full-list MRR, and for questions with two ground-truth sections AllSections@k
(every ground-truth section represented in the top k).

Usage (from the backend folder; the Docwise server may keep running)
--------------------------------------------------------------------
    python run_experiment.py
    python run_experiment.py --configs A_baseline B_multilingual
    python run_experiment.py --reindex        # rebuild the eval collections
"""

import argparse
import ast
import csv
import datetime
import hashlib
import json
import os
import platform
import re
import sys
import time
import uuid
from collections import defaultdict
from importlib import metadata

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import regex
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchAny, MatchValue,
                                  PointStruct, VectorParams)
from rank_bm25 import BM25Okapi

CONFIGS = {
    "A_baseline": {
        "model": "all-MiniLM-L6-v2", "max_seq_length": None,
        "collection": "docwise_eval_A_baseline_minilm", "role": "primary"},
    "B_multilingual": {
        "model": "paraphrase-multilingual-MiniLM-L12-v2", "max_seq_length": None,
        "collection": "docwise_eval_B_multilingual_minilm", "role": "primary"},
    "B256_multilingual": {
        "model": "paraphrase-multilingual-MiniLM-L12-v2", "max_seq_length": 256,
        "collection": "docwise_eval_B256_multilingual_minilm", "role": "supplementary"},
    # Experiment D: a multilingual model whose training data does include Bengali.
    # E5 models REQUIRE the "query: " / "passage: " prefixes — without them the
    # embeddings are out of distribution, so the prefixes are part of the model,
    # not a change to the pipeline.
    "D_e5_bengali": {
        "model": "intfloat/multilingual-e5-small", "max_seq_length": None,
        "collection": "docwise_eval_D_e5_small", "role": "supplementary",
        "query_prefix": "query: ", "passage_prefix": "passage: "},
}
COMPARISONS = [("A_baseline", "B_multilingual"), ("A_baseline", "B256_multilingual"),
               ("B_multilingual", "B256_multilingual"),
               ("A_baseline", "D_e5_bengali"), ("B_multilingual", "D_e5_bengali")]
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EVAL_CID = "docwise-eval"
DOCS = ["B1", "B2", "U1", "U2", "I1", "I2", "E1", "E2"]
TOP_K = 40                     # hybrid_search(top_k=40) as in search_stream
PRODUCTION_TOP_N = 5           # rerank(top_n=5) as in search_stream
KS = (1, 5, 10)
STAGES = ["bm25", "dense", "hybrid", "rerank", "production"]
GROUPS = ["doc_type", "doc_code", "query_language", "question_type", "cross_lingual_direction"]
BOOTSTRAP, SEED = 10000, 12345
FUNCS_FROM_MAIN = ["tokenize", "hybrid_search", "rerank"]


# ──────────────────────────────────────────────────────────────────────────────
def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def pkg(name):
    try:
        return metadata.version(name)
    except Exception:
        return None


def load_main_functions(main_path):
    src = open(main_path, encoding="utf-8").read()
    if RERANK_MODEL not in src:
        sys.exit(f"[exp] ABORT: main.py no longer uses {RERANK_MODEL}")
    tree = ast.parse(src)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS_FROM_MAIN]
    missing = set(FUNCS_FROM_MAIN) - {n.name for n in nodes}
    if missing:
        sys.exit(f"[exp] ABORT: not found in main.py: {sorted(missing)}")
    ns = {"__name__": "docwise_main_retrieval", "re": re, "_regex": regex,
          "BM25Okapi": BM25Okapi, "Filter": Filter, "FieldCondition": FieldCondition,
          "MatchValue": MatchValue, "MatchAny": MatchAny}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), main_path, "exec"), ns)
    if ns["tokenize"]("বাংলাদেশের প্রধান নদী") != ["বাংলাদেশের", "প্রধান", "নদী"]:
        sys.exit("[exp] ABORT: main.py tokenize() still splits Bengali words — apply the BM25 fix first.")
    return ns


class PrefixedEmbedder:
    """Adds the model's required instruction prefixes (E5 family).

    A list of texts is a passage batch (indexing); a single string is a query —
    exactly how main.py calls encode().
    """

    def __init__(self, model, query_prefix, passage_prefix):
        self.m, self.qp, self.pp = model, query_prefix, passage_prefix

    def encode(self, x, **kw):
        if isinstance(x, str):
            return self.m.encode(self.qp + x, **kw)
        return self.m.encode([self.pp + t for t in x], **kw)

    @property
    def max_seq_length(self):
        return self.m.max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, v):
        self.m.max_seq_length = v

    def __getattr__(self, name):
        return getattr(self.m, name)

    def __getitem__(self, i):
        return self.m[i]


def load_embedder(cfg):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(cfg["model"])
    default_len = int(m.max_seq_length)
    if cfg["max_seq_length"]:
        try:
            limit = int(m[0].auto_model.config.max_position_embeddings)
        except Exception:
            limit = None
        if limit is not None and cfg["max_seq_length"] > limit:
            sys.exit(f"[exp] ABORT: {cfg['model']} supports at most {limit} positions")
        m.max_seq_length = int(cfg["max_seq_length"])
    if cfg.get("query_prefix") or cfg.get("passage_prefix"):
        m = PrefixedEmbedder(m, cfg.get("query_prefix", ""), cfg.get("passage_prefix", ""))
    return m, default_len


def load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, max_length=512)


def model_info(m, default_len):
    info = {"max_seq_length_default": default_len, "max_seq_length_used": int(m.max_seq_length)}
    try:
        info["parameters"] = int(sum(p.numel() for p in m.parameters()))
    except Exception:
        info["parameters"] = None
    try:
        fn = getattr(m, "get_embedding_dimension", None) or m.get_sentence_embedding_dimension
        info["dimension"] = int(fn())
    except Exception:
        info["dimension"] = None
    info["device"] = str(getattr(m, "device", "unknown"))
    return info


# ──────────────────────────────────────────────────────────────────────────────
# Indexing — mirrors _process_pdf_background (batch_size=64, cosine, same payload)
# ──────────────────────────────────────────────────────────────────────────────
def point_id(chunk_id):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"docwise-eval/{chunk_id}"))


def ensure_index(client, cfg, model, chunks, chunks_sha, reindex):
    coll = cfg["collection"]
    marker = {"embed_model": cfg["model"], "embed_max_seq_length": int(model.max_seq_length),
              "chunks_sha256": chunks_sha, "passage_prefix": cfg.get("passage_prefix", "")}
    exists = coll in [c.name for c in client.get_collections().collections]
    if exists and not reindex:
        pts, _ = client.scroll(coll, limit=len(chunks) + 10, with_payload=True, with_vectors=False)
        ok = (len(pts) == len(chunks) and
              all(all(p.payload.get(k) == v for k, v in marker.items()) for p in pts))
        if ok:
            print(f"[exp]   reusing collection {coll} ({len(pts)} points)")
            return {"reused": True, "index_seconds": None}
        print(f"[exp]   collection {coll} does not match the frozen chunks/model — rebuilding")
    if exists:
        client.delete_collection(coll)

    texts = [c["text"] for c in chunks]
    t0 = time.perf_counter()
    vectors = model.encode(texts, batch_size=64, show_progress_bar=False)
    enc_s = time.perf_counter() - t0
    client.create_collection(coll, vectors_config=VectorParams(size=len(vectors[0]),
                                                               distance=Distance.COSINE))
    points = [PointStruct(id=point_id(c["chunk_id"]), vector=vectors[i].tolist(), payload={
        "chat_id": EVAL_CID, "doc_id": c["doc_code"], "text": c["text"], "page": c["page"],
        "filename": c["source_file"], "chunk_id": c["chunk_id"], **marker})
        for i, c in enumerate(chunks)]
    for i in range(0, len(points), 100):
        client.upsert(coll, points=points[i:i + 100])
    print(f"[exp]   indexed {len(points)} chunks into {coll} in {enc_s:.2f}s")
    return {"reused": False, "index_seconds": round(enc_s, 3)}


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval with every stage recorded, checked against main.py's own functions
# ──────────────────────────────────────────────────────────────────────────────
def traced_hybrid(ns, query, n_chunks):
    """Same computation as main.hybrid_search, but also returns the stages."""
    tokenize = ns["tokenize"]
    qv = ns["embed_model"].encode(query).tolist()
    flt = Filter(must=[FieldCondition(key="chat_id", match=MatchValue(value=EVAL_CID)),
                       FieldCondition(key="doc_id", match=MatchAny(any=DOCS))])
    pool = ns["qdrant"].query_points(ns["QDRANT_COLLECTION"], query=qv, limit=max(TOP_K, 200),
                                     query_filter=flt).points
    if len(pool) != n_chunks:
        sys.exit(f"[exp] ABORT: candidate pool has {len(pool)} chunks, expected {n_chunks}")
    dense = [h for h in pool if h.score > 0.02]
    bm25 = BM25Okapi([tokenize(h.payload["text"]) for h in pool])
    sc = bm25.get_scores(tokenize(query))
    RRF_K = 60
    d_rank = {h.id: r + 1 for r, h in enumerate(dense)}
    b_order = sorted(range(len(pool)), key=lambda i: sc[i], reverse=True)
    b_rank = {pool[i].id: r + 1 for r, i in enumerate(b_order) if sc[i] > 0}

    def rrf(h):
        s = 0.0
        if h.id in d_rank: s += 1.0 / (RRF_K + d_rank[h.id])
        if h.id in b_rank: s += 1.0 / (RRF_K + b_rank[h.id])
        return s

    scored = [h for h in pool if h.id in d_rank or h.id in b_rank]
    hybrid = sorted(scored, key=rrf, reverse=True)[:TOP_K]
    cid = lambda h: h.payload["chunk_id"]
    return {
        "dense": [(cid(h), float(h.score)) for h in dense],
        "bm25": [(cid(pool[i]), float(sc[i])) for i in b_order if sc[i] > 0],
        "hybrid": [(cid(h), rrf(h)) for h in hybrid],
    }


def run_config(name, cfg, main_fns, client, reranker, chunks, chunks_sha, questions, reindex):
    print(f"\n[exp] === {name}: {cfg['model']} (max_seq_length={cfg['max_seq_length'] or 'default'}) ===")
    t0 = time.perf_counter()
    model, default_len = load_embedder(cfg)
    load_s = time.perf_counter() - t0
    info = model_info(model, default_len)
    idx = ensure_index(client, cfg, model, chunks, chunks_sha, reindex)

    # the functions loaded from main.py look up their globals in main_fns itself,
    # so the per-configuration objects must be placed into that same dict
    ns = main_fns
    ns.update(qdrant=client, QDRANT_COLLECTION=cfg["collection"], embed_model=model, reranker=reranker)
    raw, lat_retrieval, lat_rerank = [], [], []
    for i, q in enumerate(questions, 1):
        query = q["question"]
        t1 = time.perf_counter()
        st = traced_hybrid(ns, query, len(chunks))
        lat_retrieval.append(time.perf_counter() - t1)

        # the real Docwise functions — must give exactly the same result
        candidates = ns["hybrid_search"](query, EVAL_CID, DOCS, top_k=TOP_K)
        if [c.payload["chunk_id"] for c in candidates] != [c for c, _ in st["hybrid"]]:
            sys.exit(f"[exp] ABORT: traced hybrid differs from main.hybrid_search for {q['question_id']}")
        t2 = time.perf_counter()
        scores = reranker.predict([(query, c.payload["text"][:512]) for c in candidates])
        lat_rerank.append(time.perf_counter() - t2)
        full = sorted(zip((float(s) for s in scores), (c.payload["chunk_id"] for c in candidates)),
                      key=lambda x: x[0], reverse=True)
        production = [c.payload["chunk_id"] for c in ns["rerank"](query, candidates, top_n=PRODUCTION_TOP_N)]
        expected = [c for s, c in full if s > -6][:PRODUCTION_TOP_N]
        if production != expected:
            sys.exit(f"[exp] ABORT: rerank stage mismatch for {q['question_id']}")
        st["rerank"] = [(c, s) for s, c in full]
        st["production"] = [(c, dict((cc, s) for s, cc in full)[c]) for c in production]
        raw.append({"question_id": q["question_id"], "stages": st})
        if i % 50 == 0:
            print(f"[exp]   {i}/{len(questions)} questions")

    run_info = {"config": name, **cfg, "model_info": info, "model_load_seconds": round(load_s, 2),
                **idx,
                "mean_retrieval_ms": round(1000 * float(np.mean(lat_retrieval)), 2),
                "mean_rerank_ms": round(1000 * float(np.mean(lat_rerank)), 2)}
    del model
    return raw, run_info


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def per_question_metrics(ranked, rel, gt_sections, stage):
    ids = [c for c, _ in ranked]
    ranks = [i + 1 for i, c in enumerate(ids) if c in rel]
    first = ranks[0] if ranks else None
    m = {}
    for k in KS:
        m[f"R@{k}"] = None if (stage == "production" and k > PRODUCTION_TOP_N) else int(bool(first and first <= k))
    m["MRR@10"] = (1.0 / first) if first and first <= 10 else 0.0
    m["MRR"] = (1.0 / first) if first else 0.0
    m["first_rank"] = first
    if len(gt_sections) > 1:
        for k in KS:
            if stage == "production" and k > PRODUCTION_TOP_N:
                m[f"AllSec@{k}"] = None
                continue
            found = set()
            for c in ids[:k]:
                found.update(rel.get(c, []))
            m[f"AllSec@{k}"] = int(all(s in found for s in gt_sections))
    return m


METRICS = [f"R@{k}" for k in KS] + ["MRR@10", "MRR"] + [f"AllSec@{k}" for k in KS]


def aggregate(pq_rows, questions):
    qmap = {q["question_id"]: q for q in questions}

    def summarise(rows):
        out = {}
        for stage in STAGES:
            out[stage] = {}
            for met in METRICS:
                vals = [r[stage].get(met) for r in rows if r[stage].get(met) is not None]
                out[stage][met] = {"mean": round(float(np.mean(vals)), 4) if vals else None, "n": len(vals)}
        return out

    res = {"overall": {"all": {"n": len(pq_rows), **summarise(pq_rows)}}}
    for g in GROUPS:
        buckets = defaultdict(list)
        for r in pq_rows:
            buckets[qmap[r["question_id"]][g]].append(r)
        res[g] = {v: {"n": len(rs), **summarise(rs)} for v, rs in sorted(buckets.items())}
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Paired comparison: bootstrap CI + McNemar (binary) / Wilcoxon (reciprocal rank)
# ──────────────────────────────────────────────────────────────────────────────
def paired_stats(x, y, binary, rng):
    from scipy.stats import binomtest, wilcoxon
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = y - x
    idx = rng.integers(0, len(d), size=(BOOTSTRAP, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if binary:
        b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
        p = binomtest(c, b + c, 0.5).pvalue if b + c else 1.0
        test = f"McNemar exact (only_reference={b}, only_compared={c})"
    else:
        p = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
        test = "Wilcoxon signed-rank"
    return float(d.mean()), float(lo), float(hi), float(p), test


def compare(per_q, questions, out_dir):
    qmap = {q["question_id"]: q for q in questions}
    rng = np.random.default_rng(SEED)
    rows = []
    for a, b in COMPARISONS:
        if a not in per_q or b not in per_q:
            continue
        A = {r["question_id"]: r for r in per_q[a]}
        B = {r["question_id"]: r for r in per_q[b]}
        groups = [("overall", "all", list(A))]
        for g in ["doc_type", "query_language", "question_type", "cross_lingual_direction"]:
            buckets = defaultdict(list)
            for qid in A:
                buckets[qmap[qid][g]].append(qid)
            groups += [(g, v, ids) for v, ids in sorted(buckets.items())]
        for stage in ["dense", "hybrid", "rerank", "production"]:
            for met in ["R@1", "R@5", "R@10", "MRR@10"]:
                for g, v, ids in groups:
                    xs = [A[i][stage][met] for i in ids]
                    ys = [B[i][stage][met] for i in ids]
                    if any(val is None for val in xs + ys):
                        continue
                    diff, lo, hi, p, test = paired_stats(xs, ys, met != "MRR@10", rng)
                    rows.append({"reference": a, "compared": b, "stage": stage, "metric": met,
                                 "group": g, "value": v, "n": len(ids),
                                 "mean_reference": round(float(np.mean(xs)), 4),
                                 "mean_compared": round(float(np.mean(ys)), 4),
                                 "difference_compared_minus_reference": round(diff, 4), "ci95_low": round(lo, 4),
                                 "ci95_high": round(hi, 4), "p_value": round(p, 5), "test": test})
    path = os.path.join(out_dir, "comparison.csv")
    if rows:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="main.py")
    ap.add_argument("--chunks", default="evaluation/frozen/chunks.jsonl")
    ap.add_argument("--questions", default="evaluation/dataset/questions.jsonl")
    ap.add_argument("--qrels", default="evaluation/labels/qrels.jsonl")
    ap.add_argument("--qdrant-path", default="evaluation/qdrant_eval")
    ap.add_argument("--results", default="evaluation/results")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--compare-with", default=None,
                    help="an earlier results folder; its configurations are reused for the "
                         "comparison table so they need not be run again")
    args = ap.parse_args()

    chunks = read_jsonl(args.chunks)
    questions = read_jsonl(args.questions)
    qrels = read_jsonl(args.qrels)
    assert len(questions) == 200 and len({c["chunk_id"] for c in chunks}) == len(chunks)
    rel = defaultdict(dict)
    for r in qrels:
        rel[r["question_id"]][r["chunk_id"]] = r["matched_sections"]
    missing = [q["question_id"] for q in questions if not rel[q["question_id"]]]
    if missing:
        sys.exit(f"[exp] ABORT: questions without relevant chunks: {missing}")

    chunks_sha = sha256(args.chunks)
    main_fns = load_main_functions(args.main)
    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.results, run_id)   # created when the first results are written

    print("[exp] loading reranker ...")
    reranker = load_reranker()
    client = QdrantClient(path=args.qdrant_path)

    manifest = {
        "run_id": run_id, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "inputs_sha256": {"main.py": sha256(args.main), "chunks": chunks_sha,
                          "questions": sha256(args.questions), "qrels": sha256(args.qrels)},
        "settings": {"scope": "all 8 documents", "top_k": TOP_K, "production_top_n": PRODUCTION_TOP_N,
                     "rrf_k": 60, "dense_floor": 0.02, "rerank_floor": -6, "rerank_model": RERANK_MODEL,
                     "rerank_passage_chars": 512, "bootstrap": BOOTSTRAP, "seed": SEED},
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        **{p: pkg(p) for p in ["sentence-transformers", "transformers", "torch",
                                                "qdrant-client", "rank-bm25", "regex", "numpy", "scipy"]}},
        "configs": {},
    }

    per_q, overall = {}, {}
    for name in args.configs:
        cfg = CONFIGS[name]
        raw, info = run_config(name, cfg, main_fns, client, reranker, chunks, chunks_sha, questions, args.reindex)
        manifest["configs"][name] = info
        cdir = os.path.join(out_dir, name)
        os.makedirs(cdir, exist_ok=True)
        rows = []
        with open(os.path.join(cdir, "raw_results.jsonl"), "w", encoding="utf-8") as f:
            for q, r in zip(questions, raw):
                qrel = rel[q["question_id"]]
                stages = {s: [{"rank": i + 1, "chunk_id": c, "score": round(sc, 6), "relevant": int(c in qrel)}
                              for i, (c, sc) in enumerate(r["stages"][s][:TOP_K])] for s in STAGES}
                f.write(json.dumps({"question_id": q["question_id"], "stages": stages}, ensure_ascii=False) + "\n")
                row = {"question_id": q["question_id"]}
                for s in STAGES:
                    row[s] = per_question_metrics(r["stages"][s], qrel, q["gt_section_list"], s)
                rows.append(row)
        with open(os.path.join(cdir, "per_question.jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics = aggregate(rows, questions)
        with open(os.path.join(cdir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        per_q[name] = rows
        overall[name] = metrics["overall"]["all"]

    client.close()
    os.makedirs(out_dir, exist_ok=True)
    if args.compare_with:
        for name in CONFIGS:
            f = os.path.join(args.compare_with, name, "per_question.jsonl")
            if name not in per_q and os.path.exists(f):
                per_q[name] = read_jsonl(f)
                print(f"[exp] reusing earlier results for {name} from {args.compare_with}")
        manifest["compare_with"] = os.path.abspath(args.compare_with)
    comp = compare(per_q, questions, out_dir)
    with open(os.path.join(out_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ── console summary (overall only; details are in the files) ──
    print("\n[exp] OVERALL (200 questions, search over all 8 documents)")
    print(f" {'config':<19}{'stage':<12}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR@10':>8}")
    for name, m in overall.items():
        for s in ["bm25", "dense", "hybrid", "rerank", "production"]:
            v = lambda k: "-" if m[s][k]["mean"] is None else f"{m[s][k]['mean']:.3f}"
            print(f" {name:<19}{s:<12}{v('R@1'):>7}{v('R@5'):>7}{v('R@10'):>7}{v('MRR@10'):>8}")
    print(f"\n[exp] {len(comp)} paired comparisons written to comparison.csv")
    print(f"[exp] results: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()