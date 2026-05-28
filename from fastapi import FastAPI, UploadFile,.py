
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchAny, MatchValue, FilterSelector,
)

from sse_starlette.sse import EventSourceResponse

import os, uuid, json, shutil, sqlite3, datetime, re

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://127.0.0.1:5173","http://localhost:3000","http://127.0.0.1:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker    = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

qdrant = QdrantClient(path="./qdrant_data")
if "rag_docs" not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection("rag_docs", vectors_config=VectorParams(size=384, distance=Distance.COSINE))

# ── SQLite ──
def db():
    conn = sqlite3.connect("docwise.db", check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    c = db()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_docs (
            id TEXT PRIMARY KEY, chat_id TEXT NOT NULL,
            filename TEXT NOT NULL, display_name TEXT NOT NULL,
            pages INTEGER, file_size INTEGER DEFAULT 0, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, sources TEXT, created_at TEXT NOT NULL
        );
    """)
    try: c.execute("ALTER TABLE chat_docs ADD COLUMN file_size INTEGER DEFAULT 0"); c.commit()
    except: pass
    c.commit(); c.close()

init_db()

chat_memory:    dict[str, list] = {}
chat_summaries: dict[str, str]  = {}
WINDOW = 6; SUMMARY_TRIGGER = 12

def _build_history(cid):
    msgs    = chat_memory.get(cid, [])
    summary = chat_summaries.get(cid)
    history = []
    if summary:
        history.append({"role":"system","content":f"Summary of earlier conversation:\n{summary}"})
    history.extend(msgs[-(WINDOW*2):])
    return history

def _maybe_summarise(cid):
    msgs = chat_memory.get(cid, [])
    if len(msgs) < SUMMARY_TRIGGER: return
    to_sum = msgs[:-(WINDOW*2)]
    if not to_sum: return
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=256,
            messages=[
                {"role":"system","content":"Summarise the conversation in 3-5 concise sentences. Return only the summary."},
                {"role":"user","content":"\n".join(f"{m['role'].upper()}: {m['content']}" for m in to_sum)},
            ],
        )
        chat_summaries[cid] = resp.choices[0].message.content.strip()
        chat_memory[cid]    = msgs[-(WINDOW*2):]
    except: pass

def now(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def extract_pages(path):
    pages = []
    for i, page in enumerate(PdfReader(path).pages):
        t = page.extract_text()
        if t and t.strip(): pages.append({"page": i+1, "text": t.strip()})
    return pages

def chunk_text(text, chunk_size=700, overlap=120):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap
    return chunks

def tokenize(text): return re.findall(r"\w+", text.lower())

def hybrid_search(query, cid, doc_ids, top_k=20):
    qv = embed_model.encode(query).tolist()
    hits = qdrant.query_points(
        "rag_docs", query=qv, limit=top_k,
        query_filter=Filter(must=[
            FieldCondition(key="chat_id", match=MatchValue(value=cid)),
            FieldCondition(key="doc_id",  match=MatchAny(any=doc_ids)),
        ])
    ).points
    hits = [h for h in hits if h.score > 0.15]
    if not hits: return []
    corpus   = [h.payload["text"] for h in hits]
    bm25     = BM25Okapi([tokenize(t) for t in corpus])
    bm25_sc  = bm25.get_scores(tokenize(query))
    RRF_K    = 60
    d_rank   = {h.id: r+1 for r,h in enumerate(hits)}
    b_order  = sorted(range(len(hits)), key=lambda i: bm25_sc[i], reverse=True)
    b_rank   = {hits[i].id: r+1 for r,i in enumerate(b_order)}
    rrf      = {h.id: 1/(RRF_K+d_rank[h.id]) + 1/(RRF_K+b_rank[h.id]) for h in hits}
    return sorted(hits, key=lambda h: rrf[h.id], reverse=True)

def rerank(query, candidates, top_n=5):
    if not candidates: return []
    pairs  = [(query, c.payload["text"][:512]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top    = [(s,c) for s,c in ranked if s > -3][:top_n]
    for s,c in top: c.payload["rerank_score"] = float(s)
    return [c for _,c in top]

# ── FIX #11 + #4: aggregate sources — accurate pages, filter low-confidence ──
RERANK_THRESHOLD = -2.5

def aggregate_sources(chunks):
    """
    Group chunks by doc_id. Collect ALL page numbers per document.
    Filter docs whose best rerank_score is below threshold (overview/summary queries).
    Return at most 1 source entry (the best scoring doc) with all its pages.
    If no chunk scores above threshold → return [] (no sources shown).
    """
    by_doc: dict = {}
    for chunk in chunks:
        p      = chunk.payload
        doc_id = p.get("doc_id","unknown")
        page   = p.get("page")
        score  = p.get("rerank_score", 0.0)
        fname  = p.get("filename","Unknown")
        if doc_id not in by_doc:
            by_doc[doc_id] = {"doc_id":doc_id,"filename":fname,"pages":set(),"best_score":score}
        if page: by_doc[doc_id]["pages"].add(page)
        if score > by_doc[doc_id]["best_score"]: by_doc[doc_id]["best_score"] = score

    # FIX #4: discard docs whose best chunk didn't score above threshold
    valid = [d for d in by_doc.values() if d["best_score"] > RERANK_THRESHOLD and d["pages"]]

    if not valid: return []   # ← no sources for overview/summary answers

    # Sort by best_score, return the top-1 source with all its pages
    best = sorted(valid, key=lambda d: d["best_score"], reverse=True)[:1]
    return [{
        "doc_id":   d["doc_id"],
        "filename": d["filename"],
        "pages":    sorted(d["pages"]),   # e.g. [2,3,4]
    } for d in best]

class SearchRequest(BaseModel):
    query: str; chat_id: str; doc_ids: List[str]

class RenameBody(BaseModel):
    name: str

# ════════════════════════════════════════
# CHAT ROUTES
# ════════════════════════════════════════
@app.get("/chats")
def get_chats():
    c = db(); rows = c.execute("SELECT * FROM chats ORDER BY created_at DESC").fetchall(); c.close()
    return [dict(r) for r in rows]

@app.post("/chats")
def create_chat():
    cid = str(uuid.uuid4()); c = db()
    c.execute("INSERT INTO chats VALUES (?,?,?)", (cid,"New Chat",now()))
    c.commit(); c.close()
    return {"id":cid,"name":"New Chat","created_at":now()}

@app.put("/chats/{cid}")
def rename_chat(cid, body: RenameBody):
    c = db(); c.execute("UPDATE chats SET name=? WHERE id=?", (body.name,cid)); c.commit(); c.close()
    return {"status":"ok"}

@app.delete("/chats/{cid}")
def delete_chat(cid):
    c = db()
    docs = c.execute("SELECT * FROM chat_docs WHERE chat_id=?", (cid,)).fetchall()
    for d in docs:
        try: qdrant.delete("rag_docs", points_selector=FilterSelector(filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=d["id"]))])))
        except: pass
        try: os.remove(f"uploads/{d['id']}_{d['filename']}")
        except: pass
    c.execute("DELETE FROM messages WHERE chat_id=?", (cid,))
    c.execute("DELETE FROM chat_docs WHERE chat_id=?", (cid,))
    c.execute("DELETE FROM chats WHERE id=?", (cid,))
    c.commit(); c.close()
    chat_memory.pop(cid,None); chat_summaries.pop(cid,None)
    return {"status":"deleted"}

@app.get("/chats/{cid}/messages")
def get_messages(cid):
    c = db(); rows = c.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY created_at ASC",(cid,)).fetchall(); c.close()
    result = []
    for r in rows:
        row = dict(r); row["sources"] = json.loads(row["sources"]) if row["sources"] else []; result.append(row)
    return result

# ════════════════════════════════════════
# DOCUMENT ROUTES
# ════════════════════════════════════════
@app.get("/chats/{cid}/documents")
def get_documents(cid):
    c = db(); rows = c.execute("SELECT * FROM chat_docs WHERE chat_id=? ORDER BY created_at ASC",(cid,)).fetchall(); c.close()
    return [dict(r) for r in rows]

@app.post("/chats/{cid}/upload")
async def upload_pdf(cid, file: UploadFile = File(...)):
    try:
        c = db()
        if not c.execute("SELECT id FROM chats WHERE id=?",(cid,)).fetchone(): raise HTTPException(404,"Chat not found")
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"): raise HTTPException(400,"Only PDF files are allowed")
        doc_id = str(uuid.uuid4())
        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{doc_id}_{filename}"
        with open(path,"wb") as f: shutil.copyfileobj(file.file,f)
        file_size = os.path.getsize(path)
        pages = extract_pages(path)
        if not pages: os.remove(path); return {"error":"No readable text found in PDF"}
        points = []
        for item in pages:
            for chunk in chunk_text(item["text"]):
                points.append(PointStruct(
                    id=str(uuid.uuid4()), vector=embed_model.encode(chunk[:4000]).tolist(),
                    payload={"chat_id":cid,"doc_id":doc_id,"text":chunk,"page":item["page"],"filename":filename}
                ))
        for i in range(0,len(points),100): qdrant.upsert("rag_docs",points=points[i:i+100])
        dn = " ".join(w.capitalize() for w in os.path.splitext(filename)[0].replace("_"," ").replace("-"," ").split()[:5])
        c.execute("INSERT INTO chat_docs VALUES (?,?,?,?,?,?,?)",(doc_id,cid,filename,dn,len(pages),file_size,now()))
        chat = c.execute("SELECT name FROM chats WHERE id=?",(cid,)).fetchone()
        if chat and chat["name"]=="New Chat": c.execute("UPDATE chats SET name=? WHERE id=?",(dn,cid))
        c.commit(); c.close()
        return {"doc_id":doc_id,"filename":filename,"display_name":dn,"pages":len(pages),"file_size":file_size}
    except HTTPException: raise
    except Exception as e: return {"error":str(e)}

@app.put("/chats/{cid}/documents/{did}")
def rename_document(cid,did,body:RenameBody):
    c=db(); c.execute("UPDATE chat_docs SET display_name=? WHERE id=? AND chat_id=?",(body.name,did,cid)); c.commit(); c.close()
    return {"status":"ok"}

@app.delete("/chats/{cid}/documents/{did}")
def delete_document(cid,did):
    c=db(); row=c.execute("SELECT filename FROM chat_docs WHERE id=? AND chat_id=?",(did,cid)).fetchone()
    try: qdrant.delete("rag_docs",points_selector=FilterSelector(filter=Filter(must=[FieldCondition(key="doc_id",match=MatchValue(value=did))])))
    except: pass
    if row:
        try: os.remove(f"uploads/{did}_{row['filename']}")
        except: pass
    c.execute("DELETE FROM chat_docs WHERE id=? AND chat_id=?",(did,cid)); c.commit(); c.close()
    return {"status":"deleted"}

# ════════════════════════════════════════
# SEARCH + STREAM
# ════════════════════════════════════════
@app.post("/chats/{cid}/search-stream")
async def search_stream(cid, req: SearchRequest):
    try:
        if not req.doc_ids: return {"error":"No documents selected"}
        if cid not in chat_memory: chat_memory[cid]=[]
        candidates = hybrid_search(req.query, cid, req.doc_ids, top_k=20)
        if not candidates:
            async def empty():
                yield {"data":json.dumps({"text":"I could not find relevant information in the selected documents."})}
            return EventSourceResponse(empty())
        top_chunks = rerank(req.query, candidates, top_n=5)
        context    = "".join(f"[Page {c.payload['page']}]\n{c.payload['text']}\n\n" for c in top_chunks)
        sources    = aggregate_sources(top_chunks)
        history    = _build_history(cid)
        async def generate():
            try:
                resp = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=1024, stream=True,
                    messages=[
                        {"role":"system","content":(
                            "You are a precise document assistant. "
                            "Answer ONLY from the provided context. "
                            "If the answer is not in the context, say so clearly. "
                            "Format with markdown: **bold** for key terms, bullet lists, code blocks for code. "
                            "Do NOT mention filenames or page numbers in your answer. Be direct and concise."
                        )},
                        *history,
                        {"role":"user","content":f"Context:\n{context}\n\nQuestion:\n{req.query}"},
                    ],
                )
                full = ""
                for chunk in resp:
                    t = chunk.choices[0].delta.content
                    if t: full+=t; yield {"data":json.dumps({"text":t})}
                c = db()
                c.execute("INSERT INTO messages (chat_id,role,content,sources,created_at) VALUES (?,?,?,?,?)",(cid,"user",req.query,None,now()))
                c.execute("INSERT INTO messages (chat_id,role,content,sources,created_at) VALUES (?,?,?,?,?)",(cid,"assistant",full,json.dumps(sources),now()))
                c.commit(); c.close()
                chat_memory[cid].append({"role":"user","content":req.query})
                chat_memory[cid].append({"role":"assistant","content":full})
                _maybe_summarise(cid)
                yield {"data":json.dumps({"sources":sources,"done":True})}
            except Exception as e:
                yield {"data":json.dumps({"error":str(e)})}
        return EventSourceResponse(generate())
    except Exception as e:
        return {"error":str(e)}

# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    FilterSelector,
)

from sse_starlette.sse import EventSourceResponse

import os
import uuid
import json
import shutil
import sqlite3
import datetime
import re

# ════════════════════════════════════════
# ENV
# ════════════════════════════════════════
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ════════════════════════════════════════
# FASTAPI
# ════════════════════════════════════════
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════
# MODELS
# ════════════════════════════════════════
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

# ════════════════════════════════════════
# QDRANT
# ════════════════════════════════════════
qdrant = QdrantClient(path="./qdrant_data")

existing = [c.name for c in qdrant.get_collections().collections]

if "rag_docs" not in existing:
    qdrant.create_collection(
        collection_name="rag_docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# ════════════════════════════════════════
# SQLITE
# ════════════════════════════════════════
def db():
    conn = sqlite3.connect("docwise.db", check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    c = db()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_docs (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            display_name TEXT NOT NULL,
            pages INTEGER,
            file_size INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS _migrations (key TEXT PRIMARY KEY);
    """)

    try:
        c.execute("ALTER TABLE chat_docs ADD COLUMN file_size INTEGER DEFAULT 0")
        c.commit()
    except Exception:
        pass

    c.commit()
    c.close()


init_db()

# ════════════════════════════════════════
# MEMORY  (sliding window + summarisation)
# ════════════════════════════════════════
chat_memory: dict[str, list] = {}
chat_summaries: dict[str, str] = {}

WINDOW = 6
SUMMARY_TRIGGER = 12


def _build_history(cid: str) -> list:
    msgs = chat_memory.get(cid, [])
    summary = chat_summaries.get(cid)
    history = []
    if summary:
        history.append({
            "role": "system",
            "content": f"Summary of earlier conversation:\n{summary}",
        })
    history.extend(msgs[-WINDOW * 2:])
    return history


def _maybe_summarise(cid: str):
    msgs = chat_memory.get(cid, [])
    if len(msgs) < SUMMARY_TRIGGER:
        return
    to_summarise = msgs[:-WINDOW * 2]
    if not to_summarise:
        return
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=256,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Summarise the following "
                        "conversation in 3-5 concise sentences capturing the key "
                        "topics and decisions. Return only the summary, no preamble."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        f"{m['role'].upper()}: {m['content']}" for m in to_summarise
                    ),
                },
            ],
        )
        chat_summaries[cid] = resp.choices[0].message.content.strip()
        chat_memory[cid] = msgs[-WINDOW * 2:]
    except Exception:
        pass

# ════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════
def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_pages(path: str):
    pages = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    return pages


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ════════════════════════════════════════
# HYBRID SEARCH  (dense + BM25 fusion)
# ════════════════════════════════════════
def hybrid_search(query: str, cid: str, doc_ids: list[str], top_k: int = 20) -> list:
    query_vec = embed_model.encode(query).tolist()

    dense_hits = qdrant.query_points(
        collection_name="rag_docs",
        query=query_vec,
        limit=top_k,
        query_filter=Filter(
            must=[
                FieldCondition(key="chat_id", match=MatchValue(value=cid)),
                FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)),
            ]
        ),
    ).points

    dense_hits = [h for h in dense_hits if h.score > 0.15]

    if not dense_hits:
        return []

    corpus = [h.payload["text"] for h in dense_hits]
    tokenised_corpus = [tokenize(t) for t in corpus]
    bm25 = BM25Okapi(tokenised_corpus)
    bm25_scores = bm25.get_scores(tokenize(query))

    RRF_K = 60
    dense_rank = {h.id: rank + 1 for rank, h in enumerate(dense_hits)}
    bm25_order = sorted(range(len(dense_hits)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_rank = {dense_hits[i].id: rank + 1 for rank, i in enumerate(bm25_order)}

    rrf_scores = {}
    for h in dense_hits:
        d_rank = dense_rank.get(h.id, top_k + 1)
        b_rank = bm25_rank.get(h.id, top_k + 1)
        rrf_scores[h.id] = (1 / (RRF_K + d_rank)) + (1 / (RRF_K + b_rank))

    merged = sorted(dense_hits, key=lambda h: rrf_scores[h.id], reverse=True)
    return merged


# ════════════════════════════════════════
# RERANKING
# ════════════════════════════════════════
def rerank(query: str, candidates: list, top_n: int = 5) -> list:
    if not candidates:
        return []
    pairs = [(query, c.payload["text"][:512]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    filtered = [(score, c) for score, c in ranked if score > -3]
    top = filtered[:top_n]
    result = []
    for score, c in top:
        c.payload["rerank_score"] = float(score)
        result.append(c)
    return result


# ════════════════════════════════════════
# FIX #11: SOURCE AGGREGATION — accurate page numbers per document
#
# Strategy:
# 1. After reranking we have up to 5 chunks, possibly multiple per document.
# 2. Group all chunks by doc_id.
# 3. For each document, collect ALL page numbers from all its matching chunks.
#    This gives accurate multi-page attribution (e.g. "pp. 3, 7, 12").
# 4. Use the best rerank_score per doc for filtering/ordering.
# 5. Cap at 3 documents total.
# 6. FIX #4: Only include sources whose best chunk score > threshold.
#    This prevents showing sources for overview/summary questions where
#    no chunk had a strong direct match.
# ════════════════════════════════════════
RERANK_SCORE_THRESHOLD = -2.5  # chunks below this score are not trustworthy sources

def aggregate_sources(chunks: list) -> list:
    """
    Group chunks by doc_id, collecting all page numbers and tracking the
    best rerank_score per document. Returns deduplicated source list with
    accurate page information.
    """
    by_doc: dict[str, dict] = {}

    for chunk in chunks:
        payload = chunk.payload
        doc_id = payload.get("doc_id", "unknown")
        page = payload.get("page")
        score = payload.get("rerank_score", 0)
        filename = payload.get("filename", "Unknown")

        if doc_id not in by_doc:
            by_doc[doc_id] = {
                "doc_id": doc_id,
                "filename": filename,
                "pages": set(),
                "rerank_score": score,
            }

        # Collect all pages this doc contributed
        if page:
            by_doc[doc_id]["pages"].add(page)

        # Track the best score for this document
        if score > by_doc[doc_id]["rerank_score"]:
            by_doc[doc_id]["rerank_score"] = score

    # FIX #4: filter out docs whose best chunk didn't score above threshold
    valid = [
        doc for doc in by_doc.values()
        if doc["rerank_score"] > RERANK_SCORE_THRESHOLD and len(doc["pages"]) > 0
    ]

    # Sort by rerank_score descending, cap at 3 docs
    sorted_sources = sorted(valid, key=lambda x: x["rerank_score"], reverse=True)[:3]

    # Convert sets to sorted lists for JSON serialisation
    result = []
    for s in sorted_sources:
        result.append({
            "doc_id": s["doc_id"],
            "filename": s["filename"],
            "pages": sorted(list(s["pages"])),   # sorted list, e.g. [3, 7, 12]
            "rerank_score": round(s["rerank_score"], 4),
        })

    return result


# ════════════════════════════════════════
# REQUEST MODELS
# ════════════════════════════════════════
class SearchRequest(BaseModel):
    query: str
    chat_id: str
    doc_ids: List[str]


class RenameBody(BaseModel):
    name: str

# ════════════════════════════════════════
# CHAT ROUTES
# ════════════════════════════════════════
@app.get("/chats")
def get_chats():
    c = db()
    rows = c.execute("SELECT * FROM chats ORDER BY created_at DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]


@app.post("/chats")
def create_chat():
    cid = str(uuid.uuid4())
    c = db()
    c.execute("INSERT INTO chats VALUES (?,?,?)", (cid, "New Chat", now()))
    c.commit()
    c.close()
    return {"id": cid, "name": "New Chat", "created_at": now()}


@app.put("/chats/{cid}")
def rename_chat(cid: str, body: RenameBody):
    c = db()
    c.execute("UPDATE chats SET name=? WHERE id=?", (body.name, cid))
    c.commit()
    c.close()
    return {"status": "ok"}


@app.delete("/chats/{cid}")
def delete_chat(cid: str):
    c = db()
    docs = c.execute("SELECT * FROM chat_docs WHERE chat_id=?", (cid,)).fetchall()

    for d in docs:
        try:
            qdrant.delete(
                collection_name="rag_docs",
                points_selector=FilterSelector(
                    filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=d["id"]))])
                ),
            )
        except Exception:
            pass
        try:
            os.remove(f"uploads/{d['id']}_{d['filename']}")
        except Exception:
            pass

    c.execute("DELETE FROM messages WHERE chat_id=?", (cid,))
    c.execute("DELETE FROM chat_docs WHERE chat_id=?", (cid,))
    c.execute("DELETE FROM chats WHERE id=?", (cid,))
    c.commit()
    c.close()

    chat_memory.pop(cid, None)
    chat_summaries.pop(cid, None)
    return {"status": "deleted"}

# ════════════════════════════════════════
# MESSAGE ROUTES
# ════════════════════════════════════════
@app.get("/chats/{cid}/messages")
def get_messages(cid: str):
    c = db()
    rows = c.execute(
        "SELECT * FROM messages WHERE chat_id=? ORDER BY created_at ASC", (cid,)
    ).fetchall()
    c.close()
    result = []
    for r in rows:
        row = dict(r)
        row["sources"] = json.loads(row["sources"]) if row["sources"] else []
        result.append(row)
    return result

# ════════════════════════════════════════
# DOCUMENT ROUTES
# ════════════════════════════════════════
@app.get("/chats/{cid}/documents")
def get_documents(cid: str):
    c = db()
    rows = c.execute(
        "SELECT * FROM chat_docs WHERE chat_id=? ORDER BY created_at ASC", (cid,)
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


@app.post("/chats/{cid}/upload")
async def upload_pdf(cid: str, file: UploadFile = File(...)):
    try:
        c = db()
        exists = c.execute("SELECT id FROM chats WHERE id=?", (cid,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        doc_id = str(uuid.uuid4())
        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{doc_id}_{filename}"

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_size = os.path.getsize(path)

        pages = extract_pages(path)
        if not pages:
            os.remove(path)
            return {"error": "No readable text found in PDF"}

        points = []
        for item in pages:
            for chunk in chunk_text(item["text"]):
                vector = embed_model.encode(chunk[:4000]).tolist()
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "chat_id": cid,
                            "doc_id": doc_id,
                            "text": chunk,
                            "page": item["page"],
                            "filename": filename,
                        },
                    )
                )

        for i in range(0, len(points), 100):
            qdrant.upsert(collection_name="rag_docs", points=points[i : i + 100])

        display_name = os.path.splitext(filename)[0]
        display_name = display_name.replace("_", " ").replace("-", " ")
        display_name = " ".join(w.capitalize() for w in display_name.split()[:5])

        c.execute(
            "INSERT INTO chat_docs VALUES (?,?,?,?,?,?,?)",
            (doc_id, cid, filename, display_name, len(pages), file_size, now()),
        )

        chat = c.execute("SELECT name FROM chats WHERE id=?", (cid,)).fetchone()
        if chat and chat["name"] == "New Chat":
            c.execute("UPDATE chats SET name=? WHERE id=?", (display_name, cid))

        c.commit()
        c.close()

        return {
            "doc_id": doc_id,
            "filename": filename,
            "display_name": display_name,
            "pages": len(pages),
            "file_size": file_size,
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@app.put("/chats/{cid}/documents/{did}")
def rename_document(cid: str, did: str, body: RenameBody):
    c = db()
    c.execute(
        "UPDATE chat_docs SET display_name=? WHERE id=? AND chat_id=?",
        (body.name, did, cid),
    )
    c.commit()
    c.close()
    return {"status": "ok"}


@app.delete("/chats/{cid}/documents/{did}")
def delete_document(cid: str, did: str):
    c = db()
    row = c.execute(
        "SELECT filename FROM chat_docs WHERE id=? AND chat_id=?", (did, cid)
    ).fetchone()

    try:
        qdrant.delete(
            collection_name="rag_docs",
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=did))])
            ),
        )
    except Exception:
        pass

    if row:
        try:
            os.remove(f"uploads/{did}_{row['filename']}")
        except Exception:
            pass

    c.execute("DELETE FROM chat_docs WHERE id=? AND chat_id=?", (did, cid))
    c.commit()
    c.close()
    return {"status": "deleted"}


# ════════════════════════════════════════
# SEARCH + STREAM  (hybrid + reranked)
# FIX #11: uses aggregate_sources for accurate page-level attribution
# FIX #4: filters low-confidence sources
# ════════════════════════════════════════
@app.post("/chats/{cid}/search-stream")
async def search_stream(cid: str, req: SearchRequest):
    try:
        if not req.doc_ids:
            return {"error": "No documents selected"}

        if cid not in chat_memory:
            chat_memory[cid] = []

        # 1. Hybrid retrieval
        candidates = hybrid_search(req.query, cid, req.doc_ids, top_k=20)

        if not candidates:
            async def empty():
                yield {"data": json.dumps({"text": "I could not find relevant information in the selected documents."})}
            return EventSourceResponse(empty())

        # 2. Rerank with cross-encoder (keep top 5 chunks for context quality)
        top_chunks = rerank(req.query, candidates, top_n=5)

        # 3. Build context from ALL top chunks (good for answer quality)
        context = ""
        for chunk in top_chunks:
            payload = chunk.payload
            context += f"[Page {payload['page']}]\n{payload['text']}\n\n"

        # FIX #11: Aggregate sources — group by doc, collect all pages, filter by score
        # This produces accurate page-level attribution per document.
        sources = aggregate_sources(top_chunks)

        history = _build_history(cid)

        async def generate():
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    max_tokens=1024,
                    stream=True,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise document assistant. "
                                "Answer ONLY from the provided context. "
                                "If the answer is not in the context, say so clearly. "
                                "Format your response with markdown: use **bold** for key terms, "
                                "bullet lists where appropriate, and code blocks for any code. "
                                "Do not mention filenames or page numbers in your answer — "
                                "sources are shown separately. Be direct and concise."
                            ),
                        },
                        *history,
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion:\n{req.query}",
                        },
                    ],
                )

                full_answer = ""
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if not delta or not delta.content:
                        continue
                    text = delta.content
                    full_answer += text
                    yield {"data": json.dumps({"text": text})}

                # Persist messages
                c = db()
                c.execute(
                    "INSERT INTO messages (chat_id,role,content,sources,created_at) VALUES (?,?,?,?,?)",
                    (cid, "user", req.query, None, now()),
                )
                c.execute(
                    "INSERT INTO messages (chat_id,role,content,sources,created_at) VALUES (?,?,?,?,?)",
                    (cid, "assistant", full_answer, json.dumps(sources), now()),
                )
                c.commit()
                c.close()

                chat_memory[cid].append({"role": "user", "content": req.query})
                chat_memory[cid].append({"role": "assistant", "content": full_answer})
                _maybe_summarise(cid)

                yield {"data": json.dumps({"sources": sources, "done": True})}

            except Exception as e:
                yield {"data": json.dumps({"error": str(e)})}

        return EventSourceResponse(generate())

    except Exception as e:
        return {"error": str(e)}

# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    FilterSelector,
)

from sse_starlette.sse import EventSourceResponse

import os
import uuid
import json
import shutil
import sqlite3
import datetime
import re

# ════════════════════════════════════════
# ENV
# ════════════════════════════════════════
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ════════════════════════════════════════
# FASTAPI
# ════════════════════════════════════════
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════
# MODELS
# ════════════════════════════════════════
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

# ════════════════════════════════════════
# QDRANT
# ════════════════════════════════════════
qdrant = QdrantClient(path="./qdrant_data")

existing = [c.name for c in qdrant.get_collections().collections]

if "rag_docs" not in existing:
    qdrant.create_collection(
        collection_name="rag_docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# ════════════════════════════════════════
# SQLITE
# ════════════════════════════════════════
def db():
    conn = sqlite3.connect("docwise.db", check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    c = db()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_docs (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            display_name TEXT NOT NULL,
            pages INTEGER,
            file_size INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS _migrations (key TEXT PRIMARY KEY);
    """)

    try:
        c.execute("ALTER TABLE chat_docs ADD COLUMN file_size INTEGER DEFAULT 0")
        c.commit()
    except Exception:
        pass

    c.commit()
    c.close()


init_db()

# ════════════════════════════════════════
# MEMORY  (sliding window + summarisation)
# ════════════════════════════════════════
chat_memory: dict[str, list] = {}
chat_summaries: dict[str, str] = {}

WINDOW = 6
SUMMARY_TRIGGER = 12


def _build_history(cid: str) -> list:
    msgs = chat_memory.get(cid, [])
    summary = chat_summaries.get(cid)
    history = []
    if summary:
        history.append({
            "role": "system",
            "content": f"Summary of earlier conversation:\n{summary}",
        })
    history.extend(msgs[-WINDOW * 2:])
    return history


def _maybe_summarise(cid: str):
    msgs = chat_memory.get(cid, [])
    if len(msgs) < SUMMARY_TRIGGER:
        return
    to_summarise = msgs[:-WINDOW * 2]
    if not to_summarise:
        return
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=256,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Summarise the following "
                        "conversation in 3-5 concise sentences capturing the key "
                        "topics and decisions. Return only the summary, no preamble."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        f"{m['role'].upper()}: {m['content']}" for m in to_summarise
                    ),
                },
            ],
        )
        chat_summaries[cid] = resp.choices[0].message.content.strip()
        chat_memory[cid] = msgs[-WINDOW * 2:]
    except Exception:
        pass

# ════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════
def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_pages(path: str):
    pages = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    return pages


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ════════════════════════════════════════
# HYBRID SEARCH  (dense + BM25 fusion)
# ════════════════════════════════════════
def hybrid_search(query: str, cid: str, doc_ids: list[str], top_k: int = 20) -> list:
    query_vec = embed_model.encode(query).tolist()

    dense_hits = qdrant.query_points(
        collection_name="rag_docs",
        query=query_vec,
        limit=top_k,
        query_filter=Filter(
            must=[
                FieldCondition(key="chat_id", match=MatchValue(value=cid)),
                FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)),
            ]
        ),
    ).points

    dense_hits = [h for h in dense_hits if h.score > 0.15]

    if not dense_hits:
        return []

    corpus = [h.payload["text"] for h in dense_hits]
    tokenised_corpus = [tokenize(t) for t in corpus]
    bm25 = BM25Okapi(tokenised_corpus)
    bm25_scores = bm25.get_scores(tokenize(query))

    RRF_K = 60
    dense_rank = {h.id: rank + 1 for rank, h in enumerate(dense_hits)}
    bm25_order = sorted(range(len(dense_hits)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_rank = {dense_hits[i].id: rank + 1 for rank, i in enumerate(bm25_order)}

    rrf_scores = {}
    for h in dense_hits:
        d_rank = dense_rank.get(h.id, top_k + 1)
        b_rank = bm25_rank.get(h.id, top_k + 1)
        rrf_scores[h.id] = (1 / (RRF_K + d_rank)) + (1 / (RRF_K + b_rank))

    merged = sorted(dense_hits, key=lambda h: rrf_scores[h.id], reverse=True)
    return merged


# ════════════════════════════════════════
# RERANKING
# ════════════════════════════════════════
def rerank(query: str, candidates: list, top_n: int = 5) -> list:
    if not candidates:
        return []
    pairs = [(query, c.payload["text"][:512]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    filtered = [(score, c) for score, c in ranked if score > -3]
    top = filtered[:top_n]
    result = []
    for score, c in top:
        c.payload["rerank_score"] = float(score)
        result.append(c)
    return result


# ════════════════════════════════════════
# FIX #5: SOURCE DEDUPLICATION — one source per document
# Strategy: after reranking we have up to 5 chunks, possibly multiple
# from the same document. We group by doc_id and keep only the BEST
# chunk per document (highest rerank_score). This means the context
# still benefits from multiple chunks, but only ONE source chip is
# shown per document in the UI. Cap at 3 docs total.
# ════════════════════════════════════════
def deduplicate_sources_by_doc(sources: list) -> list:
    """
    Group chunks by doc_id. For each document, keep the single chunk
    with the highest rerank_score. Return at most 3 documents.
    """
    best_per_doc: dict[str, dict] = {}
    for s in sources:
        key = s.get("doc_id") or s.get("filename") or "unknown"
        existing = best_per_doc.get(key)
        if existing is None:
            best_per_doc[key] = s
        else:
            # Keep the one with the higher rerank_score (if available)
            if s.get("rerank_score", 0) > existing.get("rerank_score", 0):
                best_per_doc[key] = s
    # Sort by rerank_score descending, cap at 3
    sorted_sources = sorted(
        best_per_doc.values(),
        key=lambda x: x.get("rerank_score", 0),
        reverse=True,
    )
    return sorted_sources[:3]


# ════════════════════════════════════════
# REQUEST MODELS
# ════════════════════════════════════════
class SearchRequest(BaseModel):
    query: str
    chat_id: str
    doc_ids: List[str]


class RenameBody(BaseModel):
    name: str

# ════════════════════════════════════════
# CHAT ROUTES
# ════════════════════════════════════════
@app.get("/chats")
def get_chats():
    c = db()
    rows = c.execute("SELECT * FROM chats ORDER BY created_at DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]


@app.post("/chats")
def create_chat():
    cid = str(uuid.uuid4())
    c = db()
    c.execute("INSERT INTO chats VALUES (?,?,?)", (cid, "New Chat", now()))
    c.commit()
    c.close()
    return {"id": cid, "name": "New Chat", "created_at": now()}


@app.put("/chats/{cid}")
def rename_chat(cid: str, body: RenameBody):
    c = db()
    c.execute("UPDATE chats SET name=? WHERE id=?", (body.name, cid))
    c.commit()
    c.close()
    return {"status": "ok"}


@app.delete("/chats/{cid}")
def delete_chat(cid: str):
    c = db()
    docs = c.execute("SELECT * FROM chat_docs WHERE chat_id=?", (cid,)).fetchall()

    for d in docs:
        try:
            qdrant.delete(
                collection_name="rag_docs",
                points_selector=FilterSelector(
                    filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=d["id"]))])
                ),
            )
        except Exception:
            pass
        try:
            os.remove(f"uploads/{d['id']}_{d['filename']}")
        except Exception:
            pass

    c.execute("DELETE FROM messages WHERE chat_id=?", (cid,))
    c.execute("DELETE FROM chat_docs WHERE chat_id=?", (cid,))
    c.execute("DELETE FROM chats WHERE id=?", (cid,))
    c.commit()
    c.close()

    chat_memory.pop(cid, None)
    chat_summaries.pop(cid, None)
    return {"status": "deleted"}

# ════════════════════════════════════════
# MESSAGE ROUTES
# ════════════════════════════════════════
@app.get("/chats/{cid}/messages")
def get_messages(cid: str):
    c = db()
    rows = c.execute(
        "SELECT * FROM messages WHERE chat_id=? ORDER BY created_at ASC", (cid,)
    ).fetchall()
    c.close()
    result = []
    for r in rows:
        row = dict(r)
        row["sources"] = json.loads(row["sources"]) if row["sources"] else []
        result.append(row)
    return result

# ════════════════════════════════════════
# DOCUMENT ROUTES
# ════════════════════════════════════════
@app.get("/chats/{cid}/documents")
def get_documents(cid: str):
    c = db()
    rows = c.execute(
        "SELECT * FROM chat_docs WHERE chat_id=? ORDER BY created_at ASC", (cid,)
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


@app.post("/chats/{cid}/upload")
async def upload_pdf(cid: str, file: UploadFile = File(...)):
    try:
        c = db()
        exists = c.execute("SELECT id FROM chats WHERE id=?", (cid,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        doc_id = str(uuid.uuid4())
        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{doc_id}_{filename}"

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_size = os.path.getsize(path)

        pages = extract_pages(path)
        if not pages:
            os.remove(path)
            return {"error": "No readable text found in PDF"}

        points = []
        for item in pages:
            for chunk in chunk_text(item["text"]):
                vector = embed_model.encode(chunk[:4000]).tolist()
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "chat_id": cid,
                            "doc_id": doc_id,
                            "text": chunk,
                            "page": item["page"],
                            "filename": filename,
                        },
                    )
                )

        for i in range(0, len(points), 100):
            qdrant.upsert(collection_name="rag_docs", points=points[i : i + 100])

        display_name = os.path.splitext(filename)[0]
        display_name = display_name.replace("_", " ").replace("-", " ")
        display_name = " ".join(w.capitalize() for w in display_name.split()[:5])

        c.execute(
            "INSERT INTO chat_docs VALUES (?,?,?,?,?,?,?)",
            (doc_id, cid, filename, display_name, len(pages), file_size, now()),
        )

        chat = c.execute("SELECT name FROM chats WHERE id=?", (cid,)).fetchone()
        if chat and chat["name"] == "New Chat":
            c.execute("UPDATE chats SET name=? WHERE id=?", (display_name, cid))

        c.commit()
        c.close()

        return {
            "doc_id": doc_id,
            "filename": filename,
            "display_name": display_name,
            "pages": len(pages),
            "file_size": file_size,
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@app.put("/chats/{cid}/documents/{did}")
def rename_document(cid: str, did: str, body: RenameBody):
    c = db()
    c.execute(
        "UPDATE chat_docs SET display_name=? WHERE id=? AND chat_id=?",
        (body.name, did, cid),
    )
    c.commit()
    c.close()
    return {"status": "ok"}


@app.delete("/chats/{cid}/documents/{did}")
def delete_document(cid: str, did: str):
    c = db()
    row = c.execute(
        "SELECT filename FROM chat_docs WHERE id=? AND chat_id=?", (did, cid)
    ).fetchone()

    try:
        qdrant.delete(
            collection_name="rag_docs",
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=did))])
            ),
        )
    except Exception:
        pass

    if row:
        try:
            os.remove(f"uploads/{did}_{row['filename']}")
        except Exception:
            pass

    c.execute("DELETE FROM chat_docs WHERE id=? AND chat_id=?", (did, cid))
    c.commit()
    c.close()
    return {"status": "deleted"}


# ════════════════════════════════════════
# FIX #1: PDF PAGE PREVIEW
# Ensure PyMuPDF errors are surfaced clearly.
# ════════════════════════════════════════
@app.get("/chats/{cid}/documents/{did}/page/{page_num}")
async def get_pdf_page(cid: str, did: str, page_num: int):
    """
    Returns a specific PDF page rendered as a PNG image.
    Requires: pip install pymupdf
    """
    try:
        import fitz  # PyMuPDF

        c = db()
        row = c.execute(
            "SELECT filename FROM chat_docs WHERE id=? AND chat_id=?", (did, cid)
        ).fetchone()
        c.close()

        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        path = f"uploads/{did}_{row['filename']}"
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="File not found on disk")

        doc = fitz.open(path)

        if page_num < 1 or page_num > len(doc):
            raise HTTPException(
                status_code=400,
                detail=f"Page {page_num} is out of range. This document has {len(doc)} pages.",
            )

        page = doc[page_num - 1]
        mat = fitz.Matrix(2.0, 2.0)   # 2× zoom for sharp rendering
        pix = page.get_pixmap(matrix=mat)

        # Use a temp path that is unique per request to avoid race conditions
        img_path = f"/tmp/page_{did}_{page_num}.png"
        pix.save(img_path)
        doc.close()

        return FileResponse(
            img_path,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=300"},
        )

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF preview requires PyMuPDF. Install with: pip install pymupdf",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# SEARCH + STREAM  (hybrid + reranked)
# FIX #5: use deduplicate_sources_by_doc
# ════════════════════════════════════════
@app.post("/chats/{cid}/search-stream")
async def search_stream(cid: str, req: SearchRequest):
    try:
        if not req.doc_ids:
            return {"error": "No documents selected"}

        if cid not in chat_memory:
            chat_memory[cid] = []

        # 1. Hybrid retrieval
        candidates = hybrid_search(req.query, cid, req.doc_ids, top_k=20)

        if not candidates:
            async def empty():
                yield {"data": json.dumps({"text": "I could not find relevant information in the selected documents."})}
            return EventSourceResponse(empty())

        # 2. Rerank with cross-encoder (keep top 5 chunks for context quality)
        top_chunks = rerank(req.query, candidates, top_n=5)

        # 3. Build context from ALL top chunks (good for answer quality)
        context = ""
        raw_sources = []

        for chunk in top_chunks:
            payload = chunk.payload
            context += f"[Page {payload['page']}]\n{payload['text']}\n\n"
            raw_sources.append({
                "filename": payload["filename"],
                "page": payload["page"],
                "preview": payload["text"][:300],
                "doc_id": payload["doc_id"],
                "rerank_score": payload.get("rerank_score", 0),
            })

        # FIX #5: One source chip per document — context still has all chunks
        sources = deduplicate_sources_by_doc(raw_sources)

        history = _build_history(cid)

        async def generate():
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    max_tokens=1024,
                    stream=True,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise document assistant. "
                                "Answer ONLY from the provided context. "
                                "If the answer is not in the context, say so clearly. "
                                "Format your response with markdown: use **bold** for key terms, "
                                "bullet lists where appropriate, and code blocks for any code. "
                                "Do not mention filenames or page numbers in your answer — "
                                "sources are shown separately. Be direct and concise."
                            ),
                        },
                        *history,
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion:\n{req.query}",
                        },
                    ],
                )

                full_answer = ""
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if not delta or not delta.content:
                        continue
                    text = delta.content
                    full_answer += text
                    yield {"data": json.dumps({"text": text})}

                # Persist messages
                c = db()
                c.execute(
                    "INSERT INTO messages (chat_id,role,content,sources,created_at) VALUES (?,?,?,?,?)",
                    (cid, "user", req.query, None, now()),
                )
                c.execute(
                    "INSERT INTO messages (chat_id,role,content,sources,created_at) VALUES (?,?,?,?,?)",
                    (cid, "assistant", full_answer, json.dumps(sources), now()),
                )
                c.commit()
                c.close()

                chat_memory[cid].append({"role": "user", "content": req.query})
                chat_memory[cid].append({"role": "assistant", "content": full_answer})
                _maybe_summarise(cid)

                yield {"data": json.dumps({"sources": sources, "done": True})}

            except Exception as e:
                yield {"data": json.dumps({"error": str(e)})}

        return EventSourceResponse(generate())

    except Exception as e:
        return {"error": str(e)}
    
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    FilterSelector,
)

from sse_starlette.sse import EventSourceResponse

import os
import uuid
import json
import shutil
import sqlite3
import datetime
import re

# ════════════════════════════════════════
# ENV
# ════════════════════════════════════════
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ════════════════════════════════════════
# FASTAPI
# ════════════════════════════════════════
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════
# MODELS
# ════════════════════════════════════════
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Cross-encoder reranker — fast and accurate
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

# ════════════════════════════════════════
# QDRANT
# ════════════════════════════════════════
qdrant = QdrantClient(path="./qdrant_data")

existing = [c.name for c in qdrant.get_collections().collections]

if "rag_docs" not in existing:
    qdrant.create_collection(
        collection_name="rag_docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# ════════════════════════════════════════
# SQLITE
# ════════════════════════════════════════
def db():
    conn = sqlite3.connect("docwise.db", check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    c = db()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_docs (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            display_name TEXT NOT NULL,
            pages INTEGER,
            file_size INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL
        );

        -- Add file_size column if it doesn't exist (migration)
        CREATE TABLE IF NOT EXISTS _migrations (key TEXT PRIMARY KEY);
    """)

    # Safe column migration
    try:
        c.execute("ALTER TABLE chat_docs ADD COLUMN file_size INTEGER DEFAULT 0")
        c.commit()
    except Exception:
        pass  # Column already exists

    c.commit()
    c.close()


init_db()

# ════════════════════════════════════════
# MEMORY  (sliding window + summarisation)
# ════════════════════════════════════════
chat_memory: dict[str, list] = {}
chat_summaries: dict[str, str] = {}

WINDOW = 6          # last N full turns kept verbatim
SUMMARY_TRIGGER = 12  # summarise when history exceeds this many messages


def _build_history(cid: str) -> list:
    """Return the memory list to inject into the LLM call."""
    msgs = chat_memory.get(cid, [])
    summary = chat_summaries.get(cid)

    history = []
    if summary:
        history.append({
            "role": "system",
            "content": f"Summary of earlier conversation:\n{summary}",
        })
    history.extend(msgs[-WINDOW * 2:])   # keep last WINDOW user+assistant pairs
    return history


def _maybe_summarise(cid: str):
    """If memory is getting long, ask the LLM to summarise the older portion."""
    msgs = chat_memory.get(cid, [])
    if len(msgs) < SUMMARY_TRIGGER:
        return

    to_summarise = msgs[:-WINDOW * 2]
    if not to_summarise:
        return

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=256,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. Summarise the following "
                        "conversation in 3-5 concise sentences capturing the key "
                        "topics and decisions. Return only the summary, no preamble."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        f"{m['role'].upper()}: {m['content']}" for m in to_summarise
                    ),
                },
            ],
        )
        chat_summaries[cid] = resp.choices[0].message.content.strip()
        # Keep only the recent window in memory
        chat_memory[cid] = msgs[-WINDOW * 2:]
    except Exception:
        pass  # Non-fatal — just skip summarisation this round

# ════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════
def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_pages(path: str):
    pages = []
    reader = PdfReader(path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    return pages


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokeniser for BM25."""
    return re.findall(r"\w+", text.lower())


# ════════════════════════════════════════
# HYBRID SEARCH  (dense + BM25 fusion)
# ════════════════════════════════════════
def hybrid_search(query: str, cid: str, doc_ids: list[str], top_k: int = 20) -> list:
    """
    1. Dense retrieval via Qdrant  (semantic similarity)
    2. BM25 over the same candidate set  (keyword match)
    3. Reciprocal Rank Fusion to merge scores
    Returns up to top_k merged results (payload dicts with a `score` key).
    """
    query_vec = embed_model.encode(query).tolist()

    # ── Dense retrieval ──
    dense_hits = qdrant.query_points(
        collection_name="rag_docs",
        query=query_vec,
        limit=top_k,
        query_filter=Filter(
            must=[
                FieldCondition(key="chat_id", match=MatchValue(value=cid)),
                FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)),
            ]
        ),
    ).points

    # Filter low-confidence dense hits early
    dense_hits = [h for h in dense_hits if h.score > 0.15]

    if not dense_hits:
        return []

    # ── BM25 over the same candidate pool ──
    corpus = [h.payload["text"] for h in dense_hits]
    tokenised_corpus = [tokenize(t) for t in corpus]
    bm25 = BM25Okapi(tokenised_corpus)
    bm25_scores = bm25.get_scores(tokenize(query))

    # ── Reciprocal Rank Fusion ──
    RRF_K = 60

    dense_rank = {h.id: rank + 1 for rank, h in enumerate(dense_hits)}
    bm25_order = sorted(range(len(dense_hits)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_rank = {dense_hits[i].id: rank + 1 for rank, i in enumerate(bm25_order)}

    rrf_scores = {}
    for h in dense_hits:
        d_rank = dense_rank.get(h.id, top_k + 1)
        b_rank = bm25_rank.get(h.id, top_k + 1)
        rrf_scores[h.id] = (1 / (RRF_K + d_rank)) + (1 / (RRF_K + b_rank))

    merged = sorted(dense_hits, key=lambda h: rrf_scores[h.id], reverse=True)
    return merged


# ════════════════════════════════════════
# RERANKING
# ════════════════════════════════════════
def rerank(query: str, candidates: list, top_n: int = 5) -> list:
    """
    Use cross-encoder to score (query, chunk) pairs and return top_n.
    Filters out anything below a confidence threshold afterwards.
    """
    if not candidates:
        return []

    pairs = [(query, c.payload["text"][:512]) for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    # Cross-encoder scores are logits — keep anything > -2 (roughly top relevant)
    filtered = [(score, c) for score, c in ranked if score > -3]
    top = filtered[:top_n]

    # Attach rerank score to payload for transparency
    result = []
    for score, c in top:
        c.payload["rerank_score"] = float(score)
        result.append(c)
    return result


# ════════════════════════════════════════
# SOURCE DEDUPLICATION
# ════════════════════════════════════════
def deduplicate_sources(sources: list) -> list:
    """
    Keep at most ONE chunk per (filename, page) pair — the one with the
    highest rerank score — so we don't show the same page twice.
    """
    seen: dict[tuple, dict] = {}
    for s in sources:
        key = (s["filename"], s["page"])
        if key not in seen:
            seen[key] = s
    return list(seen.values())


# ════════════════════════════════════════
# REQUEST MODELS
# ════════════════════════════════════════
class SearchRequest(BaseModel):
    query: str
    chat_id: str
    doc_ids: List[str]


class RenameBody(BaseModel):
    name: str

# ════════════════════════════════════════
# CHAT ROUTES
# ════════════════════════════════════════
@app.get("/chats")
def get_chats():
    c = db()
    rows = c.execute("SELECT * FROM chats ORDER BY created_at DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]


@app.post("/chats")
def create_chat():
    cid = str(uuid.uuid4())
    c = db()
    c.execute("INSERT INTO chats VALUES (?,?,?)", (cid, "New Chat", now()))
    c.commit()
    c.close()
    return {"id": cid, "name": "New Chat", "created_at": now()}


@app.put("/chats/{cid}")
def rename_chat(cid: str, body: RenameBody):
    c = db()
    c.execute("UPDATE chats SET name=? WHERE id=?", (body.name, cid))
    c.commit()
    c.close()
    return {"status": "ok"}


@app.delete("/chats/{cid}")
def delete_chat(cid: str):
    c = db()
    docs = c.execute("SELECT * FROM chat_docs WHERE chat_id=?", (cid,)).fetchall()

    for d in docs:
        try:
            qdrant.delete(
                collection_name="rag_docs",
                points_selector=FilterSelector(
                    filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=d["id"]))])
                ),
            )
        except Exception:
            pass
        try:
            os.remove(f"uploads/{d['id']}_{d['filename']}")
        except Exception:
            pass

    c.execute("DELETE FROM messages WHERE chat_id=?", (cid,))
    c.execute("DELETE FROM chat_docs WHERE chat_id=?", (cid,))
    c.execute("DELETE FROM chats WHERE id=?", (cid,))
    c.commit()
    c.close()

    chat_memory.pop(cid, None)
    chat_summaries.pop(cid, None)
    return {"status": "deleted"}

# ════════════════════════════════════════
# MESSAGE ROUTES
# ════════════════════════════════════════
@app.get("/chats/{cid}/messages")
def get_messages(cid: str):
    c = db()
    rows = c.execute(
        "SELECT * FROM messages WHERE chat_id=? ORDER BY created_at ASC", (cid,)
    ).fetchall()
    c.close()
    result = []
    for r in rows:
        row = dict(r)
        row["sources"] = json.loads(row["sources"]) if row["sources"] else []
        result.append(row)
    return result

# ════════════════════════════════════════
# DOCUMENT ROUTES
# ════════════════════════════════════════
@app.get("/chats/{cid}/documents")
def get_documents(cid: str):
    c = db()
    rows = c.execute(
        "SELECT * FROM chat_docs WHERE chat_id=? ORDER BY created_at ASC", (cid,)
    ).fetchall()
    c.close()
    return [dict(r) for r in rows]


@app.post("/chats/{cid}/upload")
async def upload_pdf(cid: str, file: UploadFile = File(...)):
    try:
        c = db()
        exists = c.execute("SELECT id FROM chats WHERE id=?", (cid,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Chat not found")

        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        doc_id = str(uuid.uuid4())
        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{doc_id}_{filename}"

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_size = os.path.getsize(path)  # bytes

        pages = extract_pages(path)
        if not pages:
            os.remove(path)
            return {"error": "No readable text found in PDF"}

        points = []
        for item in pages:
            for chunk in chunk_text(item["text"]):
                vector = embed_model.encode(chunk[:4000]).tolist()
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "chat_id": cid,
                            "doc_id": doc_id,
                            "text": chunk,
                            "page": item["page"],
                            "filename": filename,
                        },
                    )
                )

        for i in range(0, len(points), 100):
            qdrant.upsert(collection_name="rag_docs", points=points[i : i + 100])

        display_name = os.path.splitext(filename)[0]
        display_name = display_name.replace("_", " ").replace("-", " ")
        display_name = " ".join(w.capitalize() for w in display_name.split()[:5])

        c.execute(
            "INSERT INTO chat_docs VALUES (?,?,?,?,?,?,?)",
            (doc_id, cid, filename, display_name, len(pages), file_size, now()),
        )

        chat = c.execute("SELECT name FROM chats WHERE id=?", (cid,)).fetchone()
        if chat and chat["name"] == "New Chat":
            c.execute("UPDATE chats SET name=? WHERE id=?", (display_name, cid))

        c.commit()
        c.close()

        return {
            "doc_id": doc_id,
            "filename": filename,
            "display_name": display_name,
            "pages": len(pages),
            "file_size": file_size,
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


@app.put("/chats/{cid}/documents/{did}")
def rename_document(cid: str, did: str, body: RenameBody):
    c = db()
    c.execute(
        "UPDATE chat_docs SET display_name=? WHERE id=? AND chat_id=?",
        (body.name, did, cid),
    )
    c.commit()
    c.close()
    return {"status": "ok"}


@app.delete("/chats/{cid}/documents/{did}")
def delete_document(cid: str, did: str):
    c = db()
    row = c.execute(
        "SELECT filename FROM chat_docs WHERE id=? AND chat_id=?", (did, cid)
    ).fetchone()

    try:
        qdrant.delete(
            collection_name="rag_docs",
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=did))])
            ),
        )
    except Exception:
        pass

    if row:
        try:
            os.remove(f"uploads/{did}_{row['filename']}")
        except Exception:
            pass

    c.execute("DELETE FROM chat_docs WHERE id=? AND chat_id=?", (did, cid))
    c.commit()
    c.close()
    return {"status": "deleted"}


# ════════════════════════════════════════
# PDF PAGE PREVIEW
# ════════════════════════════════════════
@app.get("/chats/{cid}/documents/{did}/page/{page_num}")
async def get_pdf_page(cid: str, did: str, page_num: int):
    """
    Returns a specific page of the PDF rendered as a PNG image.
    Requires: pip install pymupdf
    """
    try:
        import fitz  # PyMuPDF

        c = db()
        row = c.execute(
            "SELECT filename FROM chat_docs WHERE id=? AND chat_id=?", (did, cid)
        ).fetchone()
        c.close()

        if not row:
            raise HTTPException(status_code=404, detail="Document not found")

        path = f"uploads/{did}_{row['filename']}"
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="File not found on disk")

        doc = fitz.open(path)

        if page_num < 1 or page_num > len(doc):
            raise HTTPException(status_code=400, detail="Page number out of range")

        page = doc[page_num - 1]
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for sharp rendering
        pix = page.get_pixmap(matrix=mat)

        img_path = f"/tmp/page_{did}_{page_num}.png"
        pix.save(img_path)
        doc.close()

        return FileResponse(img_path, media_type="image/png")

    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PDF preview requires PyMuPDF. Install with: pip install pymupdf",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════
# SEARCH + STREAM  (hybrid + reranked)
# ════════════════════════════════════════
@app.post("/chats/{cid}/search-stream")
async def search_stream(cid: str, req: SearchRequest):
    try:
        if not req.doc_ids:
            return {"error": "No documents selected"}

        if cid not in chat_memory:
            chat_memory[cid] = []

        # ── 1. Hybrid retrieval (dense + BM25 fusion) ──
        candidates = hybrid_search(req.query, cid, req.doc_ids, top_k=20)

        if not candidates:
            async def empty():
                yield {"data": json.dumps({"text": "I could not find relevant information in the selected documents."})}
            return EventSourceResponse(empty())

        # ── 2. Rerank with cross-encoder ──
        top_chunks = rerank(req.query, candidates, top_n=5)

        # ── 3. Build context & deduplicated sources ──
        context = ""
        raw_sources = []

        for chunk in top_chunks:
            payload = chunk.payload
            context += f"[Page {payload['page']}]\n{payload['text']}\n\n"
            raw_sources.append({
                "filename": payload["filename"],
                "page": payload["page"],
                "preview": payload["text"][:300],
                "doc_id": payload["doc_id"],
            })

        sources = deduplicate_sources(raw_sources)
        # Cap at 3 sources maximum — only the most relevant
        sources = sources[:3]

        history = _build_history(cid)

        async def generate():
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    max_tokens=1024,
                    stream=True,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a precise document assistant. "
                                "Answer ONLY from the provided context. "
                                "If the answer is not in the context, say so clearly. "
                                "Format your response with markdown: use **bold** for key terms, "
                                "bullet lists where appropriate, and code blocks for any code. "
                                "Do not mention filenames or page numbers in your answer — "
                                "sources are shown separately. Be direct and concise."
                            ),
                        },
                        *history,
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion:\n{req.query}",
                        },
                    ],
                )

                full_answer = ""
                for chunk in response:
                    delta = chunk.choices[0].delta
                    if not delta or not delta.content:
                        continue
                    text = delta.content
                    full_answer += text
                    yield {"data": json.dumps({"text": text})}

                # Persist messages
                c = db()
                c.execute(
                    "INSERT INTO messages (chat_id,role,content,sources,created_at) VALUES (?,?,?,?,?)",
                    (cid, "user", req.query, None, now()),
                )
                c.execute(
                    "INSERT INTO messages (chat_id,role,content,sources,created_at) VALUES (?,?,?,?,?)",
                    (cid, "assistant", full_answer, json.dumps(sources), now()),
                )
                c.commit()
                c.close()

                # Update sliding memory
                chat_memory[cid].append({"role": "user", "content": req.query})
                chat_memory[cid].append({"role": "assistant", "content": full_answer})

                # Maybe summarise old turns
                _maybe_summarise(cid)

                yield {"data": json.dumps({"sources": sources, "done": True})}

            except Exception as e:
                yield {"data": json.dumps({"error": str(e)})}

        return EventSourceResponse(generate())

    except Exception as e:
        return {"error": str(e)}

# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from pypdf import PdfReader
from groq import Groq

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    FilterSelector,
)

from sse_starlette.sse import EventSourceResponse

import os
import uuid
import json
import shutil
import sqlite3
import datetime

# ════════════════════════════════════════
# ENV
# ════════════════════════════════════════
load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ════════════════════════════════════════
# FASTAPI
# ════════════════════════════════════════
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════
# EMBEDDING MODEL
# ════════════════════════════════════════
model = SentenceTransformer("all-MiniLM-L6-v2")

# ════════════════════════════════════════
# QDRANT
# ════════════════════════════════════════
client = QdrantClient(path="./qdrant_data")

collections = [
    c.name for c in client.get_collections().collections
]

if "rag_docs" not in collections:
    client.create_collection(
        collection_name="rag_docs",
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

# ════════════════════════════════════════
# SQLITE
# ════════════════════════════════════════
def db():
    conn = sqlite3.connect(
        "docwise.db",
        check_same_thread=False,
        timeout=30,
    )
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    c = db()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_docs (
            id TEXT PRIMARY KEY,
            chat_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            display_name TEXT NOT NULL,
            pages INTEGER,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL
        );
    """)

    c.commit()
    c.close()


init_db()

# ════════════════════════════════════════
# MEMORY
# ════════════════════════════════════════
chat_memory = {}

# ════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════
def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_pages(path):
    pages = []

    reader = PdfReader(path)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text and text.strip():
            pages.append({
                "page": i + 1,
                "text": text.strip(),
            })

    return pages


def chunk_text(text, chunk_size=700, overlap=120):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunks.append(
            text[start:end]
        )

        start += chunk_size - overlap

    return chunks

# ════════════════════════════════════════
# REQUEST MODELS
# ════════════════════════════════════════
class SearchRequest(BaseModel):
    query: str
    chat_id: str
    doc_ids: List[str]


class RenameBody(BaseModel):
    name: str

# ════════════════════════════════════════
# CHAT ROUTES
# ════════════════════════════════════════
@app.get("/chats")
def get_chats():
    c = db()

    rows = c.execute("""
        SELECT *
        FROM chats
        ORDER BY created_at DESC
    """).fetchall()

    c.close()

    return [dict(r) for r in rows]


@app.post("/chats")
def create_chat():
    cid = str(uuid.uuid4())

    c = db()

    c.execute(
        "INSERT INTO chats VALUES (?,?,?)",
        (
            cid,
            "New Chat",
            now(),
        ),
    )

    c.commit()
    c.close()

    return {
        "id": cid,
        "name": "New Chat",
        "created_at": now(),
    }


@app.put("/chats/{cid}")
def rename_chat(cid: str, body: RenameBody):
    c = db()

    c.execute(
        "UPDATE chats SET name=? WHERE id=?",
        (
            body.name,
            cid,
        ),
    )

    c.commit()
    c.close()

    return {"status": "ok"}


@app.delete("/chats/{cid}")
def delete_chat(cid: str):
    c = db()

    docs = c.execute("""
        SELECT *
        FROM chat_docs
        WHERE chat_id=?
    """, (cid,)).fetchall()

    for d in docs:
        try:
            client.delete(
                collection_name="rag_docs",
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="doc_id",
                                match=MatchValue(
                                    value=d["id"]
                                )
                            )
                        ]
                    )
                )
            )
        except:
            pass

        try:
            os.remove(
                f"uploads/{d['id']}_{d['filename']}"
            )
        except:
            pass

    c.execute(
        "DELETE FROM messages WHERE chat_id=?",
        (cid,),
    )

    c.execute(
        "DELETE FROM chat_docs WHERE chat_id=?",
        (cid,),
    )

    c.execute(
        "DELETE FROM chats WHERE id=?",
        (cid,),
    )

    c.commit()
    c.close()

    if cid in chat_memory:
        del chat_memory[cid]

    return {"status": "deleted"}

# ════════════════════════════════════════
# MESSAGE ROUTES
# ════════════════════════════════════════
@app.get("/chats/{cid}/messages")
def get_messages(cid: str):
    c = db()

    rows = c.execute("""
        SELECT *
        FROM messages
        WHERE chat_id=?
        ORDER BY created_at ASC
    """, (cid,)).fetchall()

    c.close()

    result = []

    for r in rows:
        row = dict(r)

        row["sources"] = (
            json.loads(row["sources"])
            if row["sources"]
            else []
        )

        result.append(row)

    return result

# ════════════════════════════════════════
# DOCUMENT ROUTES
# ════════════════════════════════════════
@app.get("/chats/{cid}/documents")
def get_documents(cid: str):
    c = db()

    rows = c.execute("""
        SELECT *
        FROM chat_docs
        WHERE chat_id=?
        ORDER BY created_at ASC
    """, (cid,)).fetchall()

    c.close()

    return [dict(r) for r in rows]


@app.post("/chats/{cid}/upload")
async def upload_pdf(
    cid: str,
    file: UploadFile = File(...),
):
    try:
        c = db()

        exists = c.execute("""
            SELECT id
            FROM chats
            WHERE id=?
        """, (cid,)).fetchone()

        if not exists:
            raise HTTPException(
                status_code=404,
                detail="Chat not found",
            )

        filename = file.filename or ""

        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed",
            )

        doc_id = str(uuid.uuid4())

        os.makedirs("uploads", exist_ok=True)

        path = f"uploads/{doc_id}_{filename}"

        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        pages = extract_pages(path)

        if not pages:
            os.remove(path)

            return {
                "error": "No readable text found in PDF"
            }

        points = []

        for item in pages:
            chunks = chunk_text(item["text"])

            for chunk in chunks:
                vector = model.encode(
                    chunk[:4000]
                ).tolist()

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "chat_id": cid,
                            "doc_id": doc_id,
                            "text": chunk,
                            "page": item["page"],
                            "filename": filename,
                        },
                    )
                )

        for i in range(0, len(points), 100):
            client.upsert(
                collection_name="rag_docs",
                points=points[i:i + 100],
            )

        display_name = os.path.splitext(
            filename
        )[0]

        display_name = (
            display_name
            .replace("_", " ")
            .replace("-", " ")
        )

        display_name = " ".join(
            w.capitalize()
            for w in display_name.split()[:5]
        )

        c.execute("""
            INSERT INTO chat_docs
            VALUES (?,?,?,?,?,?)
        """, (
            doc_id,
            cid,
            filename,
            display_name,
            len(pages),
            now(),
        ))

        chat = c.execute("""
            SELECT name
            FROM chats
            WHERE id=?
        """, (cid,)).fetchone()

        if chat and chat["name"] == "New Chat":
            c.execute("""
                UPDATE chats
                SET name=?
                WHERE id=?
            """, (
                display_name,
                cid,
            ))

        c.commit()
        c.close()

        return {
            "doc_id": doc_id,
            "filename": filename,
            "display_name": display_name,
            "pages": len(pages),
        }

    except HTTPException:
        raise

    except Exception as e:
        return {
            "error": str(e)
        }


@app.put("/chats/{cid}/documents/{did}")
def rename_document(
    cid: str,
    did: str,
    body: RenameBody,
):
    c = db()

    c.execute("""
        UPDATE chat_docs
        SET display_name=?
        WHERE id=? AND chat_id=?
    """, (
        body.name,
        did,
        cid,
    ))

    c.commit()
    c.close()

    return {"status": "ok"}


@app.delete("/chats/{cid}/documents/{did}")
def delete_document(
    cid: str,
    did: str,
):
    c = db()

    row = c.execute("""
        SELECT filename
        FROM chat_docs
        WHERE id=? AND chat_id=?
    """, (
        did,
        cid,
    )).fetchone()

    try:
        client.delete(
            collection_name="rag_docs",
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(
                                value=did
                            )
                        )
                    ]
                )
            )
        )
    except:
        pass

    if row:
        try:
            os.remove(
                f"uploads/{did}_{row['filename']}"
            )
        except:
            pass

    c.execute("""
        DELETE FROM chat_docs
        WHERE id=? AND chat_id=?
    """, (
        did,
        cid,
    ))

    c.commit()
    c.close()

    return {"status": "deleted"}

# ════════════════════════════════════════
# SEARCH + STREAM
# ════════════════════════════════════════
@app.post("/chats/{cid}/search-stream")
async def search_stream(
    cid: str,
    req: SearchRequest,
):
    try:
        if not req.doc_ids:
            return {
                "error": "No documents selected"
            }

        if cid not in chat_memory:
            chat_memory[cid] = []

        query_vector = model.encode(
            req.query
        ).tolist()

        results = client.query_points(
            collection_name="rag_docs",
            query=query_vector,
            limit=8,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="chat_id",
                        match=MatchValue(value=cid),
                    ),
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(any=req.doc_ids),
                    ),
                ]
            ),
        ).points

        # ✅ More permissive — catches relevant results
        results = [
            r for r in results
            if r.score > 0.2
        ]

        if not results:
            async def empty():
                yield {
                    "data": json.dumps({
                        "text": (
                            "I could not find relevant "
                            "information in the selected documents."
                        )
                    })
                }

            return EventSourceResponse(empty())

        context = ""
        sources = []

        for r in results:
            payload = r.payload

            context += (
                f"[Page {payload['page']}]\n"
                f"{payload['text']}\n\n"
            )

            sources.append({
                "filename": payload["filename"],
                "page": payload["page"],
                "preview": payload["text"][:200],
            })

        memory = chat_memory[cid][-4:]

        async def generate():
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    max_tokens=1024,
                    stream=True,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Answer ONLY from the provided context. "
                                "If the answer is not in the context, say so clearly. "
                                "Do not mention filenames or page numbers. "
                                "Be direct and concise."
                            ),
                        },
                        *memory,
                        {
                            "role": "user",
                            "content": (
                                f"Context:\n{context}\n\n"
                                f"Question:\n{req.query}"
                            ),
                        },
                    ],
                )

                full_answer = ""

                for chunk in response:
                    delta = chunk.choices[0].delta

                    if not delta:
                        continue

                    text = delta.content

                    if not text:
                        continue

                    full_answer += text

                    yield {
                        "data": json.dumps({
                            "text": text
                        })
                    }

                c = db()

                c.execute("""
                    INSERT INTO messages
                    (
                        chat_id,
                        role,
                        content,
                        sources,
                        created_at
                    )
                    VALUES (?,?,?,?,?)
                """, (
                    cid,
                    "user",
                    req.query,
                    None,
                    now(),
                ))

                c.execute("""
                    INSERT INTO messages
                    (
                        chat_id,
                        role,
                        content,
                        sources,
                        created_at
                    )
                    VALUES (?,?,?,?,?)
                """, (
                    cid,
                    "assistant",
                    full_answer,
                    json.dumps(sources),
                    now(),
                ))

                c.commit()
                c.close()

                chat_memory[cid].append({
                    "role": "user",
                    "content": req.query,
                })

                chat_memory[cid].append({
                    "role": "assistant",
                    "content": full_answer,
                })

                yield {
                    "data": json.dumps({
                        "sources": sources,
                        "done": True,
                    })
                }

            except Exception as e:
                yield {
                    "data": json.dumps({
                        "error": str(e)
                    })
                }

        return EventSourceResponse(generate())

    except Exception as e:
        return {
            "error": str(e)
        }

# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil, os, uuid
from pypdf import PdfReader
from typing import List

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchAny
)

from groq import Groq
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import json

# ---------------- ENV ----------------
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------------- APP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODEL ----------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- QDRANT ----------------
client = QdrantClient(path="./qdrant_data")

existing = [c.name for c in client.get_collections().collections]
if "rag_docs" not in existing:
    client.create_collection(
        collection_name="rag_docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# ---------------- STORAGE ----------------
chat_memory = {}
documents = {}

# ---------------- REQUEST ----------------
class SearchRequest(BaseModel):
    query: str
    doc_ids: List[str]
    session_id: str = "user-1"

class RenameRequest(BaseModel):
    doc_id: str
    new_name: str

# ---------------- PDF ----------------
def extract_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    return pages

# ---------------- UPLOAD ----------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    doc_id = str(uuid.uuid4())

    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    pages = extract_pages(path)

    points = []

    for item in pages:
        vector = model.encode(item["text"]).tolist()

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "text": item["text"],
                    "page": item["page"],
                    "filename": file.filename,
                }
            )
        )

    for i in range(0, len(points), 100):
        client.upsert(
            collection_name="rag_docs",
            points=points[i:i + 100]
        )

    documents[doc_id] = file.filename

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": len(pages)
    }

# ---------------- GET DOCS ----------------
@app.get("/documents")
def get_documents():
    return [
        {"doc_id": k, "filename": v}
        for k, v in documents.items()
    ]

# ---------------- DELETE DOC ----------------
@app.delete("/document/{doc_id}")
def delete_document(doc_id: str):
    try:
        if doc_id in documents:
            del documents[doc_id]

        client.delete(
            collection_name="rag_docs",
            points_selector={
                "filter": {
                    "must": [
                        {
                            "key": "doc_id",
                            "match": {"value": doc_id}
                        }
                    ]
                }
            }
        )

        return {"message": "Deleted successfully"}

    except Exception as e:
        return {"error": str(e)}

# ---------------- RENAME DOC ----------------
class RenameRequest(BaseModel):
    doc_id: str
    new_name: str

@app.put("/document/rename")
def rename_document(req: RenameRequest):
    if req.doc_id not in documents:
        return {"error": "Not found"}

    documents[req.doc_id] = req.new_name

    return {
        "doc_id": req.doc_id,
        "new_name": req.new_name
    }

# ---------------- SEARCH STREAM ----------------
@app.post("/search-stream")
async def search_stream(req: SearchRequest):

    if not req.doc_ids:
        return {"error": "No documents selected"}

    session_id = req.session_id

    if session_id not in chat_memory:
        chat_memory[session_id] = []

    query_vector = model.encode(req.query).tolist()

    results = client.query_points(
        collection_name="rag_docs",
        query=query_vector,
        limit=8,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchAny(any=req.doc_ids)
                )
            ]
        )
    ).points

    context = ""
    sources = []

    for r in results:
        fname = r.payload.get("filename", "Unknown")

        context += f"[{fname} | Page {r.payload['page']}]\n{r.payload['text']}\n\n"

        sources.append({
            "filename": fname,
            "page": r.payload["page"],
            "preview": r.payload["text"][:150]
        })

    memory = chat_memory[session_id][-6:]

    async def event_generator():

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Answer only from context."},
                *memory,
                {"role": "user", "content": context + "\n\nQ: " + req.query}
            ],
            stream=True
        )

        full = ""

        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                full += token
                yield {"data": json.dumps({"text": token})}

        chat_memory[session_id].append({"role": "user", "content": req.query})
        chat_memory[session_id].append({"role": "assistant", "content": full})

        yield {"data": json.dumps({"sources": sources})}

    return EventSourceResponse(event_generator())

# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================
# ==========================================

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil, os, uuid
from pypdf import PdfReader
from typing import List

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchAny
)

from groq import Groq
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse
import json

# ── ENV ──
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── APP ──
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── EMBEDDING MODEL ──
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── QDRANT ──
client = QdrantClient(path="./qdrant_data")

existing = [c.name for c in client.get_collections().collections]
if "rag_docs" not in existing:
    client.create_collection(
        collection_name="rag_docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# ── In-memory stores ──
chat_memory = {}
documents   = {}  # { doc_id: filename }

# ── REQUEST MODEL ──
# ✅ doc_ids is now a LIST — supports multiple PDFs
class SearchRequest(BaseModel):
    query:      str
    doc_ids:    List[str]       # ✅ list of selected doc IDs
    session_id: str = "user-1"

# ── Extract text per page ──
def extract_pages(pdf_path):
    reader = PdfReader(pdf_path)
    pages  = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    return pages

# ─────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        doc_id = str(uuid.uuid4())

        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{file.filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        pages = extract_pages(path)
        if not pages:
            return {"error": "Could not extract text from this PDF"}

        points = []
        for item in pages:
            vector = model.encode(item["text"]).tolist()
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "doc_id":   doc_id,
                        "text":     item["text"],
                        "page":     item["page"],
                        "filename": file.filename,
                    }
                )
            )

        # Upsert in batches
        for i in range(0, len(points), 100):
            client.upsert(
                collection_name="rag_docs",
                points=points[i:i + 100]
            )

        documents[doc_id] = file.filename

        return {
            "doc_id":   doc_id,
            "filename": file.filename,
            "pages":    len(pages),
        }

    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────
# GET documents
# ─────────────────────────────────────────
@app.get("/documents")
def get_documents():
    return [
        {"doc_id": k, "filename": v}
        for k, v in documents.items()
    ]

# ─────────────────────────────────────────
# SEARCH ACROSS MULTIPLE PDFs + STREAM
# ─────────────────────────────────────────
@app.post("/search-stream")
async def search_stream(req: SearchRequest):
    try:
        if not req.doc_ids:
            return {"error": "No documents selected"}

        session_id = req.session_id
        if session_id not in chat_memory:
            chat_memory[session_id] = []

        query_vector = model.encode(req.query).tolist()

        # ✅ MatchAny filters across ALL selected doc_ids at once
        search_results = client.query_points(
            collection_name="rag_docs",
            query=query_vector,
            limit=5,  # get more results since we have multiple PDFs
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchAny(any=req.doc_ids)  # ✅ matches any of the selected docs
                    )
                ]
            )
        )
        results = search_results.points

        # Build context — include filename so LLM knows which doc each chunk is from
        context = ""
        sources = []
        for r in results:
            fname = r.payload.get("filename", "Unknown")
            context += f"[From: {fname}, Page {r.payload['page']}]\n{r.payload['text']}\n\n"
            sources.append({
                "filename": fname,
                "page":     r.payload["page"],
                "preview":  r.payload["text"][:200],
            })

        memory = chat_memory[session_id][-6:]

        # Build a note about which docs are selected
        selected_names = [documents.get(d, d) for d in req.doc_ids]
        docs_note = ", ".join(selected_names)

        async def event_generator():
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                f"You are a helpful assistant. The user has selected "
    f"these documents: {docs_note}. "
    "Answer questions using only the provided context. "
    "Give direct, clean answers only — do NOT mention filenames, "
    "page numbers, or source references in your answer. "
    "Never say 'according to the document' or 'on page X' or "
    "'as stated in filename'. Just answer directly as if you "
    "already know the information. "
    "The sources are shown separately to the user. "
    "If the answer is not in the context, say so clearly."
                            )
                        },
                        *memory,
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion:\n{req.query}"
                        }
                    ],
                    stream=True
                )

                full = ""
                for chunk in response:
                    token = chunk.choices[0].delta.content
                    if token:
                        full += token
                        yield {"data": json.dumps({"text": token})}

                # Save to memory
                chat_memory[session_id].append({"role": "user",      "content": req.query})
                chat_memory[session_id].append({"role": "assistant", "content": full})

                # Send sources at end
                yield {"data": json.dumps({"sources": sources, "done": True})}

            except Exception as e:
                yield {"data": json.dumps({"error": str(e)})}

        return EventSourceResponse(event_generator())

    except Exception as e:
        return {"error": str(e)}