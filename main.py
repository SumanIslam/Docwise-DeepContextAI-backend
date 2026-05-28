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

# ── SQLite ────────────────────────────────────────────────────────────────────
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

# ── FIX #7: OCR fallback for image-based PDFs ─────────────────────────────────
def _ocr_page_with_pymupdf(path: str, page_num: int) -> str:
    """
    Render the page to a pixmap and run pytesseract OCR on it.
    Returns empty string if pymupdf or pytesseract are not installed.
    """
    try:
        import fitz          # PyMuPDF
        import pytesseract   # pip install pytesseract  (also needs Tesseract binary)
        from PIL import Image
        import io

        doc  = fitz.open(path)
        page = doc[page_num]
        mat  = fitz.Matrix(2.0, 2.0)     # 2× zoom for better OCR accuracy
        pix  = page.get_pixmap(matrix=mat)
        img  = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img)
        doc.close()
        return text.strip()
    except Exception:
        return ""


def extract_pages(path: str) -> list:
    """
    Extract text from every PDF page.
    For pages where pdfminer/pypdf yields < 30 chars of text (likely image-only),
    attempt OCR via PyMuPDF + Tesseract.
    """
    pages = []
    reader = PdfReader(path)
    total  = len(reader.pages)

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()

        # FIX #7: if text is very short this page is probably an image → try OCR
        if len(text) < 30:
            ocr_text = _ocr_page_with_pymupdf(path, i)
            if ocr_text:
                text = ocr_text

        if text:
            pages.append({"page": i + 1, "text": text})

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

# ── FIX #3: summary/overview queries — lower threshold, always provide context ─
# The system prompt already ensures the model uses ONLY the supplied context.
# We keep sources hidden on the frontend when rerank_score is low (overview queries
# naturally scatter low scores across many pages rather than hitting one page hard).
# The threshold is set deliberately low so that broad queries still get sources
# suppressed while still receiving an answer from the full context window.

RERANK_THRESHOLD = -2.5

def aggregate_sources(chunks):
    """
    Group by doc_id. Collect all pages. Filter if best score < threshold.
    For overview/summary queries the scores are low → sources list is empty
    → frontend shows no source chip. The answer is still generated normally.
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

    valid = [d for d in by_doc.values() if d["best_score"] > RERANK_THRESHOLD and d["pages"]]
    if not valid: return []

    best = sorted(valid, key=lambda d: d["best_score"], reverse=True)[:1]
    return [{"doc_id":d["doc_id"],"filename":d["filename"],"pages":sorted(d["pages"])} for d in best]


# ── FIX #3: detect overview/summary intent so we broaden the context ──────────
OVERVIEW_PATTERNS = re.compile(
    r'\b(overview|summarize|summarise|summary|what is this (doc|document|pdf|file|about)|'
    r'what does this (doc|document|pdf|file) (say|cover|contain|discuss)|'
    r'tell me about this|describe this|give me a summary|brief summary|'
    r'what topics|main topics|key points|key takeaways|abstract)\b',
    re.IGNORECASE
)

def is_overview_query(query: str) -> bool:
    return bool(OVERVIEW_PATTERNS.search(query))


class SearchRequest(BaseModel):
    query: str; chat_id: str; doc_ids: List[str]

class RenameBody(BaseModel):
    name: str

# ── CHAT ROUTES ───────────────────────────────────────────────────────────────
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

# ── DOCUMENT ROUTES ───────────────────────────────────────────────────────────
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
        pages = extract_pages(path)   # FIX #7: now includes OCR fallback
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

# ── SEARCH + STREAM ───────────────────────────────────────────────────────────
@app.post("/chats/{cid}/search-stream")
async def search_stream(cid, req: SearchRequest):
    try:
        if not req.doc_ids: return {"error":"No documents selected"}
        if cid not in chat_memory: chat_memory[cid]=[]

        overview = is_overview_query(req.query)

        # FIX #3: for overview queries fetch more chunks so the whole doc is covered
        top_k_retrieval = 40 if overview else 20
        top_n_rerank    = 10 if overview else 5

        candidates = hybrid_search(req.query, cid, req.doc_ids, top_k=top_k_retrieval)
        if not candidates:
            async def empty():
                yield {"data":json.dumps({"text":"I could not find relevant information in the selected documents."})}
            return EventSourceResponse(empty())

        top_chunks = rerank(req.query, candidates, top_n=top_n_rerank)
        context    = "".join(f"[Page {c.payload['page']}]\n{c.payload['text']}\n\n" for c in top_chunks)

        # FIX #3: overview queries → no sources shown in UI
        sources = [] if overview else aggregate_sources(top_chunks)

        history = _build_history(cid)

        # FIX #3: tailor system prompt for overview vs specific queries
        if overview:
            system_prompt = (
                "You are a precise document assistant. "
                "The user wants a summary or overview. "
                "Using ONLY the provided context, write a clear, well-structured overview. "
                "Cover the main topics, key points, and important details found in the context. "
                "Use markdown: headings, bullet lists, and **bold** for key terms. "
                "Do NOT mention filenames, page numbers, or say the information is missing. "
                "Be thorough but concise."
            )
        else:
            system_prompt = (
                "You are a precise document assistant. "
                "Answer ONLY from the provided context. "
                "If the answer is not in the context, say so clearly. "
                "Format with markdown: **bold** for key terms, bullet lists, code blocks for code. "
                "Do NOT mention filenames or page numbers. Be direct and concise."
            )

        async def generate():
            try:
                resp = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=1024, stream=True,
                    messages=[
                        {"role":"system","content":system_prompt},
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