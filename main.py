from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
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
    allow_origins=[
        "http://localhost:5173","http://127.0.0.1:5173",
        "http://localhost:3000","http://127.0.0.1:3000",
    ],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker    = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

qdrant = QdrantClient(path="./qdrant_data")
if "rag_docs" not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        "rag_docs",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

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
    try:
        c.execute("ALTER TABLE chat_docs ADD COLUMN file_size INTEGER DEFAULT 0")
        c.commit()
    except:
        pass
    c.commit()
    c.close()

init_db()

chat_memory:    dict[str, list] = {}
chat_summaries: dict[str, str]  = {}
WINDOW          = 6
SUMMARY_TRIGGER = 12

def _build_history(cid: str) -> list:
    msgs    = chat_memory.get(cid, [])
    summary = chat_summaries.get(cid)
    history = []
    if summary:
        history.append({
            "role": "system",
            "content": f"Summary of earlier conversation:\n{summary}",
        })
    history.extend(msgs[-(WINDOW * 2):])
    return history

def _maybe_summarise(cid: str):
    msgs = chat_memory.get(cid, [])
    if len(msgs) < SUMMARY_TRIGGER:
        return
    to_sum = msgs[:-(WINDOW * 2)]
    if not to_sum:
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
                        "Summarise the conversation in 3-5 concise sentences. "
                        "Return only the summary."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n".join(
                        f"{m['role'].upper()}: {m['content']}" for m in to_sum
                    ),
                },
            ],
        )
        chat_summaries[cid] = resp.choices[0].message.content.strip()
        chat_memory[cid]    = msgs[-(WINDOW * 2):]
    except:
        pass

def now() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# ── FIX #3 & #5: robust page extraction with per-block image OCR ──────────────

def _has_pymupdf() -> bool:
    try:
        import fitz  # noqa
        return True
    except ImportError:
        return False

def _has_tesseract() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

# Cache capability flags at startup so we don't check on every page
_PYMUPDF_OK    = _has_pymupdf()
_TESSERACT_OK  = _has_tesseract()

def _ocr_page(path: str, page_idx: int) -> str:
    """
    Render the page at 2× zoom and run Tesseract OCR.
    Returns extracted text or '' if dependencies are missing.
    """
    if not (_PYMUPDF_OK and _TESSERACT_OK):
        return ""
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io

        doc  = fitz.open(path)
        page = doc[page_idx]
        mat  = fitz.Matrix(2.0, 2.0)
        pix  = page.get_pixmap(matrix=mat)
        img  = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, config="--psm 3")
        doc.close()
        return text.strip()
    except Exception:
        return ""

def _ocr_image_blocks(path: str, page_idx: int) -> str:
    """
    FIX #3: For pages that have BOTH text and images, extract only the image
    blocks on that page via PyMuPDF, rasterise each image block, and OCR it.
    This handles mixed pages (text + diagrams/screenshots/charts with captions).
    Returns concatenated OCR text from all image regions, or '' if none.
    """
    if not (_PYMUPDF_OK and _TESSERACT_OK):
        return ""
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io

        doc  = fitz.open(path)
        page = doc[page_idx]
        ocr_parts = []

        # get_images returns (xref, smask, w, h, bpc, cs, alt, name, filter, referencer)
        for img_info in page.get_images(full=True):
            xref  = img_info[0]
            image = doc.extract_image(xref)
            img_bytes = image["image"]
            pil_img   = Image.open(io.BytesIO(img_bytes))

            # Skip tiny icons / decorative elements (< 100px in either dim)
            w, h = pil_img.size
            if w < 100 or h < 100:
                continue

            ocr_text = pytesseract.image_to_string(pil_img, config="--psm 3")
            if ocr_text.strip():
                ocr_parts.append(ocr_text.strip())

        doc.close()
        return "\n\n".join(ocr_parts)
    except Exception:
        return ""

def extract_pages(path: str) -> list:
    """
    Extract text from every PDF page.

    Strategy per page:
    1. Use pypdf to extract native text.
    2. If text is very short  (< 50 chars) → full-page OCR via PyMuPDF+Tesseract.
    3. If text is present but the page also contains images → append per-image OCR
       so that diagrams, screenshots, or code-in-images are also captured (FIX #3).
    """
    pages  = []
    reader = PdfReader(path)

    for i, page in enumerate(reader.pages):
        raw  = page.extract_text() or ""
        text = raw.strip()

        if len(text) < 50:
            # Likely a scanned / image-only page → full OCR
            ocr = _ocr_page(path, i)
            if ocr:
                text = ocr
        else:
            # Mixed page: append OCR of embedded image blocks
            img_ocr = _ocr_image_blocks(path, i)
            if img_ocr:
                text = text + "\n\n[Image content on this page:]\n" + img_ocr

        if text:
            pages.append({"page": i + 1, "text": text})

    return pages

def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks

def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())

# ── Hybrid search (dense + BM25) ──────────────────────────────────────────────
def hybrid_search(query: str, cid: str, doc_ids: list, top_k: int = 20) -> list:
    qv = embed_model.encode(query).tolist()
    hits = qdrant.query_points(
        "rag_docs", query=qv, limit=top_k,
        query_filter=Filter(must=[
            FieldCondition(key="chat_id", match=MatchValue(value=cid)),
            FieldCondition(key="doc_id",  match=MatchAny(any=doc_ids)),
        ]),
    ).points
    hits = [h for h in hits if h.score > 0.15]
    if not hits:
        return []
    corpus  = [h.payload["text"] for h in hits]
    bm25    = BM25Okapi([tokenize(t) for t in corpus])
    bm25_sc = bm25.get_scores(tokenize(query))
    RRF_K   = 60
    d_rank  = {h.id: r + 1 for r, h in enumerate(hits)}
    b_order = sorted(range(len(hits)), key=lambda i: bm25_sc[i], reverse=True)
    b_rank  = {hits[i].id: r + 1 for r, i in enumerate(b_order)}
    rrf     = {h.id: 1 / (RRF_K + d_rank[h.id]) + 1 / (RRF_K + b_rank[h.id]) for h in hits}
    return sorted(hits, key=lambda h: rrf[h.id], reverse=True)

def rerank(query: str, candidates: list, top_n: int = 5) -> list:
    if not candidates:
        return []
    pairs  = [(query, c.payload["text"][:512]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top    = [(s, c) for s, c in ranked if s > -4][:top_n]
    for s, c in top:
        c.payload["rerank_score"] = float(s)
    return [c for _, c in top]

# ── FIX #5: proper overview/summary handling ──────────────────────────────────

OVERVIEW_RE = re.compile(
    r'\b(overview|summarize|summarise|summary|summarization|'
    r'what is this|what does this|tell me about|describe this|'
    r'give me a (summary|brief|overview)|brief summary|'
    r'what (topics|is covered|does it cover|are the main)|'
    r'main (topics|points|ideas|concepts|findings)|'
    r'key (points|takeaways|findings|topics|concepts|ideas)|'
    r'abstract|introduction|conclusion|'
    r'explain (this|the) (doc|document|pdf|file)|'
    r'what can (I|you) (learn|find) (in|from) (this|the))\b',
    re.IGNORECASE,
)

def is_overview_query(query: str) -> bool:
    return bool(OVERVIEW_RE.search(query))

def _fetch_overview_chunks(cid: str, doc_ids: list) -> list:
    """
    FIX #5: For summary/overview queries, instead of top-k similarity search
    (which clusters on one topic), we fetch chunks SPREAD across the document
    by dividing the page range into equal buckets and taking the best chunk from
    each bucket. This ensures the overview covers the full document.
    """
    # Fetch a large pool first
    pool = hybrid_search("document overview summary main topics key points", cid, doc_ids, top_k=60)
    if not pool:
        # Fallback: fetch anything
        qv = embed_model.encode("content").tolist()
        pool = qdrant.query_points(
            "rag_docs", query=qv, limit=60,
            query_filter=Filter(must=[
                FieldCondition(key="chat_id", match=MatchValue(value=cid)),
                FieldCondition(key="doc_id",  match=MatchAny(any=doc_ids)),
            ]),
        ).points

    if not pool:
        return []

    # Group by page, keep one chunk per page
    by_page: dict[int, object] = {}
    for h in pool:
        pg = h.payload.get("page", 0)
        if pg not in by_page:
            by_page[pg] = h

    pages_sorted = sorted(by_page.keys())
    total_pages  = len(pages_sorted)

    # Select up to 15 spread-out pages to cover the whole document
    n_buckets = min(15, total_pages)
    selected  = []
    if n_buckets > 0:
        step = max(1, total_pages // n_buckets)
        for i in range(0, total_pages, step):
            pg = pages_sorted[i]
            selected.append(by_page[pg])
            if len(selected) >= n_buckets:
                break

    return selected

RERANK_THRESHOLD = -2.5

def aggregate_sources(chunks: list) -> list:
    """
    One source entry per document.
    Lists all pages found across the top chunks.
    Hidden from UI when best rerank score < threshold (broad/overview queries).
    """
    by_doc: dict = {}
    for chunk in chunks:
        p      = chunk.payload
        doc_id = p.get("doc_id", "unknown")
        page   = p.get("page")
        score  = p.get("rerank_score", 0.0)
        fname  = p.get("filename", "Unknown")
        if doc_id not in by_doc:
            by_doc[doc_id] = {
                "doc_id": doc_id, "filename": fname,
                "pages": set(), "best_score": score,
            }
        if page:
            by_doc[doc_id]["pages"].add(page)
        if score > by_doc[doc_id]["best_score"]:
            by_doc[doc_id]["best_score"] = score

    valid = [
        d for d in by_doc.values()
        if d["best_score"] > RERANK_THRESHOLD and d["pages"]
    ]
    if not valid:
        return []

    best = sorted(valid, key=lambda d: d["best_score"], reverse=True)[:1]
    return [
        {"doc_id": d["doc_id"], "filename": d["filename"], "pages": sorted(d["pages"])}
        for d in best
    ]

# ── Request models ────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    chat_id: str
    doc_ids: List[str]

class RenameBody(BaseModel):
    name: str

# ── Chat routes ───────────────────────────────────────────────────────────────
@app.get("/chats")
def get_chats():
    c = db()
    rows = c.execute("SELECT * FROM chats ORDER BY created_at DESC").fetchall()
    c.close()
    return [dict(r) for r in rows]

@app.post("/chats")
def create_chat():
    cid = str(uuid.uuid4())
    c   = db()
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
    c    = db()
    docs = c.execute(
        "SELECT * FROM chat_docs WHERE chat_id=?", (cid,)
    ).fetchall()
    for d in docs:
        try:
            qdrant.delete(
                "rag_docs",
                points_selector=FilterSelector(
                    filter=Filter(must=[
                        FieldCondition(key="doc_id", match=MatchValue(value=d["id"]))
                    ])
                ),
            )
        except:
            pass
        try:
            os.remove(f"uploads/{d['id']}_{d['filename']}")
        except:
            pass
    c.execute("DELETE FROM messages WHERE chat_id=?",  (cid,))
    c.execute("DELETE FROM chat_docs WHERE chat_id=?", (cid,))
    c.execute("DELETE FROM chats WHERE id=?",          (cid,))
    c.commit()
    c.close()
    chat_memory.pop(cid, None)
    chat_summaries.pop(cid, None)
    return {"status": "deleted"}

@app.get("/chats/{cid}/messages")
def get_messages(cid: str):
    c    = db()
    rows = c.execute(
        "SELECT * FROM messages WHERE chat_id=? ORDER BY created_at ASC", (cid,)
    ).fetchall()
    c.close()
    result = []
    for r in rows:
        row            = dict(r)
        row["sources"] = json.loads(row["sources"]) if row["sources"] else []
        result.append(row)
    return result

# ── Document routes ───────────────────────────────────────────────────────────
@app.get("/chats/{cid}/documents")
def get_documents(cid: str):
    c    = db()
    rows = c.execute(
        "SELECT * FROM chat_docs WHERE chat_id=? ORDER BY created_at ASC", (cid,)
    ).fetchall()
    c.close()
    # FIX #6: coerce file_size to int so frontend always gets a number
    result = []
    for r in rows:
        row = dict(r)
        # Access by explicit column name to avoid any ordering confusion
        try:
            row["file_size"] = int(row["file_size"] or 0)
        except (ValueError, TypeError, KeyError):
            row["file_size"] = 0
        try:
            row["pages"] = int(row["pages"] or 0)
        except (ValueError, TypeError, KeyError):
            row["pages"] = 0
        # Normalize created_at to proper ISO-8601
        ca = str(row.get("created_at") or "")
        if ca and "T" not in ca:
            row["created_at"] = ca.replace(" ", "T") + "Z"
        elif ca and not ca.endswith("Z") and "+" not in ca:
            row["created_at"] = ca + "Z"
        result.append(row)
    return result

@app.post("/chats/{cid}/upload")
async def upload_pdf(cid: str, file: UploadFile = File(...)):
    try:
        c = db()
        if not c.execute("SELECT id FROM chats WHERE id=?", (cid,)).fetchone():
            raise HTTPException(404, "Chat not found")
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Only PDF files are allowed")
        doc_id = str(uuid.uuid4())
        os.makedirs("uploads", exist_ok=True)
        path = f"uploads/{doc_id}_{filename}"
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_size = int(os.path.getsize(path))
        pages = extract_pages(path)
        if not pages:
            os.remove(path)
            return {"error": "No readable text found in PDF"}
        points = []
        for item in pages:
            for chunk in chunk_text(item["text"]):
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embed_model.encode(chunk[:4000]).tolist(),
                    payload={
                        "chat_id": cid, "doc_id": doc_id,
                        "text": chunk, "page": item["page"],
                        "filename": filename,
                    },
                ))
        for i in range(0, len(points), 100):
            qdrant.upsert("rag_docs", points=points[i : i + 100])
        dn = " ".join(
            w.capitalize()
            for w in os.path.splitext(filename)[0]
               .replace("_", " ").replace("-", " ").split()[:5]
        )
        ts = now()
        c.execute(
            "INSERT INTO chat_docs VALUES (?,?,?,?,?,?,?)",
            (doc_id, cid, filename, dn, len(pages), file_size, ts),
        )
        chat = c.execute("SELECT name FROM chats WHERE id=?", (cid,)).fetchone()
        if chat and chat["name"] == "New Chat":
            c.execute("UPDATE chats SET name=? WHERE id=?", (dn, cid))
        c.commit()
        c.close()
        return {
            "doc_id": doc_id, "filename": filename,
            "display_name": dn, "pages": len(pages),
            "file_size": file_size, "created_at": ts,
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
    c   = db()
    row = c.execute(
        "SELECT filename FROM chat_docs WHERE id=? AND chat_id=?", (did, cid)
    ).fetchone()
    try:
        qdrant.delete(
            "rag_docs",
            points_selector=FilterSelector(
                filter=Filter(must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=did))
                ])
            ),
        )
    except:
        pass
    if row:
        try:
            os.remove(f"uploads/{did}_{row['filename']}")
        except:
            pass
    c.execute(
        "DELETE FROM chat_docs WHERE id=? AND chat_id=?", (did, cid)
    )
    c.commit()
    c.close()
    return {"status": "deleted"}

# ── Search + stream ───────────────────────────────────────────────────────────
@app.post("/chats/{cid}/search-stream")
async def search_stream(cid: str, req: SearchRequest):
    try:
        if not req.doc_ids:
            return {"error": "No documents selected"}
        if cid not in chat_memory:
            chat_memory[cid] = []

        overview = is_overview_query(req.query)

        if overview:
            # FIX #5: use spread-sampling for summary queries
            top_chunks = _fetch_overview_chunks(cid, req.doc_ids)
        else:
            candidates = hybrid_search(req.query, cid, req.doc_ids, top_k=20)
            if not candidates:
                async def empty():
                    yield {
                        "data": json.dumps({
                            "text": (
                                "I could not find relevant information "
                                "in the selected documents."
                            )
                        })
                    }
                return EventSourceResponse(empty())
            top_chunks = rerank(req.query, candidates, top_n=5)

        if not top_chunks:
            async def empty2():
                yield {
                    "data": json.dumps({
                        "text": (
                            "I could not find relevant information "
                            "in the selected documents."
                        )
                    })
                }
            return EventSourceResponse(empty2())

        context = "".join(
            f"[Page {c.payload['page']}]\n{c.payload['text']}\n\n"
            for c in top_chunks
        )

        # FIX #5: for overview, don't show sources (content spans many pages)
        sources = [] if overview else aggregate_sources(top_chunks)

        history = _build_history(cid)

        if overview:
            system_prompt = (
                "You are a precise document assistant tasked with producing a "
                "comprehensive overview or summary.\n"
                "Using ONLY the provided context (which covers multiple sections of the document), "
                "write a clear, well-structured summary that:\n"
                "- Starts with a one-sentence description of what the document is about\n"
                "- Uses ## headings for major sections/topics found\n"
                "- Uses bullet points for key facts, findings, or arguments\n"
                "- Uses **bold** for important terms\n"
                "- Ends with a brief 'Key Takeaways' section\n"
                "Do NOT mention page numbers, filenames, or admit any information is missing. "
                "Write as if you have read the whole document."
            )
        else:
            system_prompt = (
                "You are a precise document assistant. "
                "Answer ONLY from the provided context. "
                "If the answer is not in the context, say so clearly. "
                "Format with markdown: **bold** for key terms, bullet lists, "
                "code blocks for code. "
                "Do NOT mention filenames or page numbers. Be direct and concise."
            )

        async def generate():
            try:
                resp = groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                    max_tokens=2048 if overview else 1024,
                    stream=True,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *history,
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion:\n{req.query}",
                        },
                    ],
                )
                full = ""
                for chunk in resp:
                    t = chunk.choices[0].delta.content
                    if t:
                        full += t
                        yield {"data": json.dumps({"text": t})}
                c = db()
                c.execute(
                    "INSERT INTO messages "
                    "(chat_id,role,content,sources,created_at) VALUES (?,?,?,?,?)",
                    (cid, "user", req.query, None, now()),
                )
                c.execute(
                    "INSERT INTO messages "
                    "(chat_id,role,content,sources,created_at) VALUES (?,?,?,?,?)",
                    (cid, "assistant", full, json.dumps(sources), now()),
                )
                c.commit()
                c.close()
                chat_memory[cid].append({"role": "user",      "content": req.query})
                chat_memory[cid].append({"role": "assistant", "content": full})
                _maybe_summarise(cid)
                yield {"data": json.dumps({"sources": sources, "done": True})}
            except Exception as e:
                yield {"data": json.dumps({"error": str(e)})}

        return EventSourceResponse(generate())

    except Exception as e:
        return {"error": str(e)}