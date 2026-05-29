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

import os, uuid, json, shutil, sqlite3, datetime, re, io, unicodedata
import asyncio 
from concurrent.futures import ThreadPoolExecutor

# ── Windows: Tesseract binary path ───────────────────────────────────────────
if os.name == "nt":
    try:
        import pytesseract as _pt
        _pt.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    except Exception:
        pass

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

# ══════════════════════════════════════════════════════════════════════════════
# CAPABILITY FLAGS
# ══════════════════════════════════════════════════════════════════════════════
def _check(name, fn):
    try:
        result = fn()
        print(f"[startup] {name}=OK ({result})")
        return True
    except Exception as e:
        print(f"[startup] {name}=MISSING ({e})")
        return False

_HAS_PYMUPDF   = _check("PyMuPDF",   lambda: __import__("fitz").__version__)
_HAS_PILLOW    = _check("Pillow",    lambda: __import__("PIL").__version__)
_HAS_TESSERACT = _check("Tesseract", lambda: __import__("pytesseract").get_tesseract_version())
_CAN_OCR       = _HAS_PYMUPDF and _HAS_PILLOW and _HAS_TESSERACT

# Check if Bengali language data is installed in Tesseract
_HAS_BEN_LANG = False
if _CAN_OCR:
    try:
        import pytesseract
        langs = pytesseract.get_languages()
        _HAS_BEN_LANG = "ben" in langs
        print(f"[startup] Tesseract languages: {langs}")
        print(f"[startup] Bengali (ben) available: {_HAS_BEN_LANG}")
    except Exception as e:
        print(f"[startup] Could not check Tesseract languages: {e}")

print(f"[startup] OCR_enabled={_CAN_OCR}  Bengali_OCR={_HAS_BEN_LANG}")

# ══════════════════════════════════════════════════════════════════════════════
# UNICODE BANGLA DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def _is_unicode_bangla(text: str) -> bool:
    """True if text has real Unicode Bangla codepoints (U+0980–U+09FF)."""
    bangla_chars = sum(1 for ch in text if '\u0980' <= ch <= '\u09ff')
    return bangla_chars > 5

def _is_garbled(text: str) -> bool:
    """
    True if pypdf extraction produced unreadable glyph-ID garbage.
    Garbled Bijoy PDFs typically give Latin characters that form no real words,
    or strings full of replacement characters / control chars.
    """
    if not text or len(text.strip()) < 10:
        return True
    # High ratio of replacement characters
    replacements = sum(1 for c in text if c == '\ufffd' or c == '\x00')
    if replacements / max(len(text), 1) > 0.05:
        return True
    # If it has NO Bangla Unicode AND consists mostly of Latin characters
    # that don't form English words (typical Bijoy glyph dump), treat as garbled.
    # We check: no Unicode Bangla AND high ratio of non-ASCII-printable junk
    has_unicode_bn = _is_unicode_bangla(text)
    if not has_unicode_bn:
        # Count characters that are not normal English/punctuation/numbers
        weird = sum(1 for c in text if ord(c) > 127 and ord(c) < 0x0980)
        if weird / max(len(text.strip()), 1) > 0.15:
            return True
    return False

def _fix_unicode(text: str) -> str:
    """NFC normalisation + optional ftfy mojibake repair."""
    text = unicodedata.normalize("NFC", text)
    try:
        import ftfy
        text = ftfy.fix_text(text)
    except ImportError:
        pass
    return text

# ══════════════════════════════════════════════════════════════════════════════
# OCR — the ONLY reliable method for Bijoy PDFs
# ══════════════════════════════════════════════════════════════════════════════
def _ocr_page_image(path: str, page_idx: int, dpi_scale: float = 2.5) -> str:
    """
    Rasterise a PDF page at high DPI and OCR it with Tesseract (ben+eng).
    Higher DPI = better accuracy for small Bangla text.
    2.5× means ~180 DPI source → 450 DPI raster, which Tesseract handles well.
    """
    if not _CAN_OCR:
        return ""
    try:
        import fitz
        import pytesseract
        from PIL import Image

        doc = fitz.open(path)
        mat = fitz.Matrix(dpi_scale, dpi_scale)
        pix = doc[page_idx].get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()

        # Use ben+eng if Bengali data is available, else eng only
        lang   = "ben+eng" if _HAS_BEN_LANG else "eng"
        config = "--psm 3 --oem 3"   # auto page segmentation, LSTM engine
        text   = pytesseract.image_to_string(img, lang=lang, config=config)
        return text.strip()
    except Exception as e:
        print(f"[ocr_page] page {page_idx} error: {e}")
        return ""

def _ocr_embedded_images(path: str, page_idx: int) -> str:
    """OCR embedded image objects on a mixed text+image page."""
    if not _CAN_OCR:
        return ""
    try:
        import fitz
        import pytesseract
        from PIL import Image

        doc   = fitz.open(path)
        page  = doc[page_idx]
        parts = []
        lang  = "ben+eng" if _HAS_BEN_LANG else "eng"

        for img_info in page.get_images(full=True):
            try:
                img_data = doc.extract_image(img_info[0])
                pil      = Image.open(io.BytesIO(img_data["image"]))
                if pil.size[0] < 120 or pil.size[1] < 120:
                    continue
                t = pytesseract.image_to_string(pil, lang=lang, config="--psm 3 --oem 3")
                if t.strip():
                    parts.append(t.strip())
            except Exception:
                continue
        doc.close()
        return "\n\n".join(parts)
    except Exception as e:
        print(f"[ocr_images] page {page_idx} error: {e}")
        return ""

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def _classify_pdf(path: str) -> str:
    """
    Classify PDF type: 'unicode_bangla', 'bijoy', 'image_only', 'english'
    """
    reader = PdfReader(path)
    sample_pages = list(reader.pages[:min(5, len(reader.pages))])

    all_text = ""
    for page in sample_pages:
        all_text += (page.extract_text() or "")

    total = len(all_text.strip())

    # Almost no text → image-only / scanned
    if total < 50:
        return "image_only"

    # Count Unicode Bangla characters (U+0980–U+09FF)
    unicode_bn = sum(1 for ch in all_text if '\u0980' <= ch <= '\u09ff')
    bn_ratio   = unicode_bn / total
    if bn_ratio > 0.15:
        return "unicode_bangla"

    # ── Bijoy detection ──────────────────────────────────────────────────────
    # Bijoy PDFs produce one of two patterns after pypdf extraction:
    #
    # Pattern A — high-ANSI dump (chars 128–255 outside Bangla Unicode range)
    #   e.g. characters like †, ‡, ·, ¸, ¹ etc.
    high_ansi = sum(1 for ch in all_text if 128 <= ord(ch) <= 0x097F)
    high_ansi_ratio = high_ansi / total
    if high_ansi_ratio > 0.08:
        return "bijoy"

    # Pattern B — looks like ASCII Latin but words are very short and
    #   contain lots of backtick/pipe/tilde/backslash typical of Bijoy glyph IDs
    # e.g. "evsjv‡`‡ki bvMwiKMY"
    bijoy_symbols = sum(1 for ch in all_text if ch in '`~|\\†‡ˆ‰·¸¹º»¼½¾¿')
    bijoy_sym_ratio = bijoy_symbols / total
    if bijoy_sym_ratio > 0.02:
        return "bijoy"

    # Pattern C — avg word length < 3 with no real English content
    words = [w for w in all_text.split() if w.isalpha() and w.isascii()]
    if words:
        avg_len = sum(len(w) for w in words) / len(words)
        # Check against known English words
        english_common = {
            'the','and','is','in','of','to','a','that','it','was',
            'for','on','are','as','with','his','they','at','be','this',
            'from','or','an','but','not','what','all','were','when','we',
        }
        word_set  = set(w.lower() for w in words)
        eng_hits  = len(word_set & english_common)
        total_words = len(words)

        # If avg word length is very short AND very few English hits → Bijoy
        if avg_len < 3.5 and eng_hits < 5:
            return "bijoy"

        # Even if avg length is OK, if basically no English words in a
        # mostly-Latin document → likely Bijoy glyph dump
        if total_words > 20 and eng_hits < 3:
            latin_ratio = len(words) / max(total, 1)
            if latin_ratio > 0.3:
                return "bijoy"

    return "english"

def extract_pages(path: str) -> list[dict]:
    """
    Extract text from every PDF page.

    Strategy depends on PDF type:

    unicode_bangla → pypdf extraction + NFC normalisation + image OCR for diagrams
    bijoy          → FULL PAGE OCR every page (pypdf text is unusable glyph garbage)
    image_only     → FULL PAGE OCR every page
    english        → pypdf extraction + NFC normalisation + image OCR for diagrams
    """
    pdf_type = _classify_pdf(path)
    reader   = PdfReader(path)
    pages    = []
    total    = len(reader.pages)

    print(f"[extract_pages] type={pdf_type}  total_pages={total}  file={os.path.basename(path)}")

    # Warn if OCR is needed but unavailable
    if pdf_type in ("bijoy", "image_only") and not _CAN_OCR:
        print(
            "[extract_pages] WARNING: This PDF needs OCR but PyMuPDF/Tesseract/Pillow "
            "is not installed. Text extraction will fail or be empty.\n"
            "Install: pip install pymupdf pytesseract pillow\n"
            "Also install Tesseract with Bengali language data."
        )

    for i, page in enumerate(reader.pages):

        # ── Bijoy or image-only: always OCR ──────────────────────────────────
        if pdf_type in ("bijoy", "image_only"):
            text = _ocr_page_image(path, i)
            if not text and pdf_type == "bijoy":
                # Last-resort: try native extraction in case this page is unicode
                raw = (page.extract_text() or "").strip()
                if _is_unicode_bangla(raw):
                    text = _fix_unicode(raw)
            if text and len(text.strip()) > 5:
                pages.append({"page": i + 1, "text": text})
            continue

        # ── Unicode Bangla or English: native extraction ──────────────────────
        raw = (page.extract_text() or "").strip()

        if not raw or len(raw) < 10 or _is_garbled(raw):
            # Fallback to OCR for pages that failed native extraction
            ocr_text = _ocr_page_image(path, i)
            if ocr_text and len(ocr_text.strip()) > 5:
                pages.append({"page": i + 1, "text": ocr_text})
            continue

        # Normalise and optionally supplement with embedded image OCR
        text = _fix_unicode(raw)
        if _CAN_OCR:
            img_ocr = _ocr_embedded_images(path, i)
            if img_ocr:
                text += "\n\n" + img_ocr

        if text and len(text.strip()) > 5:
            pages.append({"page": i + 1, "text": text})

    print(f"[extract_pages] extracted {len(pages)}/{total} pages successfully")
    return pages

# ══════════════════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks

def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())

# ══════════════════════════════════════════════════════════════════════════════
# SQLITE
# ══════════════════════════════════════════════════════════════════════════════
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
                {"role":"system","content":"Summarise the conversation in 3-5 sentences. Return only the summary."},
                {"role":"user","content":"\n".join(f"{m['role'].upper()}: {m['content']}" for m in to_sum)},
            ],
        )
        chat_summaries[cid] = resp.choices[0].message.content.strip()
        chat_memory[cid]    = msgs[-(WINDOW*2):]
    except: pass

def now(): return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# ══════════════════════════════════════════════════════════════════════════════
# HYBRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════
def hybrid_search(query: str, cid: str, doc_ids: list, top_k: int = 20) -> list:
    qv   = embed_model.encode(query).tolist()
    hits = qdrant.query_points(
        "rag_docs", query=qv, limit=top_k,
        query_filter=Filter(must=[
            FieldCondition(key="chat_id", match=MatchValue(value=cid)),
            FieldCondition(key="doc_id",  match=MatchAny(any=doc_ids)),
        ]),
    ).points
    hits = [h for h in hits if h.score > 0.15]
    if not hits: return []
    corpus  = [h.payload["text"] for h in hits]
    bm25    = BM25Okapi([tokenize(t) for t in corpus])
    bm25_sc = bm25.get_scores(tokenize(query))
    RRF_K   = 60
    d_rank  = {h.id: r+1 for r,h in enumerate(hits)}
    b_order = sorted(range(len(hits)), key=lambda i: bm25_sc[i], reverse=True)
    b_rank  = {hits[i].id: r+1 for r,i in enumerate(b_order)}
    rrf     = {h.id: 1/(RRF_K+d_rank[h.id]) + 1/(RRF_K+b_rank[h.id]) for h in hits}
    return sorted(hits, key=lambda h: rrf[h.id], reverse=True)

def rerank(query: str, candidates: list, top_n: int = 5) -> list:
    if not candidates: return []
    pairs  = [(query, c.payload["text"][:512]) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top    = [(s,c) for s,c in ranked if s > -4][:top_n]
    for s,c in top: c.payload["rerank_score"] = float(s)
    return [c for _,c in top]

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
RERANK_THRESHOLD = -2.5

def aggregate_sources(chunks: list) -> list:
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

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW / SUMMARY QUERY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
OVERVIEW_RE = re.compile(
    r'\b(overview|summarize|summarise|summary|summarization|'
    r'what is this|what does this|tell me about|describe this|'
    r'give me a (summary|brief|overview)|brief summary|'
    r'what (topics|is covered|does it cover|are the main)|'
    r'main (topics|points|ideas|concepts|findings)|'
    r'key (points|takeaways|findings|topics|concepts|ideas)|'
    r'abstract|introduction|conclusion|'
    r'explain (this|the) (doc|document|pdf|file))\b'
    r'|সারসংক্ষেপ|সারাংশ|সংক্ষেপ|সংক্ষিপ্ত|বিষয়বস্তু|'
    r'পর্যালোচনা|বিবরণ|মূল বিষয়|প্রধান বিষয়',
    re.IGNORECASE | re.UNICODE,
)

def is_overview_query(q: str) -> bool:
    return bool(OVERVIEW_RE.search(q))

def _fetch_overview_chunks(cid: str, doc_ids: list) -> list:
    pool = hybrid_search(
        "document overview summary main topics key points", cid, doc_ids, top_k=60
    )
    if not pool:
        qv   = embed_model.encode("content").tolist()
        pool = qdrant.query_points(
            "rag_docs", query=qv, limit=60,
            query_filter=Filter(must=[
                FieldCondition(key="chat_id", match=MatchValue(value=cid)),
                FieldCondition(key="doc_id",  match=MatchAny(any=doc_ids)),
            ]),
        ).points
    if not pool: return []
    by_page: dict[int, object] = {}
    for h in pool:
        pg = h.payload.get("page", 0)
        if pg not in by_page: by_page[pg] = h
    pages_sorted = sorted(by_page.keys())
    n_buckets    = min(15, len(pages_sorted))
    selected     = []
    if n_buckets > 0:
        step = max(1, len(pages_sorted) // n_buckets)
        for i in range(0, len(pages_sorted), step):
            selected.append(by_page[pages_sorted[i]])
            if len(selected) >= n_buckets: break
    return selected

# ══════════════════════════════════════════════════════════════════════════════
# REQUEST MODELS
# ══════════════════════════════════════════════════════════════════════════════
class SearchRequest(BaseModel):
    query: str; chat_id: str; doc_ids: List[str]

class RenameBody(BaseModel):
    name: str

# ══════════════════════════════════════════════════════════════════════════════
# CHAT ROUTES
# ══════════════════════════════════════════════════════════════════════════════
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
def rename_chat(cid: str, body: RenameBody):
    c = db(); c.execute("UPDATE chats SET name=? WHERE id=?", (body.name,cid)); c.commit(); c.close()
    return {"status":"ok"}

@app.delete("/chats/{cid}")
def delete_chat(cid: str):
    c = db(); docs = c.execute("SELECT * FROM chat_docs WHERE chat_id=?",(cid,)).fetchall()
    for d in docs:
        try: qdrant.delete("rag_docs",points_selector=FilterSelector(filter=Filter(must=[FieldCondition(key="doc_id",match=MatchValue(value=d["id"]))])))
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
def get_messages(cid: str):
    c = db(); rows = c.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY created_at ASC",(cid,)).fetchall(); c.close()
    result = []
    for r in rows:
        row = dict(r); row["sources"] = json.loads(row["sources"]) if row["sources"] else []; result.append(row)
    return result

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/chats/{cid}/documents")
def get_documents(cid: str):
    c = db(); rows = c.execute("SELECT * FROM chat_docs WHERE chat_id=? ORDER BY created_at ASC",(cid,)).fetchall(); c.close()
    result = []
    for r in rows:
        row = dict(r)
        try: row["file_size"] = int(row["file_size"] or 0)
        except: row["file_size"] = 0
        try: row["pages"] = int(row["pages"] or 0)
        except: row["pages"] = 0
        ca = str(row.get("created_at") or "")
        if ca and "T" not in ca: row["created_at"] = ca.replace(" ","T")+"Z"
        elif ca and not ca.endswith("Z") and "+" not in ca: row["created_at"] = ca+"Z"
        result.append(row)
    return result



_upload_executor = ThreadPoolExecutor(max_workers=2)

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

        # Save file to disk first
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_size = int(os.path.getsize(path))

        dn = " ".join(
            w.capitalize()
            for w in os.path.splitext(filename)[0]
               .replace("_", " ").replace("-", " ").split()[:5]
        )
        ts = now()

        # Insert doc record immediately so frontend gets a response fast
        c.execute(
            "INSERT INTO chat_docs VALUES (?,?,?,?,?,?,?)",
            (doc_id, cid, filename, dn, 0, file_size, ts),
        )
        chat = c.execute("SELECT name FROM chats WHERE id=?", (cid,)).fetchone()
        if chat and chat["name"] == "New Chat":
            c.execute("UPDATE chats SET name=? WHERE id=?", (dn, cid))
        c.commit()
        c.close()

        # Return immediately to frontend — processing continues in background
        asyncio.get_event_loop().run_in_executor(
            _upload_executor,
            _process_pdf_background,
            path, doc_id, cid, filename,
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "display_name": dn,
            "pages": 0,          # will update after processing
            "file_size": file_size,
            "created_at": ts,
            "processing": True,  # tells frontend it's still processing
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}


def _process_pdf_background(path: str, doc_id: str, cid: str, filename: str):
    """Runs in a thread — does the heavy OCR + embedding work."""
    try:
        pages = extract_pages(path)
        if not pages:
            print(f"[upload] No text extracted from {filename}")
            return

        points = []
        for item in pages:
            for chunk in chunk_text(item["text"]):
                if len(chunk.strip()) < 10:
                    continue
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embed_model.encode(chunk[:4000]).tolist(),
                    payload={
                        "chat_id": cid, "doc_id": doc_id,
                        "text": chunk, "page": item["page"],
                        "filename": filename,
                    },
                ))

        if not points:
            print(f"[upload] No chunks generated from {filename}")
            return

        for i in range(0, len(points), 100):
            qdrant.upsert("rag_docs", points=points[i:i + 100])

        # Update page count now that we know it
        c = db()
        c.execute(
            "UPDATE chat_docs SET pages=? WHERE id=?",
            (len(pages), doc_id),
        )
        c.commit()
        c.close()
        print(f"[upload] Done: {filename} — {len(pages)} pages, {len(points)} chunks")

    except Exception as e:
        print(f"[upload] Background processing error for {filename}: {e}")

@app.put("/chats/{cid}/documents/{did}")
def rename_document(cid: str, did: str, body: RenameBody):
    c = db(); c.execute("UPDATE chat_docs SET display_name=? WHERE id=? AND chat_id=?",(body.name,did,cid)); c.commit(); c.close()
    return {"status":"ok"}

@app.delete("/chats/{cid}/documents/{did}")
def delete_document(cid: str, did: str):
    c = db(); row = c.execute("SELECT filename FROM chat_docs WHERE id=? AND chat_id=?",(did,cid)).fetchone()
    try: qdrant.delete("rag_docs",points_selector=FilterSelector(filter=Filter(must=[FieldCondition(key="doc_id",match=MatchValue(value=did))])))
    except: pass
    if row:
        try: os.remove(f"uploads/{did}_{row['filename']}")
        except: pass
    c.execute("DELETE FROM chat_docs WHERE id=? AND chat_id=?",(did,cid)); c.commit(); c.close()
    return {"status":"deleted"}

# ══════════════════════════════════════════════════════════════════════════════
# SEARCH + STREAM
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/chats/{cid}/search-stream")
async def search_stream(cid: str, req: SearchRequest):
    try:
        if not req.doc_ids: return {"error":"No documents selected"}
        if cid not in chat_memory: chat_memory[cid] = []
        overview   = is_overview_query(req.query)
        if overview:
            top_chunks = _fetch_overview_chunks(cid, req.doc_ids)
        else:
            candidates = hybrid_search(req.query, cid, req.doc_ids, top_k=20)
            if not candidates:
                async def empty():
                    yield {"data":json.dumps({"text":"আমি নির্বাচিত ডকুমেন্টে প্রাসঙ্গিক তথ্য খুঁজে পাইনি। / I could not find relevant information in the selected documents."})}
                return EventSourceResponse(empty())
            top_chunks = rerank(req.query, candidates, top_n=5)
        if not top_chunks:
            async def empty2():
                yield {"data":json.dumps({"text":"আমি নির্বাচিত ডকুমেন্টে প্রাসঙ্গিক তথ্য খুঁজে পাইনি। / I could not find relevant information in the selected documents."})}
            return EventSourceResponse(empty2())
        context = "".join(f"[Page {c.payload['page']}]\n{c.payload['text']}\n\n" for c in top_chunks)
        sources  = [] if overview else aggregate_sources(top_chunks)
        history  = _build_history(cid)
        if overview:
            system_prompt = (
                "You are a document assistant. The document is written in Bengali (Bangla). "
                "Produce a comprehensive overview using ONLY the provided context. "
                "Respond in Bengali (Bangla). "
                "Structure your response:\n"
                "- One sentence describing what the document is about\n"
                "- ## headings for major sections or topics found in the document\n"
                "- Bullet points for specific facts, articles, or provisions\n"
                "- **Bold** for important terms\n"
                "- A 'মূল বিষয়সমূহ' (Key Takeaways) section at the end\n"
                "Do NOT make up content. Only use what is in the context. "
                "Do NOT mention page numbers or filenames."
            )
        else:
            system_prompt = (
                "You are a precise document assistant. "
                "The document is written in Bengali (Bangla). "
                "Answer in the same language as the question: "
                "Bangla question → Bangla answer, English question → English answer. "
                "Answer ONLY from the provided context. "
                "If the answer is not in the context, say so clearly. "
                "Use **bold** for key terms, bullet lists for multiple points, "
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