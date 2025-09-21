# main.py
import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# ── load env (works locally; Railway uses real env) ───────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")
WIX_ORIGIN = os.getenv("WIX_ORIGIN", "https://www.vimarshafoundation.org")

# Don’t raise here; allow app to start so we can debug /debug/env
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Vimarsha Chat API", version="1.1")

# CORS: loosen for first deploy; tighten later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to [WIX_ORIGIN] once working
    allow_credentials=True,
    allow_methods=["POST","GET","OPTIONS"],
    allow_headers=["*"],
)

# ── models ───────────────────────────────────────────────────────────────────
class AskIn(BaseModel):
    question: str
    userId: Optional[str] = None

class AskOut(BaseModel):
    answer: str
    references: List[str] = []


# ── helpers ──────────────────────────────────────────────────────────────────
def ensure_env_ready():
    """Raise a helpful error if required env vars are missing."""
    missing = []
    if not OPENAI_API_KEY: missing.append("OPENAI_API_KEY")
    if not OPENAI_VECTOR_STORE_ID: missing.append("OPENAI_VECTOR_STORE_ID")
    if missing:
        raise HTTPException(
            500,
            f"Server not configured: missing {', '.join(missing)}. "
            "Set these in Railway → Service → Variables and redeploy."
        )

def collect_citation_file_ids(obj: Any) -> List[str]:
    order, seen = [], set()
    def walk(x: Any):
        if isinstance(x, dict):
            fc = x.get("file_citation")
            if isinstance(fc, dict):
                fid = fc.get("file_id")
                if fid and fid not in seen:
                    seen.add(fid); order.append(fid)
            fid = x.get("file_id")
            if isinstance(fid, str) and fid and fid not in seen:
                seen.add(fid); order.append(fid)
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(obj)
    return order

def file_id_to_filename(fid: str) -> str:
    try:
        meta = client.files.retrieve(fid) if client else None
        return getattr(meta, "filename", f"file:{fid}") if meta else f"file:{fid}"
    except Exception:
        return f"file:{fid}"


# ── routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/__ping")
def __ping():
    return "OK"

@app.get("/")
def root():
    return {
        "ok": True,
        "message": "Vimarsha Chat API",
        "endpoints": ["/health", "/debug/env", "POST /ask"],
    }

@app.get("/debug/env")
def debug_env():
    def present(val): return {"present": bool(val), "length": len(val) if val else 0}
    return {
        "OPENAI_API_KEY": present(OPENAI_API_KEY),
        "OPENAI_MODEL": present(OPENAI_MODEL),
        "OPENAI_VECTOR_STORE_ID": present(OPENAI_VECTOR_STORE_ID),
        "WIX_ORIGIN": present(WIX_ORIGIN),
    }

@app.post("/ask", response_model=AskOut)
def ask(body: AskIn, request: Request):
    ensure_env_ready()
    if client is None:
        raise HTTPException(500, "OpenAI client not initialized.")

    q = (body.question or "").strip()
    if not q:
        raise HTTPException(400, "Question is required")

    system_msg = (
        "You are a strict research assistant. Use ONLY the file_search tool and the provided vector store. "
        "Every sentence must be backed by retrieved passages with inline markers like [1], [2]. "
        "Do NOT use outside knowledge. If no relevant passages are found, reply exactly: "
        "'I don’t have evidence for that in the provided documents.' "
        "Do NOT include a References section; the UI will render references."
        "Do not use Markdown formatting (no **bold**, no headings). Output plain text only."
    )

    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": q},
            ],
            tools=[{"type": "file_search", "vector_store_ids": [OPENAI_VECTOR_STORE_ID]}],
            temperature=0.2,
            # optional cap:
            # max_output_tokens=800,
        )
    except Exception as e:
        detail = str(e)
        r = getattr(e, "response", None)
        if r is not None:
            try:
                status = getattr(r, "status_code", "?")
                txt = getattr(r, "text", "")
                detail = f"HTTP {status}: {txt}"
            except Exception:
                pass
        raise HTTPException(502, f"OpenAI call failed: {detail}")

    answer = getattr(resp, "output_text", None) or "No answer."

    d = resp.model_dump()
    file_ids = collect_citation_file_ids(d)
    refs = [file_id_to_filename(fid) for fid in file_ids]

    # if no citations, enforce grounded-only policy:
    if not refs:
        return AskOut(
            answer="I don’t have evidence for that in the provided documents.",
            references=[]
        )

    # word bound enforcement (1–1000 words)
    wc = len(answer.split())
    if wc < 1 or wc > 1000:
        answer = "⚠️ The response length is outside allowed bounds (1–1000 words). Please refine your question."

    return AskOut(answer=answer, references=refs)