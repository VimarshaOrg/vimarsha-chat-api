import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# ── env ──────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")

if not OPENAI_API_KEY or not OPENAI_VECTOR_STORE_ID:
    raise RuntimeError("Set OPENAI_API_KEY and OPENAI_VECTOR_STORE_ID env vars")

client = OpenAI(api_key=OPENAI_API_KEY)

# ── app & CORS ───────────────────────────────────────────────────────────────
app = FastAPI(title="Vimarsha Chat API", version="1.0")

# For quick testing. Before production, replace ["*"] with your Wix origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # e.g., ["https://www.vimarshafoundation.org"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── models ───────────────────────────────────────────────────────────────────
class AskIn(BaseModel):
    question: str
    userId: Optional[str] = None  # (optional) pass Wix member id for quotas later

class AskOut(BaseModel):
    answer: str
    references: List[str] = []    # filenames in [1], [2] order


# ── helpers ──────────────────────────────────────────────────────────────────
def collect_citation_file_ids(obj: Any) -> List[str]:
    """
    Traverse the Responses API JSON and collect file_ids from 'file_citation'
    annotations in the order they first appear.
    """
    order: List[str] = []
    seen = set()

    def walk(x: Any):
        if isinstance(x, dict):
            fc = x.get("file_citation")
            if isinstance(fc, dict):
                fid = fc.get("file_id")
                if fid and fid not in seen:
                    seen.add(fid)
                    order.append(fid)

            # sometimes file_id/quote appear directly
            fid = x.get("file_id")
            if isinstance(fid, str) and fid and fid not in seen:
                seen.add(fid)
                order.append(fid)

            for v in x.values():
                walk(v)

        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return order


def file_id_to_filename(fid: str) -> str:
    try:
        meta = client.files.retrieve(fid)
        # new SDK returns 'filename' for user-uploaded files
        return getattr(meta, "filename", f"file:{fid}")
    except Exception:
        return f"file:{fid}"


# ── routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def home():
    return {
        "ok": True,
        "message": "Vimarsha Chat API",
        "endpoints": {
            "health": "/health",
            "ask": "POST /ask (json: { question: string, userId?: string })",
            "debug_store": "/debug/store"
        }
    }

@app.get("/debug/store")
def debug_store():
    """Quick check: how many files are in the store and their status."""
    try:
        fl = client.vector_stores.files.list(OPENAI_VECTOR_STORE_ID)
        out = [{"id": f.id, "status": f.status} for f in fl.data]
        return {"count": len(out), "files": out}
    except Exception as e:
        raise HTTPException(500, f"List vector store files failed: {e}")

@app.post("/ask", response_model=AskOut)
def ask(body: AskIn):
    q = (body.question or "").strip()
    if not q:
        raise HTTPException(400, "Question is required")

    # ── call OpenAI Responses with strict grounding instructions ─────────────
    system_msg = (
        "You are a strict research assistant. Use ONLY the provided file_search tool. "
        "Every sentence in your answer must be backed by retrieved passages with inline numeric markers [1], [2]. "
        "Do NOT invent, speculate, or use outside knowledge. "
        "If no relevant passages are found, reply exactly with: "
        "'I don’t have evidence for that in the provided documents.'\n\n"
        "Do NOT include a References section. Only use inline markers. "
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
        )
    except Exception as e:
        # Show useful error details if available
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

    # ── extract text answer ───────────────────────────────────────────────────
    answer: str = getattr(resp, "output_text", None) or "No answer."

    # ── collect citation file_ids and map to filenames ───────────────────────
    d: Dict[str, Any] = resp.model_dump()
    file_ids = collect_citation_file_ids(d)
    refs: List[str] = [file_id_to_filename(fid) for fid in file_ids]

    # ── helpful fallback if no citations were returned ───────────────────────
    if not refs:
        refs = [
            "(No file citations were returned. The model may have answered from general "
            "knowledge or retrieval found no strongly matching passages.)"
        ]

    return AskOut(answer=answer, references=refs)