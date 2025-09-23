# main.py
import os, time, json, hashlib
from typing import List, Optional, Dict, Any, Deque
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
import re

# ── load env ──────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")
WIX_ORIGIN = os.getenv("WIX_ORIGIN", "https://www.vimarshafoundation.org")

# Rate limit config
MAX_REQ_PER_IP_PER_DAY = int(os.getenv("MAX_REQ_PER_IP_PER_DAY", "50"))

# Redis (optional but recommended)
REDIS_URL = os.getenv("REDIS_URL")
try:
    import redis  # add redis>=5.0.1 to requirements.txt
    r = redis.from_url(REDIS_URL) if REDIS_URL else None
except Exception:
    r = None

# OpenAI client (lazy: app still boots if missing)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Vimarsha Chat API", version="1.2")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ── models ────────────────────────────────────────────────────────────────────
class AskIn(BaseModel):
    question: str
    userId: Optional[str] = None

class AskOut(BaseModel):
    answer: str
    references: List[str] = []

# ── rate limiting + caching ───────────────────────────────────────────────────
WINDOW_SECONDS = 24 * 60 * 60
_ip_hits: Dict[str, Deque[float]] = defaultdict(deque)

def strip_bold(text: str) -> str:
    if not text:
        return text
    return re.sub(r"\*\*(.*?)\*\*", r"\1", text)

def get_client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else request.client.host

def check_rate_limit(request: Request):
    ip = get_client_ip(request)
    today_key = f"rl:{ip}:{time.strftime('%Y%m%d')}"
    if r:
        try:
            pipe = r.pipeline()
            pipe.incr(today_key, 1)
            pipe.expire(today_key, 2 * 24 * 60 * 60)
            count, _ = pipe.execute()
            if int(count) > MAX_REQ_PER_IP_PER_DAY:
                raise HTTPException(429, "Rate limit exceeded. Try again tomorrow.")
            return
        except Exception:
            pass
    # fallback: in-memory
    now = time.time()
    q = _ip_hits[ip]
    while q and (now - q[0] > WINDOW_SECONDS):
        q.popleft()
    if len(q) >= MAX_REQ_PER_IP_PER_DAY:
        raise HTTPException(429, "Rate limit exceeded. Try again tomorrow.")
    q.append(now)

def _qkey(q: str) -> str:
    norm = " ".join((q or "").lower().split())
    return "cache:q:" + hashlib.sha256(norm.encode("utf-8")).hexdigest()

def get_cached_answer(q: str):
    if not r:
        return None
    try:
        val = r.get(_qkey(q))
        return json.loads(val) if val else None
    except Exception:
        return None

def set_cached_answer(q: str, payload: Dict[str, Any], ttl: int = 86400):
    if not r:
        return
    try:
        r.setex(_qkey(q), ttl, json.dumps(payload))
    except Exception:
        pass

# ── OpenAI helpers ────────────────────────────────────────────────────────────
def ensure_env_ready():
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
        fname = getattr(meta, "filename", f"file:{fid}") if meta else f"file:{fid}"
        # strip file extension like .pdf, .txt, etc.
        base, _ = os.path.splitext(fname)
        return base
    except Exception:
        return f"file:{fid}"

# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/__ping")
def __ping():
    return "OK"

@app.get("/redis-test")
def redis_test():
    if not r:
        return {"ok": False, "error": "Redis not configured"}
    try:
        r.set("hello", "world", ex=60)
        val = r.get("hello")
        return {"ok": True, "hello": val.decode() if val else None}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/debug/env")
def debug_env():
    def present(val): return {"present": bool(val), "length": len(val) if val else 0}
    return {
        "OPENAI_API_KEY": present(OPENAI_API_KEY),
        "OPENAI_MODEL": present(OPENAI_MODEL),
        "OPENAI_VECTOR_STORE_ID": present(OPENAI_VECTOR_STORE_ID),
        "WIX_ORIGIN": present(WIX_ORIGIN),
        "REDIS_URL_set": bool(REDIS_URL),
        "MAX_REQ_PER_IP_PER_DAY": MAX_REQ_PER_IP_PER_DAY,
    }

@app.post("/ask", response_model=AskOut)
def ask(body: AskIn, request: Request):
    ensure_env_ready()
    if client is None:
        raise HTTPException(500, "OpenAI client not initialized.")

    # rate-limit
    check_rate_limit(request)

    q = (body.question or "").strip()
    if not q:
        raise HTTPException(400, "Question is required")

    # cache lookup
    cached = get_cached_answer(q)
    if cached:
        return AskOut(**cached)

    system_msg = (
        "You are a strict research assistant. Use ONLY the file_search tool and the provided vector store. "
        "Every sentence must be backed by retrieved passages with inline markers like [1], [2]. "
        "Do NOT use outside knowledge. If no relevant passages are found, reply exactly: "
        "'I don’t have evidence for that in the provided documents.' "
        "Do NOT include a References section; the UI will render references. "
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
        )
    except Exception as e:
        raise HTTPException(502, f"OpenAI call failed: {e}")

    answer = getattr(resp, "output_text", None) or "No answer."
    answer = strip_bold(answer)
    d = resp.model_dump()
    file_ids = collect_citation_file_ids(d)
    refs = [file_id_to_filename(fid) for fid in file_ids]

    if not refs:
        payload = {"answer": "I don’t have evidence for that in the provided documents.", "references": []}
        set_cached_answer(q, payload)
        return AskOut(**payload)

    wc = len(answer.split())
    if wc < 1 or wc > 1000:
        answer = "The response length is outside allowed bounds (1–1000 words). Please refine your question."

    payload = {"answer": answer, "references": refs}
    set_cached_answer(q, payload)
    return AskOut(**payload)