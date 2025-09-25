# main.py
import os, time, json, hashlib, re
from typing import List, Optional, Dict, Any, Deque
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# ── load env ──────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")
WIX_ORIGIN = os.getenv("WIX_ORIGIN", "https://www.vimarshafoundation.org")

# formatting mode: "html" (default here) or "text"
ANSWER_FORMAT = os.getenv("ANSWER_FORMAT", "html").lower().strip()

# rate limit config
MAX_REQ_PER_IP_PER_DAY = int(os.getenv("MAX_REQ_PER_IP_PER_DAY", "50"))

# cache TTL (seconds)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 1 day

# Redis (optional)
REDIS_URL = os.getenv("REDIS_URL")
try:
    import redis  # ensure redis>=5.0.1 in requirements.txt
    r = redis.from_url(REDIS_URL) if REDIS_URL else None
except Exception:
    r = None

# Manual flush protection
FLUSH_TOKEN = os.getenv("FLUSH_TOKEN")  # set a random string in Railway vars

# In-memory cache fallback
_cache_mem: Dict[str, Dict[str, Any]] = {}

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Vimarsha Chat API", version="1.7")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to [WIX_ORIGIN] when ready
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ── models ────────────────────────────────────────────────────────────────────
class AskIn(BaseModel):
    question: str
    userId: Optional[str] = None

class AskOut(BaseModel):
    answer: str   # HTML if ANSWER_FORMAT=html, plain text otherwise
    references: List[str] = []

# ── helpers: text, rate-limiting, caching ─────────────────────────────────────
WINDOW_SECONDS = 24 * 60 * 60
_ip_hits: Dict[str, Deque[float]] = defaultdict(deque)

def normalize_question(q: str) -> str:
    """Normalize to increase cache hits across small variations."""
    q = (q or "").lower().strip()
    q = re.sub(r"[\s\-–—_:;,.!?/\\]+", " ", q)  # collapse punctuation/whitespace
    return q

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
    # fallback: in-memory window counter
    now = time.time()
    q = _ip_hits[ip]
    while q and (now - q[0] > WINDOW_SECONDS):
        q.popleft()
    if len(q) >= MAX_REQ_PER_IP_PER_DAY:
        raise HTTPException(429, "Rate limit exceeded. Try again tomorrow.")
    q.append(now)

def _cache_key(q: str) -> str:
    norm = normalize_question(q)
    return "cache:q:" + hashlib.sha256(norm.encode("utf-8")).hexdigest()

def _mem_get(key: str):
    now = time.time()
    item = _cache_mem.get(key)
    if not item:
        return None
    if item["exp"] < now:
        _cache_mem.pop(key, None)
        return None
    return item["payload"]

def _mem_set(key: str, payload: Dict[str, Any], ttl: int):
    _cache_mem[key] = {"exp": time.time() + ttl, "payload": payload}

def get_cached_answer(q: str):
    key = _cache_key(q)
    # Redis first
    if r:
        try:
            val = r.get(key)
            if val:
                try:
                    payload = json.loads(val)
                    print({"cache": "hit", "backend": "redis"})
                    return payload
                except Exception:
                    pass
        except Exception:
            pass
    # Memory fallback
    payload = _mem_get(key)
    if payload:
        print({"cache": "hit", "backend": "memory"})
        return payload
    print({"cache": "miss"})
    return None

def set_cached_answer(q: str, payload: Dict[str, Any], ttl: int = CACHE_TTL_SECONDS):
    key = _cache_key(q)
    if r:
        try:
            r.setex(key, ttl, json.dumps(payload))
        except Exception:
            pass
    _mem_set(key, payload, ttl)

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
        base, _ = os.path.splitext(fname)  # strip extension (.pdf)
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
        "CACHE_TTL_SECONDS": CACHE_TTL_SECONDS,
        "ANSWER_FORMAT": ANSWER_FORMAT,
        "FLUSH_TOKEN_set": bool(FLUSH_TOKEN),
    }

# ======== SYSTEM PROMPTS (text vs html) ========
SYSTEM_MSG_TEXT = (
    "Use ONLY the file_search tool with the provided vector store. "
    "Retrieve about 6–8 of the most relevant passages from diverse documents; prefer coverage over repetition. "
    "Every claim must be supported by retrieved passages with inline numeric markers like [1], [2]. "
    "If only one document is used, cite it consistently as [1]. "
    "If no evidence is found, reply exactly: 'I don’t have evidence for that in the provided documents.' "
    "Plain text only (no Markdown). Keep between 1 and 1000 words."
)

SYSTEM_MSG_HTML = (
    "Use ONLY the file_search tool with the provided vector store. "
    "Retrieve about 6–8 of the most relevant passages from diverse documents; prefer coverage over repetition. "
    "Every claim must be supported by retrieved passages with inline numeric markers like [1], [2]. "
    "If only one document is used, cite it consistently as [1]. "
    "If no evidence is found, reply exactly: 'I don’t have evidence for that in the provided documents.' "
    "FORMAT: Output clean HTML only (no scripts/styles). Use <h3> for brief section headings, "
    "<p> for paragraphs, <strong> for key terms (bold), and <ul><li> for bullet points when helpful. "
    "Keep between 1 and 1000 words."
)

@app.post("/ask", response_model=AskOut)
def ask(body: AskIn, request: Request):
    t0 = time.time()
    ensure_env_ready()
    if client is None:
        raise HTTPException(500, "OpenAI client not initialized.")

    q = (body.question or "").strip()
    if not q:
        raise HTTPException(400, "Question is required")

    # CACHE FIRST
    cached = get_cached_answer(q)
    if cached:
        print({"phase": "cache_hit", "t_total_ms": int((time.time()-t0)*1000)})
        return AskOut(**cached)

    # Only rate-limit if we’ll call OpenAI
    check_rate_limit(request)

    # Choose prompt based on ANSWER_FORMAT
    system_msg = SYSTEM_MSG_HTML if ANSWER_FORMAT == "html" else SYSTEM_MSG_TEXT

    # OpenAI call
    t_ai0 = time.time()
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": q},
            ],
            tools=[{"type": "file_search", "vector_store_ids": [OPENAI_VECTOR_STORE_ID]}],
            temperature=0.0,
            max_output_tokens=1000,   # allow richer answers
        )
    except Exception as e:
        raise HTTPException(502, f"OpenAI call failed: {e}")
    t_ai1 = time.time()

    # Build answer + refs
    answer = getattr(resp, "output_text", None) or ""

    # enforce word bound (strip HTML tags if present for count)
    wc = len(re.sub(r"<[^>]+>", " ", answer).split())
    if wc < 1 or wc > 1000:
        answer = "The response length is outside allowed bounds (1–1000 words). Please refine your question." \
                 if ANSWER_FORMAT == "text" \
                 else "<p>The response length is outside allowed bounds (1–1000 words). Please refine your question.</p>"

    d = resp.model_dump()
    file_ids = collect_citation_file_ids(d)
    refs = [file_id_to_filename(fid) for fid in file_ids]

    # Only keep first 8 unique filenames in order of appearance
    seen, ordered = set(), []
    for nm in refs:
        if nm not in seen:
            seen.add(nm)
            ordered.append(nm)
    refs = ordered[:8]

    if not refs:
        payload = {
            "answer": ("I don’t have evidence for that in the provided documents."
                       if ANSWER_FORMAT == "text"
                       else "<p>I don’t have evidence for that in the provided documents.</p>"),
            "references": []
        }
        set_cached_answer(q, payload)
        print({
            "phase": "cache_miss_no_refs",
            "t_total_ms": int((time.time() - t0) * 1000),
            "t_openai_ms": int((t_ai1 - t_ai0) * 1000),
        })
        return AskOut(**payload)

    payload = {"answer": answer, "references": refs}
    set_cached_answer(q, payload)

    print({
        "phase": "cache_miss",
        "t_total_ms": int((time.time() - t0) * 1000),
        "t_openai_ms": int((t_ai1 - t_ai0) * 1000),
    })
    return AskOut(**payload)

# ── Manual cache wipe (Redis + in-memory) ─────────────────────────────────────
@app.post("/debug/flush-cache")
def flush_cache(x_admin_token: Optional[str] = Header(None, convert_underscores=False)):
    """
    Wipes cached Q&A. Protect with FLUSH_TOKEN env.
    Call with: curl -X POST -H "X-Admin-Token: <your-token>" https://.../debug/flush-cache
    """
    if not FLUSH_TOKEN:
        raise HTTPException(403, "Flush disabled: FLUSH_TOKEN not set on server.")
    if x_admin_token != FLUSH_TOKEN:
        raise HTTPException(403, "Forbidden: invalid X-Admin-Token.")

    # in-memory
    _cache_mem.clear()

    # Redis
    if r:
        try:
            # delete only our cache keys (prefix "cache:q:")
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = r.scan(cursor=cursor, match="cache:q:*", count=1000)
                if keys:
                    r.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            return {"ok": True, "message": f"Cache cleared (redis keys deleted: {deleted})."}
        except Exception as e:
            return {"ok": True, "message": f"Cache cleared (memory). Redis error: {str(e)}"}
    return {"ok": True, "message": "Cache cleared (memory only)."}