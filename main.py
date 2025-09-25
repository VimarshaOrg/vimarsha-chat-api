# main.py
import os, time, json, hashlib, re
from typing import List, Optional, Dict, Any, Deque
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
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

# Rate limit config
MAX_REQ_PER_IP_PER_DAY = int(os.getenv("MAX_REQ_PER_IP_PER_DAY", "50"))

# Cache TTL (seconds)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 1 day

# Redis (optional but recommended)
REDIS_URL = os.getenv("REDIS_URL")
try:
    import redis  # ensure: redis>=5.0.1 in requirements.txt
    r = redis.from_url(REDIS_URL) if REDIS_URL else None
except Exception:
    r = None

# In-memory cache fallback: key -> {"exp": float, "payload": dict}
_cache_mem: Dict[str, Dict[str, Any]] = {}

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Vimarsha Chat API", version="1.4")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later to [WIX_ORIGIN]
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

# ── helpers: text cleanup, rate limiting, caching ─────────────────────────────
WINDOW_SECONDS = 24 * 60 * 60
_ip_hits: Dict[str, Deque[float]] = defaultdict(deque)

def strip_bold(text: str) -> str:
    if not text:
        return text
    return re.sub(r"\*\*(.*?)\*\*", r"\1", text)

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

def _qkey(q: str) -> str:
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
    key = _qkey(q)
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
    key = _qkey(q)
    # Redis
    if r:
        try:
            r.setex(key, ttl, json.dumps(payload))
        except Exception:
            pass
    # Memory
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
        base, _ = os.path.splitext(fname)  # strip extension like .pdf
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
    }

@app.post("/ask", response_model=AskOut)
def ask(body: AskIn, request: Request):
    t0 = time.time()
    ensure_env_ready()
    if client is None:
        raise HTTPException(500, "OpenAI client not initialized.")

    q = (body.question or "").strip()
    if not q:
        raise HTTPException(400, "Question is required")

    # ==== CACHE FIRST (fast path) ====
    t_c0 = time.time()
    cached = get_cached_answer(q)
    t_c1 = time.time()
    if cached:
        print({
            "phase": "cache_hit",
            "t_total_ms": int((time.time() - t0) * 1000),
            "t_cache_ms": int((t_c1 - t_c0) * 1000),
        })
        return AskOut(**cached)

    # Only rate-limit if we’ll call OpenAI
    t_rl0 = time.time()
    check_rate_limit(request)
    t_rl1 = time.time()

    # System prompt (keep quality; nudge to use minimal relevant context)
    system_msg = (
        "Use ONLY the file_search tool with the provided vector store. "
        "Retrieve at most the top 3 most relevant passages; ignore lower-score matches. "
        "Every sentence must be supported by retrieved passages with inline [1], [2] markers. "
        "If no evidence, reply exactly: 'I don’t have evidence for that in the provided documents.' "
        "Plain text only (no Markdown)."
    )

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
            temperature=0.0,        # quality-focused, less rambling
            max_output_tokens=600,  # allow rich answers
        )
    except Exception as e:
        raise HTTPException(502, f"OpenAI call failed: {e}")
    t_ai1 = time.time()

    # Token usage log
    usage = getattr(resp, "usage", None)
    try:
        print({
            "phase": "usage",
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        })
    except Exception:
        pass

    # Build answer + refs
    answer = getattr(resp, "output_text", None) or "No answer."
    answer = strip_bold(answer)

    d = resp.model_dump()
    file_ids = collect_citation_file_ids(d)
    refs = [file_id_to_filename(fid) for fid in file_ids]

    if not refs:
        payload = {"answer": "I don’t have evidence for that in the provided documents.", "references": []}
        set_cached_answer(q, payload)
        print({
            "phase": "cache_miss_no_refs",
            "t_total_ms": int((time.time() - t0) * 1000),
            "t_cache_ms": int((t_c1 - t_c0) * 1000),
            "t_rate_limit_ms": int((t_rl1 - t_rl0) * 1000),
            "t_openai_ms": int((t_ai1 - t_ai0) * 1000),
        })
        return AskOut(**payload)

    # Word bound enforcement (kept from your earlier logic)
    wc = len(answer.split())
    if wc < 1 or wc > 1000:
        answer = "The response length is outside allowed bounds (1–1000 words). Please refine your question."

    payload = {"answer": answer, "references": refs}
    set_cached_answer(q, payload)

    print({
        "phase": "cache_miss",
        "t_total_ms": int((time.time() - t0) * 1000),
        "t_cache_ms": int((t_c1 - t_c0) * 1000),
        "t_rate_limit_ms": int((t_rl1 - t_rl0) * 1000),
        "t_openai_ms": int((t_ai1 - t_ai0) * 1000),
    })
    return AskOut(**payload)