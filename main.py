# main.py
import os, time, json, hashlib, re
from typing import List, Optional, Dict, Any, Deque
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Env & setup
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")
WIX_ORIGIN = os.getenv("WIX_ORIGIN", "https://www.vimarshafoundation.org")

MAX_REQ_PER_IP_PER_DAY = int(os.getenv("MAX_REQ_PER_IP_PER_DAY", "50"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 1 day

# Redis (optional)
REDIS_URL = os.getenv("REDIS_URL")
try:
    import redis  # ensure requirements.txt has: redis>=5.0.1
    r = redis.from_url(REDIS_URL) if REDIS_URL else None
except Exception:
    r = None

# In-memory cache fallback
_cache_mem: Dict[str, Dict[str, Any]] = {}

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Vimarsha Chat API", version="1.6-md")

# ──────────────────────────────────────────────────────────────────────────────
# CORS (loose for testing; tighten to [WIX_ORIGIN] when live)
# ──────────────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to [WIX_ORIGIN] when ready
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────
class AskIn(BaseModel):
    question: str
    userId: Optional[str] = None

class AskOut(BaseModel):
    answer: str           # now returns Markdown (bold + bullets)
    references: List[str] = []

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: rate limiting, caching
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_SECONDS = 24 * 60 * 60
_ip_hits: Dict[str, Deque[float]] = defaultdict(deque)

def normalize_question(q: str) -> str:
    """Normalize to increase cache hits across small variations."""
    q = (q or "").lower().strip()
    q = re.sub(r"[\s\-–—_:;,.!?/\\]+", " ", q)
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
    if r:
        try:
            r.setex(key, ttl, json.dumps(payload))
        except Exception:
            pass
    _mem_set(key, payload, ttl)

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI helpers
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Two-stage retrieval: STRICT → BROAD (fallback)
# (Now allowing basic Markdown: **bold**, and "-" bullet points)
# ──────────────────────────────────────────────────────────────────────────────
STRICT_MSG = (
    "Use ONLY the file_search tool with the provided vector store. "
    "Retrieve at most the top 3 most relevant passages; ignore lower-score matches. "
    "Every sentence must be supported by retrieved passages with inline [1], [2] markers. "
    "If no evidence, reply exactly: 'I don’t have evidence for that in the provided documents.' "
    "Output in plain text with basic Markdown: use **bold** for key terms and '-' for bullet points. "
    "Do not use headings, tables, links, or images."
)

BROAD_MSG = (
    "Use ONLY the file_search tool with the provided vector store. "
    "Retrieve about 6–8 relevant passages from diverse documents; prefer coverage over repetition. "
    "Aim to support every sentence with inline markers like [1], [2]. "
    "If any sentence cannot be strictly supported, remove it rather than speculate. "
    "If no evidence is found, reply exactly: 'I don’t have evidence for that in the provided documents.' "
    "Output in plain text with basic Markdown: use **bold** for key terms and '-' for bullet points. "
    "Do not use headings, tables, links, or images."
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

    # ==== CACHE FIRST ====
    t_c0, cached = time.time(), get_cached_answer(q)
    t_c1 = time.time()
    if cached:
        print({
            "phase": "cache_hit",
            "t_total_ms": int((time.time() - t0) * 1000),
            "t_cache_ms": int((t_c1 - t_c0) * 1000),
        })
        return AskOut(**cached)

    # Rate limit only if we’re going to call OpenAI
    t_rl0 = time.time()
    check_rate_limit(request)
    t_rl1 = time.time()

    # --- 1) STRICT attempt (top-3) ---
    t_ai0 = time.time()
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": STRICT_MSG},
                {"role": "user", "content": q},
            ],
            tools=[{"type": "file_search", "vector_store_ids": [OPENAI_VECTOR_STORE_ID]}],
            temperature=0.0,
            max_output_tokens=800,
        )
    except Exception as e:
        raise HTTPException(502, f"OpenAI call failed: {e}")
    t_ai1 = time.time()

    answer = getattr(resp, "output_text", None) or "No answer."
    d = resp.model_dump()
    file_ids = collect_citation_file_ids(d)
    refs = [file_id_to_filename(fid) for fid in file_ids]

    # If STRICT found citations → return
    if refs:
        wc = len(answer.split())
        if wc < 1 or wc > 1000:
            answer = "The response length is outside allowed bounds (1–1000 words). Please refine your question."
        payload = {"answer": answer, "references": refs}
        set_cached_answer(q, payload)
        print({
            "phase": "cache_miss_strict_success",
            "t_total_ms": int((time.time() - t0) * 1000),
            "t_cache_ms": int((t_c1 - t_c0) * 1000),
            "t_rate_limit_ms": int((t_rl1 - t_rl0) * 1000),
            "t_openai_strict_ms": int((t_ai1 - t_ai0) * 1000),
        })
        return AskOut(**payload)

    # --- 2) BROAD attempt (6–8 diverse) if STRICT produced no refs ---
    try:
        t_ai2 = time.time()
        resp2 = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": BROAD_MSG},
                {"role": "user", "content": q},
            ],
            tools=[{"type": "file_search", "vector_store_ids": [OPENAI_VECTOR_STORE_ID]}],
            temperature=0.0,
            max_output_tokens=900,
        )
        t_ai3 = time.time()

        answer2 = getattr(resp2, "output_text", None) or "No answer."
        d2 = resp2.model_dump()
        file_ids2 = collect_citation_file_ids(d2)
        refs2 = [file_id_to_filename(fid) for fid in file_ids2]

        if refs2:
            wc2 = len(answer2.split())
            if wc2 < 1 or wc2 > 1000:
                answer2 = "The response length is outside allowed bounds (1–1000 words). Please refine your question."
            payload = {"answer": answer2, "references": refs2}
            set_cached_answer(q, payload)
            print({
                "phase": "cache_miss_broad_success",
                "t_total_ms": int((time.time() - t0) * 1000),
                "t_rate_limit_ms": int((t_rl1 - t_rl0) * 1000),
                "t_openai_strict_ms": int((t_ai1 - t_ai0) * 1000),
                "t_openai_broad_ms": int((t_ai3 - t_ai2) * 1000),
            })
            return AskOut(**payload)
        else:
            payload = {"answer": "I don’t have evidence for that in the provided documents.", "references": []}
            set_cached_answer(q, payload)
            print({
                "phase": "cache_miss_no_refs_both",
                "t_total_ms": int((time.time() - t0) * 1000),
                "t_rate_limit_ms": int((t_rl1 - t_rl0) * 1000),
                "t_openai_strict_ms": int((t_ai1 - t_ai0) * 1000),
                "t_openai_broad_ms": int((t_ai3 - t_ai2) * 1000),
            })
            return AskOut(**payload)
    except Exception as e:
        # If BROAD errors, return strict fallback
        payload = {"answer": "I don’t have evidence for that in the provided documents.", "references": []}
        set_cached_answer(q, payload)
        print({
            "phase": "cache_miss_strict_no_refs_broad_error",
            "error": str(e),
            "t_total_ms": int((time.time() - t0) * 1000),
            "t_openai_strict_ms": int((t_ai1 - t_ai0) * 1000),
        })
        return AskOut(**payload)