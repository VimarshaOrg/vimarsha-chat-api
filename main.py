# main.py
import os, time, json, hashlib, re, html, concurrent.futures
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
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "86400"))            # 1 day
BROAD_TIMEOUT_SECONDS = int(os.getenv("BROAD_TIMEOUT_SECONDS", "8"))        # UX guardrail

# Redis (optional)
REDIS_URL = os.getenv("REDIS_URL")
try:
    import redis  # requirements.txt: redis>=5.0.1
    r = redis.from_url(REDIS_URL) if REDIS_URL else None
except Exception:
    r = None

# In-memory cache fallback
_cache_mem: Dict[str, Dict[str, Any]] = {}

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI(title="Vimarsha Chat API", version="2.0-paragraphs")

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
    # NOTE: answer is HTML (not Markdown)
    answer: str
    references: List[str] = []

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: rate limiting, caching
# ──────────────────────────────────────────────────────────────────────────────
WINDOW_SECONDS = 24 * 60 * 60
_ip_hits: Dict[str, Deque[float]] = defaultdict(deque)

def normalize_question(q: str) -> str:
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
# Markdown-lite → HTML (bold + numbered/unnumbered lists + paragraphs)
# ──────────────────────────────────────────────────────────────────────────────
_bold_re = re.compile(r"\*\*(.+?)\*\*")
_sentence_split = re.compile(r"(?<=[.!?])\s+")
_has_marker = re.compile(r"\[\d+\]")

def md_lite_to_html(md: str) -> str:
    """
    Converts very simple Markdown to HTML:
      - **bold** → <strong>
      - lines "1. ..." "2. ..." → <ol><li>…</li></ol>
      - lines starting with '-' or '* ' → <ul><li>…</li></ul>
      - other non-empty lines → <p>…</p>
    HTML-escapes all text first, then restores <strong>.
    """
    if not md:
        return ""
    lines = md.strip().splitlines()
    esc_lines = [html.escape(line) for line in lines]
    def restore_bold(s: str) -> str:
        return _bold_re.sub(r"<strong>\1</strong>", s)
    esc_lines = [restore_bold(s) for s in esc_lines]

    html_parts: List[str] = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            html_parts.append("</ul>")
            in_ul = False
        if in_ol:
            html_parts.append("</ol>")
            in_ol = False

    for s in esc_lines:
        stripped = s.strip()
        if not stripped:
            close_lists()
            continue

        # ordered list: "1. " or "23. "
        if re.match(r"^\d+\.\s+", stripped):
            if in_ul:
                html_parts.append("</ul>")
                in_ul = False
            if not in_ol:
                html_parts.append("<ol>")
                in_ol = True
            item = re.sub(r"^\d+\.\s+", "", stripped)
            html_parts.append(f"<li>{item}</li>")
            continue

        # unordered list: "- " or "* "
        if stripped.startswith("- ") or stripped.startswith("* "):
            if in_ol:
                html_parts.append("</ol>")
                in_ol = False
            if not in_ul:
                html_parts.append("<ul>")
                in_ul = True
            item = stripped[2:].strip()
            html_parts.append(f"<li>{item}</li>")
            continue

        # paragraph
        close_lists()
        html_parts.append(f"<p>{stripped}</p>")

    close_lists()
    return "".join(html_parts)

def force_inline_if_single_ref(answer_text: str, refs: List[str]) -> str:
    """If exactly one reference and no [n] markers, append [1] to each paragraph end."""
    if not answer_text or len(refs) != 1:
        return answer_text
    if _has_marker.search(answer_text):
        return answer_text
    # Add [1] at end of each paragraph line
    lines = [ln.strip() for ln in answer_text.strip().splitlines()]
    new_lines = []
    for ln in lines:
        if not ln:
            new_lines.append(ln)
            continue
        if not _has_marker.search(ln):
            new_lines.append(ln + " [1]")
        else:
            new_lines.append(ln)
    return "\n".join(new_lines)

# ──────────────────────────────────────────────────────────────────────────────
# Prompts (paragraph-first; optional numbered list)
# ──────────────────────────────────────────────────────────────────────────────
STRICT_MSG = (
    "Use ONLY the file_search tool with the provided vector store. "
    "Retrieve the top ~6 most relevant, non-duplicate passages. "
    "Write a coherent mini-essay of 2–3 paragraphs (free-flowing prose; no headings). "
    "End each paragraph with compact source markers like [1] or [1][2] that support that paragraph. "
    "Optionally include a short numbered list (1., 2., …) if it genuinely clarifies sub-points. "
    "Avoid speculation. If no evidence is found, reply exactly: "
    "'I don’t have evidence for that in the provided documents.' "
    "Target 180–350 words. Plain text with **bold** allowed; no links/tables/images."
)

BROAD_MSG = (
    "Use ONLY the file_search tool with the provided vector store. "
    "Retrieve about 10–12 diverse passages from different documents (prefer coverage over repetition). "
    "Write a fuller answer with 3–4 paragraphs of free-flowing prose (no headings). "
    "Each paragraph should end with compact markers like [1] or [2][3]. "
    "You may add one short numbered list (1., 2., …) if it truly helps understanding. "
    "Remove unsupported claims. If no evidence is found, reply exactly: "
    "'I don’t have evidence for that in the provided documents.' "
    "Target 250–450 words. Plain text with **bold** allowed; no links/tables/images."
)

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
        "BROAD_TIMEOUT_SECONDS": BROAD_TIMEOUT_SECONDS,
    }

@app.post("/ask", response_model=AskOut)
def ask(body: AskIn, request: Request):
    ensure_env_ready()
    if client is None:
        raise HTTPException(500, "OpenAI client not initialized.")

    q = (body.question or "").strip()
    if not q:
        raise HTTPException(400, "Question is required")

    # ==== CACHE FIRST ====
    cached = get_cached_answer(q)
    if cached:
        return AskOut(**cached)

    # Rate-limit only when calling OpenAI
    check_rate_limit(request)

    # --- 1) STRICT attempt (paragraph-first) ---
    try:
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": STRICT_MSG},
                {"role": "user", "content": q},
            ],
            tools=[{"type": "file_search", "vector_store_ids": [OPENAI_VECTOR_STORE_ID]}],
            temperature=0.3,
            max_output_tokens=1400,
        )
    except Exception as e:
        raise HTTPException(502, f"OpenAI call failed: {e}")

    answer_txt = getattr(resp, "output_text", None) or "No answer."
    d = resp.model_dump()
    file_ids = collect_citation_file_ids(d)
    refs = [file_id_to_filename(fid) for fid in file_ids]

    if refs:
        wc = len(answer_txt.split())
        if wc < 1 or wc > 1000:
            answer_txt = "The response length is outside allowed bounds (1–1000 words). Please refine your question."
        answer_txt = force_inline_if_single_ref(answer_txt, refs)
        answer_html = md_lite_to_html(answer_txt)
        payload = {"answer": answer_html, "references": refs}
        set_cached_answer(q, payload)
        return AskOut(**payload)

    # --- 2) BROAD attempt with timeout (diverse passages) ---
    def broad_call():
        return client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": BROAD_MSG},
                {"role": "user", "content": q},
            ],
            tools=[{"type": "file_search", "vector_store_ids": [OPENAI_VECTOR_STORE_ID]}],
            temperature=0.3,
            max_output_tokens=1600,
        )

    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(broad_call)
            resp2 = future.result(timeout=BROAD_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError:
        payload = {"answer": md_lite_to_html("Search took too long, please refine your question."), "references": []}
        set_cached_answer(q, payload, ttl=60)  # short cache to keep UX snappy
        return AskOut(**payload)
    except Exception:
        payload = {"answer": md_lite_to_html("I don’t have evidence for that in the provided documents."), "references": []}
        set_cached_answer(q, payload)
        return AskOut(**payload)

    answer2_txt = getattr(resp2, "output_text", None) or "No answer."
    d2 = resp2.model_dump()
    file_ids2 = collect_citation_file_ids(d2)
    refs2 = [file_id_to_filename(fid) for fid in file_ids2]

    if refs2:
        wc2 = len(answer2_txt.split())
        if wc2 < 1 or wc2 > 1000:
            answer2_txt = "The response length is outside allowed bounds (1–1000 words). Please refine your question."
        answer2_txt = force_inline_if_single_ref(answer2_txt, refs2)
        answer2_html = md_lite_to_html(answer2_txt)
        payload = {"answer": answer2_html, "references": refs2}
        set_cached_answer(q, payload)
        return AskOut(**payload)

    payload = {"answer": md_lite_to_html("I don’t have evidence for that in the provided documents."), "references": []}
    set_cached_answer(q, payload)
    return AskOut(**payload)