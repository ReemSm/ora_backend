import os
import re
import time
import math
import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Dict, Any, Callable

from openai import OpenAI
from pinecone import Pinecone


logging.basicConfig(level=logging.INFO, format="[ORA %(levelname)s] %(message)s")
log = logging.getLogger("ora")


@dataclass(frozen=True)
class Config:
    MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    PINECONE_INDEX: str = os.getenv("PINECONE_INDEX", "oraapp777")

    TOP_K_RAW: int = 12
    TOP_K_FINAL: int = 5

    MAX_REWRITE_OUTPUT_TOKENS: int = 64
    MAX_ANSWER_OUTPUT_TOKENS: int = 260
    TEMPERATURE: float = 0.25

    PINECONE_CHUNK_FIELD: str = "chunk_text"
    PINECONE_TITLE_FIELD: str = "title"
    PINECONE_AUTHORITY_FIELD: str = "authority_score"
    PINECONE_ID_FIELD: str = "id"

    HISTORY_CHAR_BUDGET: int = 2000
    MAX_REFERENCE_TITLES: int = 3
    MIN_AUTHORITY: float = 0.15

    MAX_CONTEXT_CHARS: int = 6500
    MAX_CHUNK_CHARS: int = 1500

    OPENAI_TIMEOUT_SECONDS: float = 25.0
    MAX_RETRIES: int = 4
    RETRY_BASE_SECONDS: float = 0.8


CFG = Config()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=CFG.OPENAI_TIMEOUT_SECONDS)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(CFG.PINECONE_INDEX)


ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
MULTISPACE_RE = re.compile(r"\s+")
ROLE_PREFIX_RE = re.compile(r"^\s*(system|assistant|user)\s*:\s*", re.IGNORECASE)


def with_retry(fn: Callable[[], any], label: str):
    for attempt in range(CFG.MAX_RETRIES):
        try:
            return fn()
        except Exception:
            if attempt == CFG.MAX_RETRIES - 1:
                raise
            time.sleep(CFG.RETRY_BASE_SECONDS * (2 ** attempt))


def normalize_ws(text: str) -> str:
    return MULTISPACE_RE.sub(" ", (text or "").strip())


def sanitize_context(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    lines = []
    for l in text.splitlines():
        l = ROLE_PREFIX_RE.sub("", l).strip()
        if l:
            lines.append(l)
    return "\n".join(lines)[:CFG.MAX_CHUNK_CHARS]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def is_ar(text: str) -> bool:
    ar = len(ARABIC_RE.findall(text or ""))
    en = len(re.findall(r"[A-Za-z]", text or ""))
    return ar > en * 0.3


def responses_text(messages, max_tokens, temp):
    def _call():
        return client.responses.create(
            model=CFG.MODEL,
            input=messages,
            max_output_tokens=max_tokens,
            temperature=temp,
        )
    r = with_retry(_call, "openai")
    return (getattr(r, "output_text", "") or "").strip()


@lru_cache(maxsize=512)
def embed_cached(text: str):
    def _call():
        return client.embeddings.create(model=CFG.EMBED_MODEL, input=text)
    r = with_retry(_call, "embed")
    return tuple(r.data[0].embedding)


def embed(text: str):
    return list(embed_cached(normalize_ws(text)))


def normalize_query(q: str, ar: bool):
    system = "Convert to short English dental retrieval query. Max 15 words."
    if ar:
        system = "Translate to English and convert to short dental retrieval query. Max 15 words."
    return responses_text(
        [{"role": "system", "content": system}, {"role": "user", "content": q}],
        CFG.MAX_REWRITE_OUTPUT_TOKENS,
        0.0
    ) or q


def query_pinecone(vector):
    def _call():
        return index.query(vector=vector, top_k=CFG.TOP_K_RAW, include_metadata=True)
    try:
        return with_retry(_call, "pinecone").get("matches", [])
    except:
        return []


def merge(matches_list):
    all_chunks = []
    for matches in matches_list:
        for m in matches:
            md = m.get("metadata") or {}
            text = sanitize_context(md.get(CFG.PINECONE_CHUNK_FIELD, ""))
            if not text:
                continue

            auth = md.get(CFG.PINECONE_AUTHORITY_FIELD)
            try:
                auth = float(auth) if auth is not None else None
            except:
                auth = 0.0

            if auth is not None and auth < CFG.MIN_AUTHORITY:
                continue

            score = float(m.get("score", 0))
            blended = score if auth is None else score * 0.9 + auth * 0.1

            all_chunks.append({
                "id": md.get(CFG.PINECONE_ID_FIELD) or hash_text(text),
                "title": md.get(CFG.PINECONE_TITLE_FIELD, ""),
                "text": text,
                "score": blended
            })

    all_chunks.sort(key=lambda x: x["score"], reverse=True)

    seen = set()
    out = []
    for c in all_chunks:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        out.append(c)
        if len(out) >= CFG.TOP_K_FINAL:
            break

    return out


def build_context(chunks):
    ctx = ""
    for i, c in enumerate(chunks):
        block = f"[{i+1}] {c['text']}\n\n"
        if len(ctx) + len(block) > CFG.MAX_CONTEXT_CHARS:
            break
        ctx += block
    return ctx


def answer_from_chunks(q, ar, chunks):
    ctx = build_context(chunks)

    system = (
        "You are a dental assistant. Use reference.\n"
        "Answer even if partial. Do not hallucinate specifics.\n"
        "If unsure, say limited info.\n"
    )

    if ar:
        system = (
            "أنت مساعد صحة فم وأسنان.\n"
            "استخدم المرجع فقط.\n"
            "إذا كانت المعلومات جزئية، أعطِ إجابة مفيدة بدون مبالغة.\n"
            "إذا غير متأكد، وضح أن المعلومات محدودة.\n"
        )

    return responses_text(
        [{"role": "system", "content": system + ctx}, {"role": "user", "content": q}],
        CFG.MAX_ANSWER_OUTPUT_TOKENS,
        CFG.TEMPERATURE
    )


def generate_answer(q: str, history=None):
    q = (q or "").strip()
    ar = is_ar(q)

    if not q:
        return {"answer": "Empty", "refs": []}

    rq = normalize_query(q, ar)

    queries = [q]
    if rq != q:
        queries.append(rq)

    matches = []
    for qu in queries:
        matches.append(query_pinecone(embed(qu)))

    chunks = merge(matches)

    if not chunks:
        return {"answer": "No info found.", "refs": []}

    ans = answer_from_chunks(q, ar, chunks)

    refs = []
    for c in chunks:
        if c["title"] and c["title"] not in refs:
            refs.append(c["title"])
        if len(refs) >= CFG.MAX_REFERENCE_TITLES:
            break

    return {"answer": ans, "refs": refs}
