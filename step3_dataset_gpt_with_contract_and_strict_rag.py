import os
import re
import logging
from typing import List, Dict, Any
from functools import lru_cache

from openai import OpenAI
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="[ORA %(levelname)s] %(message)s")
log = logging.getLogger("ora")

MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-large"
PINECONE_INDEX = "oraapp777"

TOP_K_RAW = 12
TOP_K_FINAL = 5
MAX_ANSWER_TOKENS = 260

# NEW: threshold to prevent garbage answers
MIN_SCORE_THRESHOLD = 0.35

PINECONE_CHUNK_FIELD = "chunk_text"
PINECONE_TITLE_FIELD = "title"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX)

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def is_ar(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))


# -------------------------
# NORMALIZATION (NEW)
# -------------------------
def normalize_ar(text: str) -> str:
    if not text:
        return text

    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)  # remove diacritics
    text = text.replace("ة", "ه").replace("ى", "ي")
    return text


def enforce_gauze_term(text: str) -> str:
    # aggressive normalization + replacement
    base = normalize_ar(text)
    base = re.sub(r"(شاش|شاشه|شمه|شمة|شاش طبي)", "قطعة الشاش", base)
    return base


# -------------------------
# REWRITE (FIXED ORDER)
# -------------------------
@lru_cache(maxsize=500)
def rewrite_query_for_retrieval(q: str) -> str:
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Fix spelling and typos only. "
                        "Preserve exact medical meaning. "
                        "Do NOT translate. "
                        "If typo resembles a dental word, correct it."
                    )
                },
                {"role": "user", "content": q},
            ],
            temperature=0,
        )
        return (r.choices[0].message.content or "").strip() or q
    except Exception as e:
        log.error(f"Rewrite error: {e}")
        return q


# -------------------------
# TRANSLATION (UNCHANGED but AFTER rewrite)
# -------------------------
@lru_cache(maxsize=500)
def translate_to_english(q: str) -> str:
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Translate to clear English for dental retrieval. Output only translation."
                },
                {"role": "user", "content": q},
            ],
            temperature=0,
        )
        return (r.choices[0].message.content or "").strip() or q
    except Exception as e:
        log.error(f"Translation error: {e}")
        return q


# -------------------------
# EMBEDDING (cached)
# -------------------------
@lru_cache(maxsize=1000)
def embed(text: str):
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def extract_text(md: Dict[str, Any]) -> str:
    return str(md.get(PINECONE_CHUNK_FIELD) or "").strip()


# -------------------------
# RETRIEVAL (FIXED)
# -------------------------
def retrieve_chunks(query: str):
    try:
        res = index.query(vector=embed(query), top_k=TOP_K_RAW, include_metadata=True)
        matches = res.get("matches", [])
    except Exception as e:
        log.error(f"Retrieval error: {e}")
        return []

    chunks = []
    for m in matches:
        md = m.get("metadata") or {}
        text = extract_text(md)
        if not text:
            continue

        chunks.append({
            "id": m.get("id"),  # NEW: dedupe by id
            "title": str(md.get(PINECONE_TITLE_FIELD) or ""),
            "text": text,
            "score": float(m.get("score", 0)),
        })

    # sort by score
    chunks.sort(key=lambda x: x["score"], reverse=True)

    # NEW: threshold filter
    if not chunks or chunks[0]["score"] < MIN_SCORE_THRESHOLD:
        log.info("Low confidence retrieval, skipping answer")
        return []

    # dedupe by id (FIXED)
    seen = set()
    unique = []
    for c in chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            unique.append(c)
        if len(unique) >= TOP_K_FINAL:
            break

    log.info(f"RAG chunks used: {[c['title'] for c in unique]}")
    log.info(f"RAG scores: {[c['score'] for c in unique]}")

    return unique


# -------------------------
# ANSWER
# -------------------------
def answer_from_chunks(q: str, chunks, lang: str, history=None):
    context = "\n\n".join(c["text"] for c in chunks)
    system = build_system_prompt(context, lang)

    messages = [{"role": "system", "content": system}]

    if history:
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": q})

    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=MAX_ANSWER_TOKENS,
    )

    answer = (r.choices[0].message.content or "").strip()

    # enforce terminology (FIXED)
    if lang == "arabic":
        answer = enforce_gauze_term(answer)

    return answer


# -------------------------
# MAIN PIPELINE (FIXED ORDER)
# -------------------------
def generate_answer(q: str, history=None):
    q = (q or "").strip()

    if not q:
        return {"answer": "Invalid query.", "refs": [], "source": "none"}

    ar = is_ar(q)
    lang = "arabic" if ar else "english"

    # STEP 1: rewrite FIRST (FIXED)
    cleaned_query = rewrite_query_for_retrieval(q)

    # STEP 2: translate AFTER rewrite
    retrieval_query = translate_to_english(cleaned_query) if ar else cleaned_query

    # STEP 3: retrieve
    chunks = retrieve_chunks(retrieval_query)

    # STEP 4: guard
    if not chunks:
        return {"answer": "No relevant data found.", "refs": [], "source": "none"}

    # STEP 5: answer
    answer = answer_from_chunks(q, chunks, lang, history)

    refs = list({c["title"] for c in chunks if c["title"]})[:3]

    return {
        "answer": answer,
        "refs": refs,
        "source": "rag"
    }
