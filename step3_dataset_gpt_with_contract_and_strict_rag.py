import os
import re
import logging
from typing import Dict, Any

from openai import OpenAI
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="[ORA %(levelname)s] %(message)s")
log = logging.getLogger("ora")

MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-large"
PINECONE_INDEX = "oraapp777"

TOP_K = 8
MAX_ANSWER_TOKENS = 260

PINECONE_CHUNK_FIELD = "chunk_text"
PINECONE_TITLE_FIELD = "title"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX)

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

GREETINGS = {"hi", "hello", "hey", "مرحبا", "هلا", "السلام", "السلام عليكم"}

_query_cache = {}
_embedding_cache = {}


def is_ar(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))


def is_greeting(q: str) -> bool:
    return q.strip().lower() in GREETINGS


def translate_to_english(q: str) -> str:
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Translate to clear English for dental retrieval. Output only translation."},
                {"role": "user", "content": q},
            ],
            temperature=0,
        )
        return (r.choices[0].message.content or "").strip() or q
    except:
        return q


def should_rewrite(q: str) -> bool:
    q = q.strip()
    if len(q.split()) <= 6:
        return True
    if re.search(r"[^\w\s]", q):
        return True
    return False


def rewrite_query(q: str) -> str:
    if q in _query_cache:
        return _query_cache[q]

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Clean the query for retrieval. Fix typos and informal wording only. Do not change meaning. Do not reinterpret the condition."
                },
                {"role": "user", "content": q},
            ],
            temperature=0,
        )
        out = (r.choices[0].message.content or "").strip() or q
        _query_cache[q] = out
        return out
    except:
        return q


def embed(text: str):
    if text in _embedding_cache:
        return _embedding_cache[text]

    emb = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    _embedding_cache[text] = emb
    return emb


def extract_text(md: Dict[str, Any]) -> str:
    return str(md.get(PINECONE_CHUNK_FIELD) or "").strip()


def retrieve_chunks(query: str):
    try:
        res = index.query(vector=embed(query), top_k=TOP_K, include_metadata=True)
        matches = res.get("matches", [])
    except:
        return []

    chunks = []
    for m in matches:
        md = m.get("metadata") or {}
        text = extract_text(md)
        if not text:
            continue

        chunks.append({
            "title": str(md.get(PINECONE_TITLE_FIELD) or ""),
            "text": text,
        })

    return chunks


def is_relevant(q: str, chunks) -> bool:
    if not chunks:
        return False

    context = " ".join(c.get("text", "") for c in chunks)

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Is this reference directly relevant to answering this exact oral health question? Answer only yes or no. Say no for greetings or non-oral-health topics. If uncertain, answer yes."
                },
                {
                    "role": "user",
                    "content": f"Question: {q}\n\nReference:\n{context}",
                },
            ],
            temperature=0,
        )

        if not r or not r.choices:
            return True

        content = r.choices[0].message.content
        if not content:
            return True

        return content.strip().lower().startswith("yes")

    except Exception as e:
        log.error(f"RELEVANCE ERROR: {e}")
        return True
