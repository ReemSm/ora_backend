import os
import re
import logging
from typing import List, Dict, Any

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

PINECONE_CHUNK_FIELD = "chunk_text"
PINECONE_TITLE_FIELD = "title"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX)

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

# cache
_query_cache = {}
_embedding_cache = {}


def is_ar(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))


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
    except Exception as e:
        log.error(f"Translation error: {e}")
        return q


def rewrite_query_for_retrieval(q: str) -> str:
    if q in _query_cache:
        return _query_cache[q]

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Rewrite into a clean short dental query. Fix spelling and clarity only. Preserve original intent exactly. Do not alter medical meaning. Do not change to a different condition. If the query looks like a typo of a dental term, correct it to the closest valid dental meaning."},
                {"role": "user", "content": q},
            ],
            temperature=0,
        )
        out = (r.choices[0].message.content or "").strip() or q
        _query_cache[q] = out
        return out
    except Exception as e:
        log.error(f"Rewrite error: {e}")
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
            "title": str(md.get(PINECONE_TITLE_FIELD) or ""),
            "text": text,
            "score": float(m.get("score", 0)),
        })

    chunks.sort(key=lambda x: x["score"], reverse=True)

    seen = set()
    unique = []
    for c in chunks:
        if c["title"] not in seen:
            seen.add(c["title"])
            unique.append(c)
        if len(unique) >= TOP_K_FINAL:
            break

    log.info(f"RAG chunks used: {[c['title'] for c in unique]}")
    log.info(f"RAG scores: {[c['score'] for c in unique]}")

    return unique


def build_system_prompt(context: str, lang: str) -> str:
    return f"""
You are a strict evaluator.

Output language: {lang}

You must answer ONLY using the reference material below.

If the answer is not explicitly found in the reference material, output exactly:
NOT_FOUND

Do not use prior knowledge.
Do not infer.
Do not guess.
Do not answer from examples.
Do not answer from training data.
Do not add information that is not explicitly supported by the reference material.
Do not explain why information is missing.
Do not say anything except the grounded answer or NOT_FOUND.

REFERENCE MATERIAL:
{context}
"""


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

    return (r.choices[0].message.content or "").strip()


def generate_answer(q: str, history=None):
    q = (q or "").strip()
    log.info(f"INCOMING QUESTION: {q}")

    ar = is_ar(q)
    lang = "arabic" if ar else "english"

    base_query = translate_to_english(q) if ar else q
    clean_query = rewrite_query_for_retrieval(base_query)

    chunks = retrieve_chunks(clean_query)
    
    print("DEBUG len_chunks:", len(chunks))
    if chunks:
        print("DEBUG top_score:", chunks[0]["score"])
        print("DEBUG titles:", [c["title"] for c in chunks])
    
    if not chunks or chunks[0]["score"] < 0.8:
        return {
        "answer": answer_from_chunks(q, [], lang, history),
        "refs": [],
        "source": "model"
    }
    
    answer = answer_from_chunks(q, chunks, lang, history)
    log.info(f"FINAL ANSWER: {answer}")

    refs = list({c["title"] for c in chunks if c["title"]})[:3]
    
    return {
        "answer": answer,
        "refs": refs,
        "source": "rag"
    }
