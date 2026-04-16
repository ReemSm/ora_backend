import os
import re
import logging
from typing import Dict, Any

from openai import OpenAI
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="[ORA %(levelname)s] %(message)s")
log = logging.getLogger("ora")

MODEL = "gpt-4o"
FAST_MODEL = "gpt-4o-mini"
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
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": "Translate to clear English for dental retrieval. Output only translation."},
                {"role": "user", "content": q},
            ],
            temperature=0,
            max_tokens=200,
        )
        return (r.choices[0].message.content or "").strip() or q
    except Exception as e:
        log.warning(f"translate_to_english failed: {e}")
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
            model=FAST_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Clean the query for retrieval. Fix typos and informal wording only. Do not change meaning. Do not reinterpret the condition."
                },
                {"role": "user", "content": q},
            ],
            temperature=0,
            max_tokens=150,
        )
        out = (r.choices[0].message.content or "").strip() or q
        _query_cache[q] = out
        return out
    except Exception as e:
        log.warning(f"rewrite_query failed: {e}")
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
    except Exception as e:
        log.warning(f"retrieve_chunks failed: {e}")
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

    context = "\n\n".join(c["text"] for c in chunks[:4])

    try:
        r = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a relevance checker for a dental health assistant. "
                        "Given a question and retrieved reference material, decide if the "
                        "reference contains information useful for answering the question. "
                        "Answer only yes or no. Say no for greetings or clearly non-dental topics. "
                        "If uncertain, answer yes."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {q}\n\nReference material:\n{context}",
                },
            ],
            temperature=0,
            max_tokens=5,
        )
        out = (r.choices[0].message.content or "").strip().lower()
        return out.startswith("yes")
    except Exception as e:
        log.warning(f"is_relevant failed: {e}")
        return True


def build_system_prompt(context: str, lang: str) -> str:
    return f"""
You are an oral health assistant.

Output language: {lang}

- Do not hallucinate
- Answer only what was asked
- Always use "lost vitality" instead of "nerve died"
- استخدم "فقد حيويته" ولا تستخدم "مات"

Q: I just had a tooth extraction what should I do
A:
- Bite on gauze for 30 minutes
- Use a cold compress during the first 30 minutes
- Do not spit or rinse for 24 hours
- Do not use a straw for 24 hours
- Avoid hot or hard food
- Brush normally but avoid the extraction site
- Take medications if prescribed
- Avoid smoking and physical activity for 24 hours

Q: خلعت سني وش أسوي
A:
- اضغط على قطعة شاش لمدة 30 دقيقة
- استخدم كمادات باردة خلال أول 30 دقيقة
- لا تبصق ولا تتمضمض لمدة 24 ساعة
- لا تستخدم الشفاط لمدة 24 ساعة
- تجنب الأكل القاسي أو الحار
- نظف أسنانك بشكل طبيعي مع تجنب مكان الخلع
- التزم بالأدوية إذا تم وصفها
- تجنب التدخين والجهد لمدة 24 ساعة

REFERENCE MATERIAL:
{context}
"""


def answer_from_chunks(q: str, chunks, lang: str):
    context = "\n\n".join(c["text"] for c in chunks)
    system = build_system_prompt(context, lang)

    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": q},
        ],
        temperature=0,
        max_tokens=MAX_ANSWER_TOKENS,
    )

    return (r.choices[0].message.content or "").strip()


def generate_answer(q: str, history=None):
    q = (q or "").strip()
    log.info(f"QUESTION: {q}")

    ar = is_ar(q)
    lang = "arabic" if ar else "english"

    if is_greeting(q):
        return {
            "answer": "كيف أقدر أساعدك؟" if ar else "How can I help you?",
            "refs": [],
            "source": "model",
        }

    base_query = translate_to_english(q) if ar else q

    if should_rewrite(base_query):
        clean_query = rewrite_query(base_query)
    else:
        clean_query = base_query

    chunks = retrieve_chunks(clean_query)

    if not is_relevant(clean_query, chunks):
        return {
            "answer": "أقدر أساعد فقط في أسئلة صحة الفم والأسنان" if ar else "I can only help with oral health related questions.",
            "refs": [],
            "source": "model",
        }

    answer = answer_from_chunks(q, chunks, lang)
    log.info(f"ANSWER: {answer}")

    refs = list({c["title"] for c in chunks if c["title"]})[:3]

    return {
        "answer": answer,
        "refs": refs,
        "source": "rag",
    }
