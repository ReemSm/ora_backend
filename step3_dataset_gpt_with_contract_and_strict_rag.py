import os
import re
import math
import json
import logging
from openai import OpenAI
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="[ORA %(levelname)s] %(message)s")
log = logging.getLogger("ora")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-large"
PINECONE_INDEX = "oraapp777"

SIM_THRESHOLD = 0.70
MIN_RELEVANCE = 0.28
MIN_AUTHORITY = 0.40
TOP_K_RAW = 6
TOP_K_FINAL = 3
MAX_GPT_TOKENS = 420
MIN_REF_DISPLAY = 0.70

PINECONE_CHUNK_FIELD = "chunk_text"
PINECONE_TITLE_FIELD = "title"
PINECONE_SOURCE_FIELD = "source_type"
PINECONE_AUTHORITY_FIELD = "authority_score"
PINECONE_PATH_FIELD = "source_path"

_CHUNK_FALLBACK_FIELDS = ("text", "content", "chunk", "body", "passage", "page_content")


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX)


# ─────────────────────────────────────────────────────────────────────────────
# STATIC DATASET
# ─────────────────────────────────────────────────────────────────────────────
DATASET = [
    {
        "field": ["Periodontics"],
        "en_q": "My gums bleed when I brush, what should I do?",
        "en_a": (
            "Bleeding gums usually mean the gum tissue is inflamed from plaque. "
            "Brush twice daily correctly and floss daily. "
            "Professional cleaning helps prevent progression."
        ),
        "ar_q": "نزيف اللثة",
        "ar_a": (
            "نزيف اللثة غالباً بسبب تراكم البلاك. "
            "يجب تنظيف الأسنان بانتظام واستخدام الخيط، مع مراجعة الطبيب للتنظيف."
        ),
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE
# ─────────────────────────────────────────────────────────────────────────────
def is_ar(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING + CACHE (FIXED)
# ─────────────────────────────────────────────────────────────────────────────
_EMBED_CACHE_FILE = "dataset_embeddings_cache.json"


def embed(text: str) -> list:
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _load_or_build_dataset_embeddings():
    if os.path.exists(_EMBED_CACHE_FILE):
        log.info("Loading dataset embeddings from cache...")
        with open(_EMBED_CACHE_FILE, "r") as f:
            data = json.load(f)
            return data["en"], data["ar"]

    log.info("Computing dataset embeddings...")
    en_vecs = [embed(d["en_q"]) for d in DATASET]
    ar_vecs = [embed(d["ar_q"]) for d in DATASET]

    with open(_EMBED_CACHE_FILE, "w") as f:
        json.dump({"en": en_vecs, "ar": ar_vecs}, f)

    log.info("Dataset embeddings cached.")
    return en_vecs, ar_vecs


_DS_EN_VECS, _DS_AR_VECS = _load_or_build_dataset_embeddings()


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC SCOPE (FIXED)
# ─────────────────────────────────────────────────────────────────────────────
_DENTAL_INTENT_ANCHORS = [
    "tooth pain",
    "gum bleeding",
    "dental treatment",
    "oral health problem",
    "toothache",
    "swollen gums",
]

_NON_DENTAL_INTENT_ANCHORS = [
    "weather forecast",
    "stock market price",
    "movie review",
    "travel booking",
    "car specifications",
    "programming tutorial",
]

_DENTAL_ANCHOR_VECS = [embed(x) for x in _DENTAL_INTENT_ANCHORS]
_NON_DENTAL_ANCHOR_VECS = [embed(x) for x in _NON_DENTAL_INTENT_ANCHORS]


def semantic_scope_score(qv):
    dental_score = max(cosine(qv, v) for v in _DENTAL_ANCHOR_VECS)
    non_dental_score = max(cosine(qv, v) for v in _NON_DENTAL_ANCHOR_VECS)
    return dental_score, non_dental_score


# ─────────────────────────────────────────────────────────────────────────────
# GPT (LAYERED PROMPT)
# ─────────────────────────────────────────────────────────────────────────────
def gpt_style_answer(q: str, context_chunks=None, history=None) -> str:
    ar = is_ar(q)

    ROLE_LAYER = "You are ORA, a dental health assistant."
    SCOPE_LAYER = "Only answer dental-related questions."
    BEHAVIOR_LAYER = "Be concise, clinically useful, and avoid unnecessary detail."
    FORMAT_LAYER = "2–4 sentences only."
    LANGUAGE_LAYER = "Respond in Arabic." if ar else "Respond in English."
    CONSTRAINT_LAYER = "Do not use lists, headings, or extra commentary."

    CONTEXT_LAYER = "\n".join(context_chunks) if context_chunks else ""

    system = "\n\n".join([
        ROLE_LAYER,
        SCOPE_LAYER,
        BEHAVIOR_LAYER,
        FORMAT_LAYER,
        LANGUAGE_LAYER,
        CONSTRAINT_LAYER,
        CONTEXT_LAYER
    ])

    msgs = [{"role": "system", "content": system}]
    msgs.append({"role": "user", "content": q})

    r = client.chat.completions.create(
        model=MODEL,
        messages=msgs,
        max_tokens=MAX_GPT_TOKENS,
        temperature=0.2,
    )

    return r.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def generate_answer(q: str, history=None):
    qv = embed(q)

    dental_score, non_dental_score = semantic_scope_score(qv)

    is_semantically_dental = dental_score >= 0.55
    is_semantically_non_dental = non_dental_score > dental_score

    if is_semantically_non_dental and not is_semantically_dental:
        return {
            "answer": "This is outside the scope of this oral health application.",
            "refs": [],
            "source": "scope_refusal",
            "debug": {
                "dental_score": dental_score,
                "non_dental_score": non_dental_score,
            },
        }

    answer = gpt_style_answer(q)

    return {
        "answer": answer,
        "refs": [],
        "source": "gpt",
        "debug": {},
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    history = []

    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        result = generate_answer(q, history=history)
        print(result["answer"])

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result["answer"]})
