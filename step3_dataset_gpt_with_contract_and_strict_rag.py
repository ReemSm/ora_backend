import os
import re
import math
import logging
from typing import List, Dict, Any, Tuple

from openai import OpenAI
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="[ORA %(levelname)s] %(message)s")
log = logging.getLogger("ora")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-large"
PINECONE_INDEX = "oraapp777"

TOP_K_RAW = 12
TOP_K_FINAL = 5

MIN_RELEVANCE = 0.35
MIN_AUTHORITY = 0.20
MIN_CONTEXT_TOP_SCORE = 0.52
MIN_CONTEXT_AVG_SCORE = 0.46
MIN_CONTEXT_TOTAL_CHARS = 220

MAX_REWRITE_TOKENS = 80
MAX_ANSWER_TOKENS = 260
TEMPERATURE = 0.0

PINECONE_CHUNK_FIELD = "chunk_text"
PINECONE_TITLE_FIELD = "title"
PINECONE_AUTHORITY_FIELD = "authority_score"

CHUNK_FALLBACK_FIELDS = ("text", "content", "chunk", "body", "passage", "page_content")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX)


# ─────────────────────────────────────────────────────────────
# LANGUAGE / SCOPE / SAFETY
# ─────────────────────────────────────────────────────────────
def is_ar(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


NON_DENTAL_BLOCK = [
    "capital of", "weather forecast", "stock price", "bitcoin", "movie review",
    "football score", "basketball game", "president of", "election results",
    "flight booking", "hotel booking", "travel itinerary", "real estate investment",
    "car review", "vehicle specs", "iphone review", "laptop specs",
    "software development", "coding tutorial", "recipe for", "investment advice",
    "عاصمة دولة", "توقعات الطقس", "أسعار الأسهم", "بيتكوين", "أفضل فيلم",
    "نتيجة مباراة", "رئيس الوزراء", "نتائج انتخابات", "حجز فندق", "تذكرة طيران",
    "استثمار عقاري", "مواصفات سيارة", "برمجة تطبيقات", "وصفة طبخ",
]

DENTAL_SIGNALS = [
    "tooth", "teeth", "gum", "gums", "mouth", "oral", "dental", "dentist",
    "jaw", "bite", "biting", "cavity", "filling", "crown", "implant",
    "root canal", "braces", "plaque", "enamel", "dentin", "pulp",
    "extraction", "wisdom tooth", "molar", "veneer", "whitening", "floss",
    "gingivitis", "periodontitis", "pulpitis", "caries", "abscess",
    "pain", "ache", "bleeding", "swelling", "sensitivity", "numbness",
    "سن", "أسنان", "ضرس", "لثة", "فم", "فك", "حشوة", "تاج", "زرعة",
    "علاج العصب", "تقويم", "تبييض", "خيط الأسنان", "التهاب اللثة", "تسوس",
    "خلع", "خراج", "ألم", "نزيف", "تورم", "حساسية", "خدر",
]

PRESCRIPTION_PATTERNS = [
    r"prescribe\s+(me\s+)?(a\s+)?medication",
    r"write\s+(me\s+)?a\s+prescription",
    r"give\s+me\s+a\s+specific\s+(prescription|treatment\s+plan)",
    r"make\s+(me\s+)?a\s+(treatment|care)\s+plan\s+for\s+(me|my\s+case)",
    r"tell\s+me\s+exactly\s+what\s+(drug|medication|antibiotic)\s+to\s+take",
    r"diagnose\s+me\s+exactly",
    r"وصّف\s+لي\s+دواء",
    r"اعطني\s+وصفة",
    r"أعطني\s+وصفة",
    r"اكتب\s+لي\s+خطة\s+علاج",
    r"شخّص\s+حالتي\s+بالضبط",
]

SOCIAL_EN = [
    r"^(hi|hello|hey|good\s*(morning|afternoon|evening|day))[\s!.,?]*$",
    r"^(thanks|thank\s*you|thx|tysm)[\s!.,?]*$",
    r"^(bye|goodbye|see\s*you|take\s*care)[\s!.,?]*$",
]

SOCIAL_AR = [
    r"^(مرحبا|أهلاً|أهلا|هلا|السلام\s*عليكم|صباح\s*الخير|مساء\s*الخير)[\s!.,؟]*$",
    r"^(شكرًا|شكراً|شكرا|ممنون|مشكور|يسلموا|يعطيك\s*العافية)[\s!.,؟]*$",
    r"^(مع\s*السلامة|وداعاً|باي)[\s!.,؟]*$",
]


def is_treatment_request(q: str) -> bool:
    ql = (q or "").lower()
    return any(re.search(p, ql) for p in PRESCRIPTION_PATTERNS)


def has_dental_signal(text: str) -> bool:
    tl = (text or "").lower()
    return any(sig.lower() in tl for sig in DENTAL_SIGNALS)


def is_out_of_scope(q: str) -> bool:
    ql = (q or "").lower()
    if has_dental_signal(ql):
        return False
    return any(term in ql for term in NON_DENTAL_BLOCK)


def is_social_exchange(q: str) -> bool:
    ql = (q or "").lower().strip()
    qa = (q or "").strip()
    return any(re.search(p, ql, re.IGNORECASE) for p in SOCIAL_EN) or any(re.search(p, qa) for p in SOCIAL_AR)


def social_response(q: str) -> str:
    if is_ar(q):
        if re.search(r"شكر|ممنون|مشكور|يسلموا|يعطيك", q):
            return "على الرحب والسعة."
        if re.search(r"السلام\s*عليكم", q):
            return "وعليكم السلام."
        if re.search(r"مع\s*السلامة|وداعاً|باي", q):
            return "مع السلامة."
        return "أهلاً."
    ql = (q or "").lower().strip()
    if re.search(r"thanks|thank\s*you|thx", ql):
        return "You're welcome."
    if re.search(r"bye|goodbye|see\s*you|take\s*care", ql):
        return "Take care."
    return "Hello."


def refusal_treatment(q: str) -> str:
    return (
        "ما أقدر أوصف أدوية أو أقدم تشخيصاً مخصصاً. يُنصح بمراجعة طبيب أسنان مرخّص."
        if is_ar(q)
        else "I can't prescribe medication or provide a personalised diagnosis. Please consult a licensed dentist."
    )


def refusal_scope(q: str) -> str:
    return (
        "هذا السؤال خارج نطاق تطبيق صحة الفم والأسنان."
        if is_ar(q)
        else "This is outside the scope of this oral health application."
    )


def insufficient_info(q: str) -> str:
    return (
        "المعلومات المتاحة لدي لا تكفي للإجابة بشكل موثوق على هذا السؤال."
        if is_ar(q)
        else "I don't have enough grounded information to answer this reliably."
    )


# ─────────────────────────────────────────────────────────────
# QUERY TYPE
# ─────────────────────────────────────────────────────────────
def detect_question_type(q: str) -> str:
    ql = (q or "").lower().strip()

    instruction_patterns = [
        "how to", "how do i", "what should i do", "what should i avoid",
        "aftercare", "post-op", "postoperative", "after extraction",
        "after root canal", "after implant", "after filling", "after treatment",
        "can i eat", "can i drink", "when can i eat", "when can i drink",
        "كيف أعتني", "ماذا أفعل بعد", "بعد الخلع", "بعد الزرعة", "بعد الحشوة",
        "هل أقدر آكل", "هل أقدر أشرب", "تعليمات بعد", "العناية بعد",
    ]
    for pattern in instruction_patterns:
        if pattern in ql:
            return "instruction"

    return "informational"


# ─────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────
def embed(text: str) -> List[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ─────────────────────────────────────────────────────────────
# GPT REWRITE
# ─────────────────────────────────────────────────────────────
def rewrite_query_for_retrieval(q: str) -> str:
    ar = is_ar(q)

    system = (
        "You rewrite patient dental questions into a short retrieval query for textbook-style dental sources.\n"
        "Rules:\n"
        "• Output one line only.\n"
        "• Preserve the user's language.\n"
        "• Expand into likely clinical dental search terms.\n"
        "• Do not answer the question.\n"
        "• Do not add formatting, bullets, explanations, or quotation marks.\n"
        "• Keep it under 25 words.\n"
    )

    if ar:
        system += (
            "• استخدم مصطلحات سنية سريرية عربية مفهومة للبحث.\n"
            "• أضف مرادفات سريرية محتملة عند الحاجة.\n"
        )

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": q},
            ],
            max_tokens=MAX_REWRITE_TOKENS,
            temperature=0.0,
        )
        rewritten = (r.choices[0].message.content or "").strip()
        return rewritten if rewritten else q
    except Exception as e:
        log.error(f"Rewrite error: {e}")
        return q


# ─────────────────────────────────────────────────────────────
# RAG
# ─────────────────────────────────────────────────────────────
def extract_text(md: Dict[str, Any]) -> str:
    primary = md.get(PINECONE_CHUNK_FIELD)
    if primary:
        text = str(primary).strip()
        if text:
            return text

    for field in CHUNK_FALLBACK_FIELDS:
        val = md.get(field)
        if val:
            text = str(val).strip()
            if text:
                return text

    for _, val in md.items():
        if isinstance(val, str) and len(val.strip()) > 40:
            return val.strip()

    return ""


def retrieve_chunks(query_vector: List[float]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    debug = {
        "match_count": 0,
        "top_score": 0.0,
        "avg_score": 0.0,
        "total_chars": 0,
        "context_sufficient": False,
    }

    try:
        res = index.query(vector=query_vector, top_k=TOP_K_RAW, include_metadata=True)
        matches = res.get("matches", [])
    except Exception as e:
        log.error(f"Pinecone query failed: {e}")
        return [], debug

    debug["match_count"] = len(matches)
    if not matches:
        return [], debug

    debug["top_score"] = round(float(matches[0].get("score", 0.0)), 4)
    if float(matches[0].get("score", 0.0)) < MIN_RELEVANCE:
        return [], debug

    accepted: List[Dict[str, Any]] = []
    for match in matches:
        score = float(match.get("score", 0.0))
        md = match.get("metadata") or {}

        raw_auth = md.get(PINECONE_AUTHORITY_FIELD)
        if raw_auth is not None:
            try:
                if float(raw_auth) < MIN_AUTHORITY:
                    continue
            except (TypeError, ValueError):
                pass

        text = extract_text(md)
        if not text:
            continue

        accepted.append(
            {
                "title": str(md.get(PINECONE_TITLE_FIELD) or ""),
                "text": text,
                "score": score,
            }
        )

    if not accepted:
        return [], debug

    unique: List[Dict[str, Any]] = []
    seen = set()
    for chunk in accepted:
        key = (chunk["title"] or chunk["text"][:180]).strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    final_chunks = unique[:TOP_K_FINAL]

    avg_score = sum(c["score"] for c in final_chunks) / len(final_chunks)
    total_chars = sum(len(c["text"]) for c in final_chunks)

    debug["avg_score"] = round(avg_score, 4)
    debug["total_chars"] = total_chars
    debug["context_sufficient"] = (
        debug["top_score"] >= MIN_CONTEXT_TOP_SCORE
        and avg_score >= MIN_CONTEXT_AVG_SCORE
        and total_chars >= MIN_CONTEXT_TOTAL_CHARS
    )

    return final_chunks, debug


# ─────────────────────────────────────────────────────────────
# GPT ANSWER FROM RAG ONLY
# ─────────────────────────────────────────────────────────────
def build_format_rules(ar: bool, qtype: str) -> str:
    if ar:
        if qtype == "instruction":
            return (
                "FORMAT:\n"
                "• استخدم الرمز • فقط.\n"
                "• من 3 إلى 5 نقاط كحد أقصى.\n"
                "• كل نقطة جملة واحدة واضحة ومباشرة.\n"
            )
        return (
            "FORMAT:\n"
            "• نثر مباشر فقط، بدون نقاط.\n"
            "• من 2 إلى 3 جمل كحد أقصى.\n"
        )

    if qtype == "instruction":
        return (
            "FORMAT:\n"
            "• Use the • symbol only.\n"
            "• 3 to 5 bullet points maximum.\n"
            "• Each bullet must be one clear, actionable sentence.\n"
        )

    return (
        "FORMAT:\n"
        "• Use plain prose only, with no bullet points.\n"
        "• 2 to 3 sentences maximum.\n"
    )


def answer_from_chunks(q: str, chunks: List[Dict[str, Any]], history: List[Dict[str, str]] | None = None) -> str:
    ar = is_ar(q)
    qtype = detect_question_type(q)

    context_text = "\n---\n".join(chunk["text"] for chunk in chunks)

    if ar:
        system = (
            "أنت مساعد صحة فم وأسنان للمرضى.\n"
            "مهمتك شرح المعلومة بلغة بسيطة وإنسانية وغير أكاديمية.\n"
            "استخدم النص المرجعي فقط.\n"
            "ممنوع إضافة أي معلومة غير مدعومة من النص.\n"
            "إذا كان النص غير كافٍ، اكتب فقط:\n"
            "المعلومات المتاحة لدي لا تكفي للإجابة بشكل موثوق على هذا السؤال.\n"
            "لا تذكر المصادر أو الكتب أو الاسترجاع.\n"
            "لا تسأل أسئلة متابعة.\n"
            "لا تكتب كأستاذ أو كتاب دراسي.\n"
            f"{build_format_rules(ar, qtype)}\n"
            "REFERENCE MATERIAL:\n"
            f"{context_text}"
        )
    else:
        system = (
            "You are an oral health assistant for patients.\n"
            "Your job is to explain in plain, calm, human language, not textbook language.\n"
            "Use only the reference material below.\n"
            "Do not add any fact not supported by it.\n"
            "If the material is not enough, output only:\n"
            "I don't have enough grounded information to answer this reliably.\n"
            "Do not mention sources, retrieval, or documents.\n"
            "Do not ask follow-up questions.\n"
            "Do not sound like a professor or textbook.\n"
            f"{build_format_rules(ar, qtype)}\n"
            "REFERENCE MATERIAL:\n"
            f"{context_text}"
        )

    messages = [{"role": "system", "content": system}]

    if history:
        for turn in history[-4:]:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": q})

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_ANSWER_TOKENS,
            temperature=TEMPERATURE,
        )
        text = (r.choices[0].message.content or "").strip()
        return text if text else insufficient_info(q)
    except Exception as e:
        log.error(f"Answer generation error: {e}")
        return insufficient_info(q)


# ─────────────────────────────────────────────────────────────
# PUBLIC PIPELINE
# ─────────────────────────────────────────────────────────────
def generate_answer(q: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
    q = (q or "").strip()

    if not q:
        return {"answer": "Empty query.", "refs": [], "source": "error", "debug": {"error": "empty_query"}}

    if is_treatment_request(q):
        return {"answer": refusal_treatment(q), "refs": [], "source": "safety_refusal", "debug": {}}

    if is_social_exchange(q):
        return {"answer": social_response(q), "refs": [], "source": "social", "debug": {}}

    if is_out_of_scope(q):
        return {"answer": refusal_scope(q), "refs": [], "source": "scope_refusal", "debug": {}}

    rewritten_query = rewrite_query_for_retrieval(q)
    query_vector = embed(rewritten_query)
    chunks, debug = retrieve_chunks(query_vector)

    debug["rewritten_query"] = rewritten_query

    if not chunks or not debug["context_sufficient"]:
        return {
            "answer": insufficient_info(q),
            "refs": [],
            "source": "insufficient_grounded_context",
            "debug": debug,
        }

    answer = answer_from_chunks(q, chunks, history=history)

    refs = [c["title"] for c in chunks if c.get("title")]
    return {
        "answer": answer,
        "refs": refs,
        "source": "rag",
        "debug": debug,
    }

