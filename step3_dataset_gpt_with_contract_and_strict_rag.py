import os
import re
import logging
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="[ORA %(levelname)s] %(message)s")
log = logging.getLogger("ora")

MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-large"
PINECONE_INDEX = "oraapp777"

TOP_K_RAW = 12
TOP_K_FINAL = 5

MIN_TOP_SCORE = 0.52
MIN_AVG_SCORE = 0.46
MIN_TOTAL_CHARS = 220

MAX_REWRITE_TOKENS = 80
MAX_ANSWER_TOKENS = 260
TEMPERATURE = 0.0

PINECONE_CHUNK_FIELD = "chunk_text"
PINECONE_TITLE_FIELD = "title"
PINECONE_AUTHORITY_FIELD = "authority_score"

HISTORY_CHAR_BUDGET = 2200
MAX_REFERENCE_TITLES = 3
MIN_AUTHORITY = 0.20

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX)

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

DENTAL_SIGNALS = [
    "tooth", "teeth", "gum", "gums", "mouth", "oral", "dental", "dentist",
    "jaw", "bite", "biting", "cavity", "filling", "crown", "implant",
    "root canal", "braces", "plaque", "enamel", "dentin", "pulp",
    "extraction", "wisdom tooth", "molar", "veneer", "whitening", "floss",
    "gingivitis", "periodontitis", "pulpitis", "caries", "abscess",
    "pain", "ache", "bleeding", "swelling", "sensitivity", "numbness",
    "toothache", "ulcer", "mouth sore", "bad breath", "retainer", "aligner",
    "سن", "أسنان", "ضرس", "لثة", "فم", "فك", "حشوة", "تاج", "زرعة",
    "علاج العصب", "تقويم", "تبييض", "خيط الأسنان", "التهاب اللثة", "تسوس",
    "خلع", "خراج", "ألم", "نزيف", "تورم", "حساسية", "خدر", "رائحة الفم",
    "قرحة", "تقويم شفاف", "مثبت",
]

NON_DENTAL_SIGNALS = [
    "capital of", "weather forecast", "stock price", "bitcoin", "movie review",
    "football score", "basketball game", "president of", "election results",
    "flight booking", "hotel booking", "travel itinerary", "real estate investment",
    "car review", "vehicle specs", "iphone review", "laptop specs",
    "software development", "coding tutorial", "recipe for", "investment advice",
    "عاصمة دولة", "توقعات الطقس", "أسعار الأسهم", "بيتكوين", "أفضل فيلم",
    "نتيجة مباراة", "رئيس الوزراء", "نتائج انتخابات", "حجز فندق", "تذكرة طيران",
    "استثمار عقاري", "مواصفات سيارة", "برمجة تطبيقات", "وصفة طبخ",
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


def is_ar(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))


def has_any_signal(text: str, signals: List[str]) -> bool:
    tl = (text or "").lower()
    return any(sig.lower() in tl for sig in signals)


def is_treatment_request(q: str) -> bool:
    ql = (q or "").lower()
    return any(re.search(p, ql) for p in PRESCRIPTION_PATTERNS)


def is_social_exchange(q: str) -> bool:
    ql = (q or "").lower().strip()
    qa = (q or "").strip()
    return any(re.search(p, ql, re.IGNORECASE) for p in SOCIAL_EN) or any(re.search(p, qa) for p in SOCIAL_AR)


def social_response(ar: bool, q: str) -> str:
    if ar:
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


def refusal_treatment(ar: bool) -> str:
    if ar:
        return "ما أقدر أوصف أدوية أو أقدم تشخيصاً مخصصاً. يُنصح بمراجعة طبيب أسنان مرخّص."
    return "I can't prescribe medication or provide a personalised diagnosis. Please consult a licensed dentist."


def refusal_scope(ar: bool) -> str:
    if ar:
        return "هذا السؤال خارج نطاق تطبيق صحة الفم والأسنان."
    return "This is outside the scope of this oral health application."


def insufficient_info(ar: bool) -> str:
    if ar:
        return "المعلومات المتاحة لدي لا تكفي للإجابة بشكل موثوق على هذا السؤال."
    return "I don't have enough grounded information to answer this reliably."


def looks_explicitly_non_dental(q: str) -> bool:
    ql = (q or "").lower()
    return has_any_signal(ql, NON_DENTAL_SIGNALS) and not has_any_signal(ql, DENTAL_SIGNALS)


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


def embed(text: str) -> List[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def translate_to_english(q: str) -> str:
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Translate the following Arabic dental query into clear, natural English for medical retrieval. Output only the translation.",
                },
                {"role": "user", "content": q},
            ],
            max_tokens=80,
            temperature=0.0,
        )
        return (r.choices[0].message.content or "").strip() or q
    except Exception as e:
        log.error(f"Translation error: {e}")
        return q


def rewrite_query_for_retrieval(q: str) -> str:
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite this into a short dental textbook retrieval query.\n"
                        "One line only.\n"
                        "No explanations.\n"
                        "Under 25 words."
                    ),
                },
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


def extract_text(md: Dict[str, Any]) -> str:
    val = md.get(PINECONE_CHUNK_FIELD)
    return str(val).strip() if val else ""


def normalize_authority_score(md: Dict[str, Any]) -> Optional[float]:
    raw_auth = md.get(PINECONE_AUTHORITY_FIELD)
    if raw_auth is None:
        return None
    try:
        return float(raw_auth)
    except Exception:
        log.warning(f"Invalid authority_score: {raw_auth}")
        return 0.0


def query_pinecone(vector: List[float]) -> List[Dict[str, Any]]:
    try:
        res = index.query(vector=vector, top_k=TOP_K_RAW, include_metadata=True)
        return res.get("matches", [])
    except Exception as e:
        log.error(f"Pinecone error: {e}")
        return []


def merge_and_rank(matches_list: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    all_chunks = []

    for matches in matches_list:
        for m in matches:
            md = m.get("metadata") or {}
            text = extract_text(md)
            if not text:
                continue

            authority = normalize_authority_score(md)
            if authority is not None and authority < MIN_AUTHORITY:
                continue

            all_chunks.append({
                "title": str(md.get(PINECONE_TITLE_FIELD) or "").strip(),
                "text": text,
                "score": float(m.get("score", 0.0)),
            })

    all_chunks.sort(key=lambda x: x["score"], reverse=True)

    seen = set()
    unique = []
    for c in all_chunks:
        key = c["text"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(c)
        if len(unique) >= TOP_K_FINAL:
            break

    return unique


def retrieve_chunks(queries: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    debug = {
        "top_score": 0,
        "avg_score": 0,
        "total_chars": 0,
        "context_sufficient": False,
    }

    matches_list = []
    for q in queries:
        matches_list.append(query_pinecone(embed(q)))

    chunks = merge_and_rank(matches_list)
    if not chunks:
        return [], debug

    top = chunks[0]["score"]
    avg = sum(c["score"] for c in chunks) / len(chunks)
    total = sum(len(c["text"]) for c in chunks)

    debug.update({
        "top_score": top,
        "avg_score": avg,
        "total_chars": total,
        "context_sufficient": top >= MIN_TOP_SCORE and (avg >= MIN_AVG_SCORE or total >= MIN_TOTAL_CHARS)
    })

    return chunks, debug


def build_history_messages(history):
    if not history:
        return []

    out = []
    size = 0
    for turn in reversed(history):
        content = (turn.get("content") or "").strip()
        role = turn.get("role")
        if role not in ("user", "assistant") or not content:
            continue
        if size + len(content) > HISTORY_CHAR_BUDGET:
            break
        out.append({"role": role, "content": content})
        size += len(content)

    return list(reversed(out))


def build_system_prompt(ar: bool, qtype: str, context: str) -> str:
    if ar:
        if qtype == "instruction":
            return (
                "أنت مساعد صحة فم وأسنان.\n"
                "استخدم فقط النص المرجعي.\n"
                "استخدم أسلوب عربي طبيعي بسيط مثل شرح الطبيب للمريض.\n"
                "لا تستخدم ترجمة حرفية.\n"
                "لا تستخدم لغة أكاديمية أو رسمية ثقيلة.\n"
                "لا تكتب أي مقدمة.\n"
                "ابدأ مباشرة بالنقاط.\n"
                "استخدم الرمز • فقط.\n"
                "من 4 إلى 9 نقاط.\n"
                "كل نقطة جملة قصيرة مباشرة.\n"
                "رتب النقاط حسب الأولوية بعد الإجراء مباشرة.\n"
                "\n"
                "مثال يجب اتباعه في الأسلوب:\n"
                "• اضغط على قطعة الشاش أول 30 دقيقة بعد الإجراء\n"
                "• استخدام الكمادات الباردة على منطقة الخلع لمدة أول 30 دقيقة بعد الإجراء\n"
                "• لا تبصق ولا تحرك الماء داخل الفم لمدة 24 ساعة\n"
                "• تجنب استخدام المضمضة لمدة 24 ساعة (بما في ذلك المضمضة وقت الوضوء)\n"
                "• لا تستخدم الشفاط أو المصاص وقت الشرب لمدة 24 ساعة\n"
                "• تجنب الأكل القاسي أو الساخن\n"
                "• نظف أسنانك بشكل طبيعي مع تجنب منطقة الإجراء\n"
                "• التزم بالأدوية الموصوفة إن وجدت\n"
                "• تجنب التدخين والجهد البدني لمدة 24 ساعة\n"
                "• إزالة الغرز تكون حسب تعليمات الطبيب\n"
                "\n"
                "إذا لم تكفِ المعلومات اكتب فقط:\n"
                "المعلومات المتاحة لدي لا تكفي للإجابة بشكل موثوق على هذا السؤال.\n"
                "\n"
                "REFERENCE MATERIAL:\n"
                f"{context}"
            )

        return (
            "أنت مساعد صحة فم وأسنان.\n"
            "استخدم فقط النص المرجعي.\n"
            "استخدم أسلوب عربي طبيعي بسيط مثل شرح الطبيب للمريض.\n"
            "يمكنك ذكر الأسباب الشائعة المعروفة في طب الأسنان إذا كانت أساسية ومقبولة.\n"
            "لا تستخدم ترجمة حرفية.\n"
            "لا تستخدم لغة أكاديمية أو رسمية ثقيلة.\n"
            "لا تكتب أي مقدمة.\n"
            "ابدأ مباشرة بالمعلومة.\n"
            "من جملتين إلى ثلاث جمل كحد أقصى.\n"
            "\n"
            "مثال يجب اتباعه في الأسلوب:\n"
            "من الأسباب الشائعة لألم الأسنان التسوس، التهاب العصب، أو التهاب في اللثة. "
            "أحياناً يكون الألم من سن آخر أو من الجيوب الأنفية. "
            "إذا استمر الألم أو زاد ننصحك بزيارة طبيب أسنان مرخص.\n"
            "\n"
            "إذا لم تكفِ المعلومات اكتب فقط:\n"
            "المعلومات المتاحة لدي لا تكفي للإجابة بشكل موثوق على هذا السؤال.\n"
            "\n"
            "REFERENCE MATERIAL:\n"
            f"{context}"
        )

    if qtype == "instruction":
        return (
            "You are an oral health assistant.\n"
            "Use only the reference material.\n"
            "Do not use academic or textbook tone.\n"
            "Do not write any introduction.\n"
            "Start directly with bullet points.\n"
            "Use the • symbol only.\n"
            "4 to 9 bullet points.\n"
            "Each bullet is one short direct action.\n"
            "Order steps by immediate priority after the procedure.\n"
            "\n"
            "Example style:\n"
            "• Bite on gauze for 30 minutes\n"
            "• Use a cold compress after that\n"
            "• Do not spit or move water in your mouth for 24 hours\n"
            "• Do not use a straw for 24 hours\n"
            "• Avoid hot or solid food\n"
            "• Clean your teeth normally but avoid the procedure site\n"
            "• Follow prescribed medication if given\n"
            "• Avoid smoking and physical activity for 24 hours\n"
            "• Remove sutures as instructed by your dentist\n"
            "\n"
            "If the material is not enough, output only:\n"
            "I don't have enough grounded information to answer this reliably.\n"
            "\n"
            "REFERENCE MATERIAL:\n"
            f"{context}"
        )

    return (
        "You are an oral health assistant.\n"
        "Use only the reference material.\n"
        "You may include common basic dental causes if they are standard and widely accepted.\n"
        "Do not use academic or textbook tone.\n"
        "Do not write any introduction.\n"
        "Start directly with useful information.\n"
        "2 to 3 sentences maximum.\n"
        "\n"
        "Example style:\n"
        "Common reasons for tooth pain are cavities, pulp inflammation (the nerve of the tooth), or gum inflammation. "
        "Sometimes the pain is referred from another tooth or from areas like sinusitis. "
        "If it continues or gets worse, a dental checkup with a licensed dentist is recommended.\n"
        "\n"
        "If the material is not enough, output only:\n"
        "I don't have enough grounded information to answer this reliably.\n"
        "\n"
        "REFERENCE MATERIAL:\n"
        f"{context}"
    )


def answer_from_chunks(q, ar, chunks, history=None):
    context = "\n\n".join(c["text"] for c in chunks)
    system = build_system_prompt(ar, detect_question_type(q), context)

    messages = [{"role": "system", "content": system}]
    messages += build_history_messages(history)
    messages.append({"role": "user", "content": q})

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_ANSWER_TOKENS,
            temperature=TEMPERATURE,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return insufficient_info(ar)


def generate_answer(q: str, history=None):
    q = (q or "").strip()
    ar = is_ar(q)

    if not q:
        return {"answer": "Empty query.", "refs": [], "source": "error", "debug": {}}

    if is_treatment_request(q):
        return {"answer": refusal_treatment(ar), "refs": [], "source": "safety_refusal", "debug": {}}

    if is_social_exchange(q):
        return {"answer": social_response(ar, q), "refs": [], "source": "social", "debug": {}}

    base_query = q
    if ar:
        base_query = translate_to_english(q)

    rewritten = rewrite_query_for_retrieval(base_query)

    queries = [base_query]
    if rewritten and rewritten != base_query:
        queries.append(rewritten)

    chunks, debug = retrieve_chunks(queries)

    if not chunks or not debug["context_sufficient"]:
        source = "insufficient"
        answer = insufficient_info(ar)

        if looks_explicitly_non_dental(q):
            source = "scope_refusal"
            answer = refusal_scope(ar)

        return {"answer": answer, "refs": [], "source": source, "debug": debug}

    answer = answer_from_chunks(q, ar, chunks, history)

    seen_titles = set()
    refs = []
    for c in chunks:
        title = (c.get("title") or "").strip()
        if title and title not in seen_titles:
            seen_titles.add(title)
            refs.append(title)
        if len(refs) >= MAX_REFERENCE_TITLES:
            break

    return {"answer": answer, "refs": refs, "source": "rag", "debug": debug}
