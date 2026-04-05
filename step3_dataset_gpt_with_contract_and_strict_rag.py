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

# Retrieval / grounding
TOP_K_RAW = 12
TOP_K_FINAL = 5
MIN_RELEVANCE = 0.38
MIN_AUTHORITY = 0.25
MIN_REF_DISPLAY = 0.70
SIM_THRESHOLD = 0.84
SEMANTIC_DENTAL_THRESHOLD = 0.56

# Context sufficiency
MIN_CONTEXT_TOTAL_CHARS = 220
MIN_CONTEXT_AVG_SCORE = 0.50
MIN_CONTEXT_TOP_SCORE = 0.55

# Generation
MAX_GPT_TOKENS = 260
TEMPERATURE = 0.0

# Pinecone metadata field names
PINECONE_CHUNK_FIELD = "chunk_text"
PINECONE_TITLE_FIELD = "title"
PINECONE_SOURCE_FIELD = "source_type"
PINECONE_AUTHORITY_FIELD = "authority_score"
PINECONE_PATH_FIELD = "source_path"

_CHUNK_FALLBACK_FIELDS = ("text", "content", "chunk", "body", "passage", "page_content")

# Local caches
_DATASET_EMBED_CACHE_FILE = "dataset_embeddings_cache.json"
_SCOPE_EMBED_CACHE_FILE = "scope_embeddings_cache.json"


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────────────────────
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX)


# ─────────────────────────────────────────────────────────────────────────────
# STATIC DATASET
# Used only as a controlled fallback when RAG returns no usable context.
# It never overrides successful RAG retrieval.
# ─────────────────────────────────────────────────────────────────────────────
DATASET = [
    {
        "field": ["Periodontics"],
        "en_q": "My gums bleed when I brush, what should I do?",
        "en_a": (
            "Bleeding gums usually mean the gum tissue is inflamed from plaque — a sticky bacterial "
            "film that builds up on teeth. Mild cases show redness and slight bleeding; advanced cases "
            "can involve swelling and recession. The fix is consistent hygiene: brush twice daily with "
            "correct technique and floss daily to reach between teeth. Plaque left undisturbed hardens "
            "into calculus, which only a dentist can remove. A professional cleaning every six months "
            "helps prevent progression."
        ),
        "ar_q": "نزيف اللثة",
        "ar_a": (
            "نزيف اللثة في الغالب علامة على التهابها بسبب تراكم البلاك — طبقة لزجة من البكتيريا "
            "تتكوّن على الأسنان. في الحالات البسيطة يظهر احمرار ونزيف خفيف، وفي الحالات المتقدمة "
            "قد يصير فيه تورم وانحسار. الحل هو الانتظام في نظافة الفم: تفريش مرتين يومياً بطريقة "
            "صحيحة، واستخدام خيط الأسنان يومياً. البلاك إذا بقي يتكلّس ويصير جيراً ما يزيله إلا "
            "التنظيف عند طبيب الأسنان. يُنصح بزيارة دورية كل ستة أشهر."
        ),
    },
    {
        "field": ["Implant", "Periodontics"],
        "en_q": "I had a dental implant and notice bluish discoloration on my gum, is that normal?",
        "en_a": (
            "Bluish discoloration near an implant is common when the surrounding gum tissue is "
            "naturally thin — the metallic components show through the tissue. This is usually an "
            "anatomical variation rather than a sign of infection or implant failure. It is mainly an "
            "aesthetic concern. A periodontist or implant specialist can assess whether soft tissue "
            "correction is appropriate."
        ),
        "ar_q": "بعد الزرعة صار عندي لون أزرق في اللثة",
        "ar_a": (
            "اللون الأزرق بالقرب من الزرعة يصير في الغالب عند من تكون لثتهم رقيقة بطبيعتها — "
            "المكونات المعدنية للزرعة تبيّن من خلال النسيج الرقيق. هذا غالباً تغيّر تشريحي طبيعي "
            "وليس علامة على فشل الزرعة أو التهاب. غالباً هو موضوع شكلي أكثر من كونه مرضي. "
            "لو كان مزعجاً، طبيب اللثة أو الزراعة يقدر يقيّم إذا كان التدخل مناسب."
        ),
    },
    {
        "field": ["Restorative Dentistry"],
        "en_q": "I had a filling and now feel pain only when biting. What does it mean?",
        "en_a": (
            "Pain only when biting after a filling usually means the restoration sits slightly too high "
            "and contacts the opposing tooth before the rest of the bite does. That creates localised "
            "pressure during chewing. This is different from general sensitivity because it is linked "
            "specifically to biting force. A simple adjustment by the dentist usually resolves it."
        ),
        "ar_q": "بعد الحشوة عندي ألم عند العض فقط",
        "ar_a": (
            "الألم عند العض فقط بعد الحشوة يعني في الغالب إن الحشوة مرتفعة شوي وتلامس السن "
            "المقابل قبل بقية الأسنان، وهذا يسبب ضغطاً موضعياً أثناء المضغ. هذا يختلف عن "
            "الحساسية العامة لأن الألم مرتبط بقوة العض تحديداً. تعديل بسيط عند طبيب الأسنان "
            "غالباً يحل المشكلة."
        ),
    },
    {
        "field": ["Endodontics"],
        "en_q": "I had severe tooth pain that disappeared on its own. What does it mean?",
        "en_a": (
            "When severe tooth pain disappears on its own, it often means the pulp — the "
            "nerve-containing tissue inside the tooth — has lost vitality. As the nerve dies, pain "
            "signals stop, but the bacterial infection can continue. This is not true recovery. "
            "An evaluation is still needed even if the pain is gone."
        ),
        "ar_q": "اختفى الألم الشديد في السن من تلقاء نفسه",
        "ar_a": (
            "اختفاء الألم من تلقاء نفسه لا يعني الشفاء. في الغالب يعني إن لب السن — نسيج العصب "
            "بداخله — فقد حيويته. لما يموت العصب تتوقف الإشارات المؤلمة، لكن العدوى البكتيرية "
            "قد تستمر. هذا ليس تحسناً حقيقياً، وتبقى المراجعة مهمة حتى لو راح الألم."
        ),
    },
    {
        "field": ["Endodontics"],
        "en_q": "What does tooth decay actually mean?",
        "en_a": (
            "Tooth decay means bacteria have damaged the tooth structure, starting from the outer layer "
            "and progressing inward. If untreated, the process can reach dentin and then pulp. The tooth "
            "does not literally rot, but the ongoing bacterial activity causes progressive structural "
            "damage. The treatment depends on how deep the decay has reached."
        ),
        "ar_q": "ما معنى تسوس أو تعفن السن؟",
        "ar_a": (
            "التسوس يعني إن البكتيريا سببت تلفاً تدريجياً في بنية السن وبدأت من الطبقات الخارجية "
            "ثم قد تمتد للداخل. إذا استمر من دون علاج قد يصل إلى الدنتين ثم اللب. السن لا يتعفن "
            "حرفياً، لكن النشاط البكتيري يسبب تآكلاً وتلفاً مستمراً. العلاج يعتمد على مدى تقدم الحالة."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# LANGUAGE / SCOPE / SAFETY
# ─────────────────────────────────────────────────────────────────────────────
def is_ar(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


_NON_DENTAL_BLOCK = [
    "capital of", "weather forecast", "stock price", "stock market",
    "bitcoin", "cryptocurrency", "crypto trading",
    "movie review", "film review", "best movie", "watch a movie",
    "music playlist", "song lyrics",
    "football score", "basketball game", "soccer match", "rugby score",
    "political party", "president of", "prime minister", "election results",
    "flight booking", "airline ticket", "hotel booking", "travel itinerary",
    "how to travel to", "best hotel in",
    "real estate investment", "property price", "buy a house",
    "porsche", "ferrari", "lamborghini", "bugatti",
    "car review", "vehicle specs", "best car model",
    "iphone review", "android phone review", "laptop specs", "best laptop",
    "software development", "coding tutorial", "how to code", "programming language",
    "recipe for", "how to cook", "how to bake", "cooking tips",
    "investment advice", "stock portfolio",
    "interior design", "penthouse",
    "عاصمة دولة", "توقعات الطقس", "أسعار الأسهم", "بيتكوين",
    "عملة رقمية", "أفضل فيلم", "كلمات أغنية",
    "نتيجة مباراة كرة", "سياسة الدولة", "رئيس الوزراء",
    "نتائج انتخابات", "حجز فندق", "تذكرة طيران",
    "استثمار عقاري", "سعر العقار", "شراء شقة",
    "بورش", "فيراري", "لامبورغيني",
    "مواصفات سيارة", "أفضل سيارة",
    "وصفة طبخ", "كيفية الطبخ", "برمجة تطبيقات",
    "استثمار في الأسهم", "شقة فارهة",
]

_DENTAL_SIGNALS = [
    "tooth", "teeth", "gum", "gums", "mouth", "oral", "dental", "dentist",
    "jaw", "bite", "biting", "cavity", "filling", "crown", "implant",
    "root canal", "braces", "plaque", "enamel", "dentin", "pulp",
    "extraction", "wisdom", "molar", "incisor", "premolar", "canine",
    "veneer", "whitening", "bleaching", "floss", "toothbrush", "toothpaste",
    "gingiv", "periodon", "endodon", "orthodon", "calculus", "tartar",
    "abscess", "deep cleaning", "scaling", "anesthesia", "x-ray",
    "panoramic", "space maintainer", "retainer", "denture", "bridge",
    "pericoronitis", "caries", "pulpitis", "gingivitis", "periodontitis",
    "bone graft", "sinus lift", "fluoride", "sealant",
    "occlusion", "malocclusion", "tmj", "bruxism", "clench", "grind",
    "gauze", "post-op", "aftercare", "socket",
    "pain", "ache", "aching", "toothache", "hurt", "hurts", "hurting",
    "swollen", "swelling", "swell",
    "bleed", "bleeding", "bleeds",
    "sensitive", "sensitivity",
    "numb", "numbness",
    "sore", "soreness",
    "discomfort", "throbbing",
    "سن", "أسنان", "ضرس", "ضروس", "لثة", "فم", "فكّ", "فك",
    "طبيب أسنان", "طبيب الأسنان", "عيادة الأسنان",
    "حشوة", "حشوات", "تاج", "زرعة",
    "علاج العصب", "تقويم", "تبييض",
    "جير الأسنان", "البلاك", "خيط الأسنان",
    "التهاب اللثة", "تسوس", "خلع",
    "تنظيف عميق", "تقليح", "تلميع",
    "حافظ مسافة", "مثبت", "طقم أسنان", "جسر أسنان",
    "تخدير", "بنج", "إبرة التخدير",
    "التهاب حول التاج", "خراج", "كيس",
    "انحسار اللثة", "قشرة", "ابتسامة هوليود",
    "رحى", "ناب", "قاطعة", "ضاحكة",
    "فلور", "سيلنت", "قطعة شاش", "شاش",
    "ألم", "وجع", "يؤلم", "يؤلمني", "مؤلم",
    "تورم", "منتفخ", "انتفاخ",
    "نزيف", "ينزف", "نزف",
    "حساسية",
    "خدر",
    "أشعر", "أحس", "لاحظت",
    "يؤذي",
]

_PRESCRIPTION_PATTERNS = [
    r"prescribe\s+(me\s+)?(a\s+)?medication",
    r"write\s+(me\s+)?a\s+prescription",
    r"give\s+me\s+a\s+specific\s+(prescription|treatment\s+plan)",
    r"make\s+(me\s+)?a\s+(treatment|care)\s+plan\s+for\s+(me|my\s+case)",
    r"tell\s+me\s+exactly\s+what\s+(drug|medication|antibiotic)\s+to\s+take\s+for\s+my",
    r"diagnose\s+me\s+exactly",
    r"وصّف\s+لي\s+دواء",
    r"وصف\s+لي\s+دواء\s+بالضبط",
    r"اعطني\s+وصفة\s+طبية\s+لحالتي",
    r"أعطني\s+وصفة\s+لحالتي",
    r"اكتب\s+لي\s+خطة\s+علاج",
    r"شخّص\s+حالتي\s+بالضبط",
]


def is_treatment_request(q: str) -> bool:
    ql = (q or "").lower()
    return any(re.search(p, ql) for p in _PRESCRIPTION_PATTERNS)


def is_out_of_scope(q: str) -> bool:
    ql = (q or "").lower()
    return any(b in ql for b in _NON_DENTAL_BLOCK)


def has_dental_signal(text: str) -> bool:
    tl = (text or "").lower()
    return any(sig.lower() in tl for sig in _DENTAL_SIGNALS)


def history_has_dental_context(history: list) -> bool:
    if not history:
        return False
    recent = history[-4:]
    for turn in recent:
        content = turn.get("content", "")
        if has_dental_signal(content):
            return True
    return False


def refusal_treatment(q: str) -> str:
    return (
        "ما أقدر أوصف أدوية أو أقدم تشخيصاً مخصصاً. يُنصح بمراجعة طبيب أسنان مرخّص."
        if is_ar(q) else
        "I can't prescribe medication or provide a personalised diagnosis. Please consult a licensed dentist."
    )


def refusal_scope(q: str) -> str:
    return (
        "هذا السؤال خارج نطاق تطبيق صحة الفم والأسنان."
        if is_ar(q) else
        "This is outside the scope of this oral health application."
    )


def insufficient_info(q: str) -> str:
    return (
        "المعلومات المتاحة لدي لا تكفي للإجابة بشكل موثوق على هذا السؤال."
        if is_ar(q) else
        "I don't have enough grounded information to answer this reliably."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LIMITED SOCIAL HANDLER
# Only for true standalone greetings / thanks / goodbye.
# ─────────────────────────────────────────────────────────────────────────────
_SOCIAL_EN = [
    r"^(hi|hello|hey|good\s*(morning|afternoon|evening|day))[\s!.,?]*$",
    r"^(thanks|thank\s*you|thank\s*u|thx|tysm)[\s!.,?]*$",
    r"^(bye|goodbye|see\s*you|take\s*care)[\s!.,?]*$",
]

_SOCIAL_AR = [
    r"^(مرحبا|أهلاً|أهلا|هلا|السلام\s*عليكم|صباح\s*الخير|مساء\s*الخير)[\s!.,؟]*$",
    r"^(شكرًا|شكراً|شكرا|ممنون|مشكور|يسلموا|يعطيك\s*العافية)[\s!.,؟]*$",
    r"^(مع\s*السلامة|وداعاً|باي)[\s!.,؟]*$",
]


def is_social_exchange(q: str) -> bool:
    ql = (q or "").lower().strip()
    qa = (q or "").strip()

    for p in _SOCIAL_EN:
        if re.search(p, ql, re.IGNORECASE):
            return True
    for p in _SOCIAL_AR:
        if re.search(p, qa):
            return True
    return False


def social_response(q: str) -> str:
    if is_ar(q):
        if re.search(r"شكر|ممنون|مشكور|يسلموا|يعطيك", q):
            return "على الرحب والسعة."
        if re.search(r"السلام\s*عليكم", q):
            return "وعليكم السلام."
        if re.search(r"مرحبا|أهلاً|أهلا|هلا|صباح|مساء", q):
            return "أهلاً."
        if re.search(r"مع\s*السلامة|وداعاً|باي", q):
            return "مع السلامة."
        return "أهلاً."
    else:
        ql = (q or "").lower().strip()
        if re.search(r"thanks|thank\s*you|thx", ql):
            return "You're welcome."
        if re.search(r"^(hi|hello|hey|good)", ql):
            return "Hello."
        if re.search(r"bye|goodbye|see\s*you|take\s*care", ql):
            return "Take care."
        return "Hello."


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION TYPE
# ─────────────────────────────────────────────────────────────────────────────
def detect_question_type(q: str) -> str:
    ql = (q or "").lower().strip()

    instruction_patterns = [
        "how to", "how do i", "what should i do", "what should i avoid",
        "what to do after", "what to avoid after",
        "aftercare", "after care", "post-op", "post op", "post operative",
        "postoperative", "post procedure", "post-procedure", "post treatment",
        "after surgery", "after extraction", "after root canal", "after implant",
        "after filling", "after procedure", "after treatment", "after whitening",
        "step by step", "care instructions", "post-operative instructions",
        "can i eat", "can i drink", "when can i eat", "when can i drink",
        "what should i be aware of", "what should i refrain from doing",
        "how should i care", "recovery instructions",
        "كيف أعتني", "كيف أتعامل", "كيف أتصرف",
        "ماذا أفعل بعد", "ماذا أعمل بعد", "بعد العملية", "بعد الخلع",
        "بعد التركيب", "بعد الزرعة", "بعد الحشوة", "بعد التنظيف", "بعد التبييض",
        "ايش أسوي بعد", "وش أسوي بعد",
        "هل أقدر آكل بعد", "هل أقدر أشرب بعد", "متى أقدر آكل", "متى أقدر أشرب",
        "تعليمات بعد", "خطوات العناية", "العناية بعد", "إرشادات بعد",
        "ما الذي يجب أن أتجنبه", "ما الذي يجب أن أنتبه له",
        "ما الذي يجب أن أكون على علم به", "تعليمات ما بعد", "تعليمات ما بعد الإجراء",
    ]

    symptom_patterns = [
        "i have", "i feel", "i notice", "i noticed",
        "i'm feeling", "i've been", "i experience", "i experienced",
        "my tooth", "my teeth", "my gum", "my gums", "my mouth", "my jaw",
        "pain that", "pain when", "pain in my",
        "عندي ألم", "عندي تورم", "عندي نزيف",
        "أشعر ب", "أحس ب", "لاحظت", "صار عندي",
        "سني يؤلم", "أسناني تؤلم", "لثتي تؤلم",
        "يؤلمني", "مؤلم", "ينزف", "متورم",
    ]

    for s in instruction_patterns:
        if s in ql:
            return "instruction"

    for s in symptom_patterns:
        if s in ql:
            return "symptom"

    return "informational"


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────
def embed(text: str) -> list:
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _load_json_file(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_json_file(path: str, payload) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        log.warning(f"Could not save cache file {path}: {e}")


def _load_or_build_dataset_embeddings():
    cached = _load_json_file(_DATASET_EMBED_CACHE_FILE)
    if cached and isinstance(cached, dict) and "en" in cached and "ar" in cached:
        if len(cached["en"]) == len(DATASET) and len(cached["ar"]) == len(DATASET):
            log.info("Loaded dataset embeddings from cache.")
            return cached["en"], cached["ar"]

    log.info("Computing dataset embeddings...")
    en_vecs = [embed(d["en_q"]) for d in DATASET]
    ar_vecs = [embed(d["ar_q"]) for d in DATASET]
    _save_json_file(_DATASET_EMBED_CACHE_FILE, {"en": en_vecs, "ar": ar_vecs})
    log.info(f"Done — {len(DATASET)} entries cached.")
    return en_vecs, ar_vecs


_DENTAL_INTENT_ANCHORS = [
    "tooth pain",
    "gum bleeding",
    "dental treatment",
    "oral health problem",
    "toothache",
    "swollen gums",
    "jaw pain dental",
]

_NON_DENTAL_INTENT_ANCHORS = [
    "weather forecast",
    "stock market price",
    "movie review",
    "travel booking",
    "car specifications",
    "programming tutorial",
]


def _load_or_build_scope_embeddings():
    cached = _load_json_file(_SCOPE_EMBED_CACHE_FILE)
    if (
        cached
        and isinstance(cached, dict)
        and cached.get("dental_anchors") == _DENTAL_INTENT_ANCHORS
        and cached.get("non_dental_anchors") == _NON_DENTAL_INTENT_ANCHORS
        and "dental_vecs" in cached
        and "non_dental_vecs" in cached
    ):
        log.info("Loaded scope embeddings from cache.")
        return cached["dental_vecs"], cached["non_dental_vecs"]

    log.info("Computing scope anchor embeddings...")
    dental_vecs = [embed(x) for x in _DENTAL_INTENT_ANCHORS]
    non_dental_vecs = [embed(x) for x in _NON_DENTAL_INTENT_ANCHORS]

    _save_json_file(
        _SCOPE_EMBED_CACHE_FILE,
        {
            "dental_anchors": _DENTAL_INTENT_ANCHORS,
            "non_dental_anchors": _NON_DENTAL_INTENT_ANCHORS,
            "dental_vecs": dental_vecs,
            "non_dental_vecs": non_dental_vecs,
        },
    )
    return dental_vecs, non_dental_vecs


_DS_EN_VECS, _DS_AR_VECS = _load_or_build_dataset_embeddings()
_DENTAL_ANCHOR_VECS, _NON_DENTAL_ANCHOR_VECS = _load_or_build_scope_embeddings()


def semantic_scope_score(qv: list):
    dental_score = max(cosine(qv, v) for v in _DENTAL_ANCHOR_VECS)
    non_dental_score = max(cosine(qv, v) for v in _NON_DENTAL_ANCHOR_VECS)
    return dental_score, non_dental_score


def dataset_match(q: str, qv=None):
    ar = is_ar(q)
    if qv is None:
        qv = embed(q)
    vecs = _DS_AR_VECS if ar else _DS_EN_VECS

    best = None
    score = -1.0
    best_idx = -1

    for i, dv in enumerate(vecs):
        s = cosine(qv, dv)
        if s > score:
            score = s
            best = DATASET[i]
            best_idx = i

    return best, score, ar, best_idx


# ─────────────────────────────────────────────────────────────────────────────
# RAG RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────
def _extract_text(md: dict) -> str:
    val = md.get(PINECONE_CHUNK_FIELD)
    if val is not None:
        text = str(val).strip()
        if text:
            return text

    for field in _CHUNK_FALLBACK_FIELDS:
        val = md.get(field)
        if val is not None:
            text = str(val).strip()
            if text:
                log.warning(
                    f"_extract_text: primary field '{PINECONE_CHUNK_FIELD}' empty. Used fallback field '{field}'."
                )
                return text

    for key, val in md.items():
        if key in (
            PINECONE_TITLE_FIELD,
            PINECONE_SOURCE_FIELD,
            PINECONE_AUTHORITY_FIELD,
            PINECONE_PATH_FIELD,
        ):
            continue
        if isinstance(val, str) and len(val.strip()) > 40:
            text = val.strip()
            log.warning(f"_extract_text: using last-resort field '{key}' (length={len(text)}).")
            return text

    log.error(f"_extract_text: no usable text field found. Fields present: {list(md.keys())}")
    return ""


def assess_context_strength(accepted_chunks: list) -> dict:
    if not accepted_chunks:
        return {
            "is_sufficient": False,
            "top_score": 0.0,
            "avg_score": 0.0,
            "total_chars": 0,
        }

    top_score = max(c["score"] for c in accepted_chunks)
    avg_score = sum(c["score"] for c in accepted_chunks) / len(accepted_chunks)
    total_chars = sum(len(c["text"]) for c in accepted_chunks)

    is_sufficient = (
        top_score >= MIN_CONTEXT_TOP_SCORE
        and avg_score >= MIN_CONTEXT_AVG_SCORE
        and total_chars >= MIN_CONTEXT_TOTAL_CHARS
    )

    return {
        "is_sufficient": is_sufficient,
        "top_score": round(top_score, 4),
        "avg_score": round(avg_score, 4),
        "total_chars": total_chars,
    }


def rag_retrieve(qv: list):
    """
    Returns:
      context_chunks, display_titles, accepted_chunks, is_off_topic, debug_info
    """
    debug_info = {
        "pinecone_match_count": 0,
        "top_score": None,
        "chunks_injected": 0,
        "authority_filter_dropped": 0,
        "field_errors": 0,
        "context_avg_score": 0.0,
        "context_total_chars": 0,
        "context_sufficient": False,
    }

    try:
        res = index.query(vector=qv, top_k=TOP_K_RAW, include_metadata=True)
        matches = res.get("matches", [])
    except Exception as e:
        log.error(f"Pinecone query failed: {e}")
        return [], [], [], False, debug_info

    debug_info["pinecone_match_count"] = len(matches)

    if not matches:
        log.info("RAG: 0 matches returned by Pinecone")
        return [], [], [], False, debug_info

    top_score = float(matches[0].get("score", 0))
    debug_info["top_score"] = round(top_score, 4)

    if top_score < MIN_RELEVANCE:
        log.info("RAG: off-topic — top_score below MIN_RELEVANCE")
        return [], [], [], True, debug_info

    accepted = []

    for m in matches:
        score = float(m.get("score", 0))
        md = m.get("metadata") or {}

        raw_auth = md.get(PINECONE_AUTHORITY_FIELD)
        if raw_auth is not None:
            try:
                auth = float(raw_auth)
                if auth < MIN_AUTHORITY:
                    debug_info["authority_filter_dropped"] += 1
                    continue
            except (TypeError, ValueError):
                log.warning(f"RAG: authority value {raw_auth!r} not parseable; accepting chunk.")

        text = _extract_text(md)
        if not text:
            debug_info["field_errors"] += 1
            continue

        accepted.append(
            {
                "title": str(md.get(PINECONE_TITLE_FIELD) or ""),
                "text": text,
                "score": score,
            }
        )

    if not accepted:
        log.warning("RAG: no usable chunks after filtering.")
        return [], [], [], False, debug_info

    seen = set()
    unique = []
    for chunk in accepted:
        key = chunk["title"].strip().lower() if chunk["title"] else chunk["text"][:180].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    final_chunks = unique[:TOP_K_FINAL]
    context_chunks = [c["text"] for c in final_chunks]
    display_titles = [
        c["title"]
        for c in final_chunks
        if c["title"] and c["score"] >= MIN_REF_DISPLAY
    ]

    strength = assess_context_strength(final_chunks)
    debug_info["chunks_injected"] = len(context_chunks)
    debug_info["context_avg_score"] = strength["avg_score"]
    debug_info["context_total_chars"] = strength["total_chars"]
    debug_info["context_sufficient"] = strength["is_sufficient"]

    return context_chunks, display_titles, final_chunks, False, debug_info


# ─────────────────────────────────────────────────────────────────────────────
# DATASET FALLBACK CONTEXT
# Used only when RAG returns no sufficient context.
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset_fallback_context(q: str, qv: list):
    match, score, ar, idx = dataset_match(q, qv)
    if match and score >= SIM_THRESHOLD:
        text = match["ar_a"] if ar else match["en_a"]
        title = f"dataset_fallback_{idx}"
        log.info(f"Dataset fallback used: idx={idx}, score={score:.4f}")
        return [text], [title], score
    return [], [], score if match is not None else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# GPT GENERATION
# Strict grounded generation only.
# ─────────────────────────────────────────────────────────────────────────────
_FORBIDDEN = (
    "STRICTLY FORBIDDEN:\n"
    "• Do not use numbered lists.\n"
    "• Do not use markdown headings.\n"
    "• Do not use bold section titles.\n"
    "• Do not add summary or conclusion phrases.\n"
    "• Do not end with offers like 'I can also tell you...' or 'let me know if...'.\n"
    "• Do not mention sources, references, retrieval, documents, context, dataset, or grounding.\n"
    "• End at the last relevant sentence.\n"
)


def _build_context_block(context_chunks: list, ar: bool) -> str:
    if ar:
        return (
            "REFERENCE MATERIAL:\n"
            "أجب اعتماداً على النص التالي فقط.\n"
            "ممنوع إضافة معلومات غير موجودة فيه.\n"
            "إذا لم يكن النص كافياً للإجابة، اكتب هذه الجملة فقط حرفياً:\n"
            "المعلومات المتاحة لدي لا تكفي للإجابة بشكل موثوق على هذا السؤال.\n\n"
            + "\n---\n".join(context_chunks)
        )

    return (
        "REFERENCE MATERIAL:\n"
        "Answer using only the text below.\n"
        "Do not add facts that are not supported by it.\n"
        "If the text is not enough to answer, output exactly this sentence:\n"
        "I don't have enough grounded information to answer this reliably.\n\n"
        + "\n---\n".join(context_chunks)
    )


def _build_format_rules(ar: bool, qtype: str) -> str:
    if ar:
        if qtype == "instruction":
            return (
                "FORMAT:\n"
                "• استخدم الرمز • فقط لكل نقطة.\n"
                "• من 3 إلى 5 نقاط كحد أقصى.\n"
                "• كل نقطة تكون جملة واحدة واضحة ومباشرة.\n"
            )
        return (
            "FORMAT:\n"
            "• نثر مباشر فقط، بدون نقاط أو قوائم.\n"
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


def _build_language_tone_rules(ar: bool) -> str:
    if ar:
        return (
            "LANGUAGE AND TONE:\n"
            "• اكتب بالعربية فقط، بدون خلط مع الإنجليزية.\n"
            "• استخدم عربية طبيعية نظيفة، حيادية، ومهنية.\n"
            "• لا تستخدم أسلوباً أكاديمياً ثقيلاً.\n"
        )

    return (
        "LANGUAGE AND TONE:\n"
        "• Write in English only.\n"
        "• Use plain, clear, professional English.\n"
        "• Avoid academic or textbook phrasing.\n"
    )


def grounded_answer(q: str, context_chunks: list, history=None) -> str:
    ar = is_ar(q)
    qtype = detect_question_type(q)

    role_layer = (
        "You are ORA, a dental health assistant for general patients.\n"
        "Your job is to produce a concise patient-facing answer that stays strictly within the provided reference material.\n"
    )

    behavior_layer = (
        "CORE BEHAVIOR:\n"
        "• Use the current message plus recent conversation only to understand what the user is asking.\n"
        "• Do not use outside knowledge when reference material is present.\n"
        "• Do not infer missing clinical details.\n"
        "• Do not guess.\n"
        "• Do not prescribe for a specific patient.\n"
        "• Do not write a personalised treatment plan.\n"
        "• Do not ask follow-up questions.\n"
        "• Do not add filler or extra reassurance.\n"
    )

    escalation_layer = (
        "ESCALATION:\n"
        "• Mention urgent assessment only if the reference material itself supports that level of concern.\n"
    )

    system = "\n".join(
        [
            role_layer,
            behavior_layer,
            escalation_layer,
            _build_format_rules(ar, qtype),
            _build_language_tone_rules(ar),
            _FORBIDDEN,
            _build_context_block(context_chunks, ar),
        ]
    )

    msgs = [{"role": "system", "content": system}]

    if history:
        for turn in history[-6:]:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})

    msgs.append({"role": "user", "content": q})

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            max_tokens=MAX_GPT_TOKENS,
            temperature=TEMPERATURE,
        )
        text = (r.choices[0].message.content or "").strip()
        return text or insufficient_info(q)
    except Exception as e:
        log.error(f"GPT error: {e}")
        return insufficient_info(q)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# Strict flow:
#   1) safety / scope
#   2) social
#   3) embed
#   4) semantic scope gate
#   5) RAG
#   6) dataset fallback only if RAG is insufficient
#   7) grounded generation only
#   8) fail closed when insufficient
# ─────────────────────────────────────────────────────────────────────────────
def generate_answer(q: str, history=None):
    if is_treatment_request(q):
        return {
            "answer": refusal_treatment(q),
            "refs": [],
            "source": "safety_refusal",
            "debug": {},
        }

    if is_social_exchange(q):
        return {
            "answer": social_response(q),
            "refs": [],
            "source": "social",
            "debug": {},
        }

    qv = embed(q)

    dental_score, non_dental_score = semantic_scope_score(qv)
    is_semantically_dental = dental_score >= SEMANTIC_DENTAL_THRESHOLD
    is_semantically_non_dental = non_dental_score > dental_score

    if (
        is_semantically_non_dental
        and not is_semantically_dental
        and is_out_of_scope(q)
        and not has_dental_signal(q)
        and not history_has_dental_context(history or [])
    ):
        return {
            "answer": refusal_scope(q),
            "refs": [],
            "source": "scope_refusal",
            "debug": {
                "dental_score": round(dental_score, 4),
                "non_dental_score": round(non_dental_score, 4),
            },
        }

    context_chunks, refs, accepted_chunks, off_topic, debug = rag_retrieve(qv)
    debug["dental_score"] = round(dental_score, 4)
    debug["non_dental_score"] = round(non_dental_score, 4)

    if (
        off_topic
        and is_semantically_non_dental
        and not is_semantically_dental
        and not has_dental_signal(q)
        and not history_has_dental_context(history or [])
    ):
        return {
            "answer": refusal_scope(q),
            "refs": [],
            "source": "scope_refusal",
            "debug": debug,
        }

    # RAG mode: answer only when context is sufficient
    if debug.get("context_sufficient"):
        answer = grounded_answer(q, context_chunks=context_chunks, history=history)
        return {
            "answer": answer,
            "refs": refs,
            "source": "rag",
            "debug": debug,
        }

    # Controlled dataset fallback: only when highly similar
    dataset_ctx, dataset_refs, dataset_score = build_dataset_fallback_context(q, qv)
    debug["dataset_score"] = round(dataset_score, 4)

    if dataset_ctx:
        answer = grounded_answer(q, context_chunks=dataset_ctx, history=history)
        return {
            "answer": answer,
            "refs": dataset_refs,
            "source": "dataset_fallback",
            "debug": debug,
        }

    # Fail closed: no free-form GPT answer
    return {
        "answer": insufficient_info(q),
        "refs": [],
        "source": "insufficient_grounded_context",
        "debug": debug,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI (testing only)
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
        log.info(f"refs: {result['refs']}")
        log.info(f"source: {result['source']}")
        log.info(f"debug: {result['debug']}")

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result["answer"]})
