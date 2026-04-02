import re
import math
import logging
import os
import uuid
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="[ORA %(levelname)s] %(message)s")
log = logging.getLogger("ora")


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MODEL                    = "gpt-4o"
EMBED_MODEL              = "text-embedding-3-large"
PINECONE_INDEX           = "oraapp777"

SIM_THRESHOLD            = 0.70
MIN_RELEVANCE            = 0.28
MIN_AUTHORITY            = 0.40
TOP_K_RAW                = 6
TOP_K_FINAL              = 3
MAX_GPT_TOKENS           = 520
MIN_REF_DISPLAY          = 0.70

SESSION_TTL_SECONDS      = 3600   # sessions expire after 1 hour of inactivity
SESSION_MAX_TURNS        = 10     # keep last 10 turns (20 messages) per session

PINECONE_CHUNK_FIELD     = "chunk_text"
PINECONE_TITLE_FIELD     = "title"
PINECONE_SOURCE_FIELD    = "source_type"
PINECONE_AUTHORITY_FIELD = "authority_score"
PINECONE_PATH_FIELD      = "source_path"

_CHUNK_FALLBACK_FIELDS   = ("text", "content", "chunk", "body", "passage", "page_content")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index  = pc.Index(PINECONE_INDEX)


# ─────────────────────────────────────────────────────────────────────────────
#  STATIC DATASET
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
            "prevents progression."
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
            "naturally thin — the metallic components show through the tissue. This is an anatomical "
            "variation, not a sign of infection or implant failure. It is primarily an aesthetic "
            "concern. A periodontist or implant specialist can assess whether soft tissue correction "
            "is appropriate."
        ),
        "ar_q": "بعد الزرعة صار عندي لون أزرق في اللثة",
        "ar_a": (
            "اللون الأزرق بالقرب من الزرعة يصير في الغالب عند من تكون لثتهم رقيقة بطبيعتها — "
            "المكونات المعدنية للزرعة تبيّن من خلال النسيج الرقيق. هذا تغيّر تشريحي طبيعي وما هو "
            "علامة على فشل الزرعة أو التهاب، هو في الغالب مسألة مظهرية. لو كان يضايقك، طبيب "
            "متخصص في أمراض اللثة يقدر يقيّم إذا كان التدخل مناسب."
        ),
    },
    {
        "field": ["Restorative Dentistry"],
        "en_q": "I had a filling and now feel pain only when biting. What does it mean?",
        "en_a": (
            "Pain only when biting after a filling usually means the restoration sits slightly too high "
            "— called a high occlusion. The filling contacts the opposing tooth before the rest of the "
            "bite does, creating localised pressure when chewing. Unlike general sensitivity, this pain "
            "is specific to biting force. A quick adjustment at your next dentist visit resolves it."
        ),
        "ar_q": "بعد الحشوة عندي ألم عند العض فقط",
        "ar_a": (
            "الألم عند العض فقط بعد الحشوة يعني في الغالب إن الحشوة مرتفعة شوي — تلامس السن "
            "المقابل قبل بقية الأسنان، وهذا يسبب ضغطاً موضعياً أثناء المضغ. هذا يختلف عن "
            "الحساسية العامة لأن الألم مرتبط بقوة العض تحديداً. تعديل بسيط عند طبيب الأسنان "
            "يحل المشكلة."
        ),
    },
    {
        "field": ["Endodontics"],
        "en_q": "I had severe tooth pain that disappeared on its own. What does it mean?",
        "en_a": (
            "When severe tooth pain disappears on its own, it often means the pulp — the "
            "nerve-containing tissue inside the tooth — has lost vitality. As the nerve dies, pain "
            "signals stop, but the bacterial infection continues. This is not recovery; the infection "
            "can spread beyond the root if left untreated. An evaluation is necessary even without pain."
        ),
        "ar_q": "اختفى الألم الشديد في السن من تلقاء نفسه",
        "ar_a": (
            "اختفاء الألم من تلقاء نفسه ما يعني الشفاء. في الغالب يعني إن لب السن — نسيج العصب "
            "بداخله — فقد حيويته. لما يموت العصب تتوقف الإشارات المؤلمة، بس العدوى البكتيرية "
            "تستمر. لو تُركت دون علاج يمكن تمتد خارج جذر السن. المراجعة ضرورية حتى في غياب "
            "الألم."
        ),
    },
    {
        "field": ["Endodontics"],
        "en_q": "What does tooth decay actually mean?",
        "en_a": (
            "Tooth decay means bacteria have invaded the inner layers of the tooth — the dentin or "
            "pulp — usually after untreated early-stage cavities or trauma. The tooth does not "
            "literally decompose, but ongoing bacterial activity causes progressive structural damage. "
            "How far it has progressed determines what treatment is needed."
        ),
        "ar_q": "ما معنى تسوس أو تعفن السن؟",
        "ar_a": (
            "التسوس يعني إن البكتيريا اخترقت الطبقات الداخلية للسن — الدنتين أو اللب — عادةً بسبب "
            "تسوس سطحي لم يُعالج في وقته أو بسبب صدمة. السن ما تتحلل حرفياً، بس النشاط البكتيري "
            "يسبب تلفاً تدريجياً في بنيتها. العلاج يعتمد على مدى تقدم التسوس."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  SCOPE & SAFETY
# ─────────────────────────────────────────────────────────────────────────────

def is_ar(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


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
    ql = q.lower()
    return any(re.search(p, ql) for p in _PRESCRIPTION_PATTERNS)


def is_out_of_scope(q: str) -> bool:
    ql = q.lower()
    return any(b in ql for b in _NON_DENTAL_BLOCK)


def has_dental_signal(text: str) -> bool:
    tl = text.lower()
    return any(sig.lower() in tl for sig in _DENTAL_SIGNALS)


def history_has_dental_context(history: list) -> bool:
    """
    Returns True if any prior turn in the conversation contains a dental signal.
    Used as a tiebreaker: a borderline or ambiguous follow-up question should not
    be refused if the conversation is clearly already about dental health.
    """
    for turn in history:
        content = turn.get("content", "")
        if has_dental_signal(content):
            return True
    return False


def refusal_treatment(q: str) -> str:
    return (
        "ما أقدر أوصف أدوية أو أقدم تشخيصاً مخصصاً. للعلاج المناسب، يُنصح بمراجعة طبيب أسنان."
        if is_ar(q) else
        "I can't prescribe medication or provide a personalised diagnosis. "
        "Please consult a licensed dentist."
    )


def refusal_scope(q: str) -> str:
    return (
        "هذا السؤال خارج نطاق تطبيق صحة الفم والأسنان."
        if is_ar(q) else
        "This is outside the scope of this oral health application."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  SOCIAL / CONVERSATIONAL INTENT
# ─────────────────────────────────────────────────────────────────────────────

_SOCIAL_EN = [
    r"^(hi|hello|hey|good\s*(morning|afternoon|evening|day))[\s!.,?]*$",
    r"^(thanks|thank\s*you|thank\s*u|thx|tysm)[\s!.,?]*$",
    r"^(ok|okay|got\s*it|alright|understood|makes\s*sense|sounds\s*good|great|perfect)[\s!.,?]*$",
    r"^(yes|no|yep|nope|sure|of\s*course|certainly)[\s!.,?]*$",
    r"^(bye|goodbye|see\s*you|take\s*care)[\s!.,?]*$",
    r"(i\s+(have|got|'ve\s+got)\s+a\s+question)",
    r"(i\s+have\s+a\s+question\s+for\s+you)",
    r"(can\s+i\s+ask(\s+you)?(\s+something)?[\s?]*$)",
    r"^i'm\s+(at\s+the\s+dentist|ready|here|listening)[\s!.,?]*$",
]

_SOCIAL_AR = [
    r"^(مرحبا|أهلاً|أهلا|هلا|السلام\s*عليكم|صباح\s*الخير|مساء\s*الخير)[\s!.,؟]*$",
    r"^(شكرًا|شكراً|شكرا|ممنون|مشكور|يسلموا|يعطيك\s*العافية)[\s!.,؟]*$",
    r"^(حسناً|حسنًا|تمام|زين|ماشي|فاهم|واضح|أوكي|اوكي|صح|نعم|أيوه|لا)[\s!.,؟]*$",
    r"^(مع\s*السلامة|وداعاً|باي)[\s!.,؟]*$",
    r"(عندي\s+سؤال)",
    r"(أبغى\s+أسأل|بغيت\s+أسأل|ودي\s+أسأل|أريد\s+أن\s+أسأل)",
]


def is_social_exchange(q: str) -> bool:
    ql = q.lower().strip()
    for p in _SOCIAL_EN:
        if re.search(p, ql, re.IGNORECASE):
            return True
    for p in _SOCIAL_AR:
        if re.search(p, q.strip()):
            return True
    return False


def social_response(q: str) -> str:
    ar = is_ar(q)
    ql = q.lower().strip()
    if ar:
        if re.search(r"شكر|ممنون|مشكور|يسلموا|يعطيك", q):
            return "على الرحب والسعة."
        if re.search(r"السلام عليكم", q):
            return "وعليكم السلام! كيف أقدر أساعدك؟"
        if re.search(r"مرحبا|أهلاً|أهلا|هلا|صباح|مساء", q):
            return "أهلاً! كيف أقدر أساعدك في صحة أسنانك؟"
        if re.search(r"مع السلامة|وداعاً|باي", q):
            return "مع السلامة!"
        if re.search(r"عندي سؤال|أبغى أسأل|بغيت أسأل|ودي أسأل", q):
            return "تفضل، أنا أسمعك."
        return "تفضل."
    else:
        if re.search(r"thanks|thank\s*you|thx", ql):
            return "You're welcome."
        if re.search(r"^(hi|hello|hey)", ql):
            return "Hello! How can I help you with your dental health today?"
        if re.search(r"bye|goodbye|see\s*you", ql):
            return "Take care!"
        if re.search(r"(have|got)\s+a\s+question|question\s+for\s+you|can\s+i\s+ask", ql):
            return "Go ahead."
        if re.search(r"i'm\s+(at|ready|here|listening)", ql):
            return "Go ahead, I'm listening."
        return "Of course, go ahead."


# ─────────────────────────────────────────────────────────────────────────────
#  QUESTION TYPE
# ─────────────────────────────────────────────────────────────────────────────

def detect_question_type(q: str) -> str:
    ql = q.lower().strip()

    _INSTRUCTION = [
        "how to", "how do i", "how should i",
        "what should i do", "what do i do after", "what do i do if",
        "aftercare", "after care",
        "after surgery", "after extraction", "after root canal",
        "after implant", "after filling", "after procedure", "after treatment",
        "step by step", "what to do after", "what to avoid", "what not to",
        "can i eat", "can i drink", "when can i eat", "when can i drink",
        "should i brush", "should i rinse", "should i stop", "should i use",
        "how many times", "how often should", "how long should i",
        "كيف أعتني", "كيف أتعامل", "كيف أتصرف", "كيف أنظف", "كيف أستخدم",
        "ماذا أفعل بعد", "ماذا أفعل إذا",
        "وش أسوي", "ايش أسوي", "إيش أسوي",
        "بعد العملية", "بعد الخلع", "بعد التركيب",
        "بعد الزرعة", "بعد الحشوة", "بعد التنظيف", "بعد علاج العصب",
        "هل أقدر آكل", "هل أقدر أشرب", "متى أقدر آكل", "متى أقدر أشرب",
        "تعليمات بعد", "خطوات العناية",
        "ايش لازم أسوي", "وش لازم أسوي", "إيش لازم أسوي",
        "كم مرة في اليوم", "كم مرة",
        "ايش أتجنب", "وش أتجنب", "إيش أتجنب", "ايش ما يصير",
        "هل لازم أ", "هل يجب أن أ",
        "متى أقدر",
    ]

    _SYMPTOM = [
        "i have", "i feel", "i notice", "i noticed",
        "i'm feeling", "i've been", "i experience", "i experienced",
        "my tooth", "my teeth", "my gum", "my gums", "my mouth", "my jaw",
        "pain that", "pain when", "pain in my", "hurts when", "hurts after",
        "there is pain", "there's pain", "there is swelling", "there's swelling",
        "i can feel", "i can taste", "i can see a",
        "عندي ألم", "عندي تورم", "عندي نزيف", "عندي حساسية",
        "أشعر ب", "أحس ب", "لاحظت في",
        "سني يؤلم", "أسناني تؤلم", "لثتي تؤلم", "فمي يؤلم",
        "يؤلمني", "يوجعني", "يوجع", "يؤلم",
        "صار عندي", "طلع عندي",
        "من لما", "من يوم ما",
        "منتفخ", "ينزف", "حاسس",
    ]

    for s in _INSTRUCTION:
        if s in ql:
            return "instruction"
    for s in _SYMPTOM:
        if s in ql:
            return "symptom"
    return "informational"


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def embed(text: str) -> list:
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


log.info("Pre-computing dataset embeddings...")
_DS_EN_VECS = [embed(d["en_q"]) for d in DATASET]
_DS_AR_VECS = [embed(d["ar_q"]) for d in DATASET]
log.info(f"Done — {len(DATASET)} entries cached.")


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET SHORTCUT
# ─────────────────────────────────────────────────────────────────────────────

INTENT_PHRASES = {
    0: ["gum bleed", "bleeding gums", "نزيف اللثة", "اللثة تنزف", "لثتي تنزف"],
    1: ["blue gum implant", "implant color", "زرقة اللثة بعد الزرعة"],
    2: ["pain when biting", "pain only when biting", "ألم عند العض"],
    3: ["pain disappeared", "tooth pain went away", "اختفى الألم", "راح الألم"],
    4: ["tooth rotting", "rotten tooth", "تعفن السن", "السن ميت"],
}


def dataset_match(q: str, qv=None):
    q_norm = q.lower().strip()
    ar = is_ar(q)
    for idx, phrases in INTENT_PHRASES.items():
        for p in phrases:
            if p in q_norm:
                return DATASET[idx], 1.0, ar, idx
    if qv is None:
        qv = embed(q)
    vecs = _DS_AR_VECS if ar else _DS_EN_VECS
    best, score, best_idx = None, -1, -1
    for i, dv in enumerate(vecs):
        s = cosine(qv, dv)
        if s > score:
            score, best, best_idx = s, DATASET[i], i
    return best, score, ar, best_idx


# ─────────────────────────────────────────────────────────────────────────────
#  RAG RETRIEVAL
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
                    f"_extract_text: primary field '{PINECONE_CHUNK_FIELD}' was empty. "
                    f"Used fallback field '{field}'. "
                    f"Consider updating PINECONE_CHUNK_FIELD to '{field}'."
                )
                return text

    for key, val in md.items():
        if key in (PINECONE_TITLE_FIELD, PINECONE_SOURCE_FIELD,
                   PINECONE_AUTHORITY_FIELD, PINECONE_PATH_FIELD):
            continue
        if isinstance(val, str) and len(val.strip()) > 40:
            text = val.strip()
            log.warning(
                f"_extract_text: no known text field found. "
                f"Using field '{key}' as last resort (length={len(text)}). "
                f"Update PINECONE_CHUNK_FIELD to '{key}'."
            )
            return text

    log.error(
        f"_extract_text: could not extract text from metadata. "
        f"Fields present: {list(md.keys())}. "
        f"Current PINECONE_CHUNK_FIELD='{PINECONE_CHUNK_FIELD}'."
    )
    return ""


def rag_retrieve(qv: list):
    debug_info = {
        "pinecone_match_count": 0,
        "top_score": None,
        "chunks_injected": 0,
        "authority_filter_dropped": 0,
        "fallback_used": False,
        "field_errors": 0,
    }

    try:
        res     = index.query(vector=qv, top_k=TOP_K_RAW, include_metadata=True)
        matches = res.get("matches", [])
        for m in matches:
            log.info(f"METADATA KEYS: {list((m.get('metadata') or {}).keys())}")
            break
    except Exception as e:
        log.error(f"Pinecone query failed: {e}")
        return [], [], False, debug_info

    debug_info["pinecone_match_count"] = len(matches)

    if not matches:
        log.info("RAG: 0 matches returned by Pinecone")
        return [], [], False, debug_info

    top_score = float(matches[0].get("score", 0))
    debug_info["top_score"] = round(top_score, 4)
    log.info(f"RAG: top_score={top_score:.4f}  threshold={MIN_RELEVANCE}")

    if top_score < MIN_RELEVANCE:
        log.info("RAG: off-topic — top_score below MIN_RELEVANCE")
        return [], [], True, debug_info

    accepted = []
    for m in matches:
        score = float(m.get("score", 0))
        md    = m.get("metadata") or {}

        raw_auth = md.get(PINECONE_AUTHORITY_FIELD)
        if raw_auth is not None:
            try:
                auth = float(raw_auth)
                if auth < MIN_AUTHORITY:
                    debug_info["authority_filter_dropped"] += 1
                    continue
            except (TypeError, ValueError):
                log.warning(f"RAG: unparseable authority_score '{raw_auth!r}' — chunk accepted.")

        text = _extract_text(md)
        if not text:
            debug_info["field_errors"] += 1
            continue

        accepted.append({
            "title": str(md.get(PINECONE_TITLE_FIELD) or ""),
            "text":  text,
            "score": score,
        })

    if not accepted and matches:
        log.warning("RAG fallback: re-running without authority filter.")
        debug_info["fallback_used"] = True
        debug_info["field_errors"]  = 0
        for m in matches[:TOP_K_FINAL]:
            md   = m.get("metadata") or {}
            text = _extract_text(md)
            if not text:
                debug_info["field_errors"] += 1
                continue
            accepted.append({
                "title": str(md.get(PINECONE_TITLE_FIELD) or ""),
                "text":  text,
                "score": float(m.get("score", 0)),
            })

    if not accepted:
        log.warning("RAG: no chunks injected after fallback.")
        return [], [], False, debug_info

    seen, unique = set(), []
    for chunk in accepted:
        key = chunk["title"] if chunk["title"] else chunk["text"][:80]
        if key not in seen:
            seen.add(key)
            unique.append(chunk)

    context_chunks = [c["text"]  for c in unique[:TOP_K_FINAL]]
    display_titles = [
        c["title"] for c in unique
        if c["title"] and c["score"] >= MIN_REF_DISPLAY
    ][:TOP_K_FINAL]

    debug_info["chunks_injected"] = len(context_chunks)
    log.info(
        f"RAG: injected={len(context_chunks)}  "
        f"authority_dropped={debug_info['authority_filter_dropped']}  "
        f"fallback={debug_info['fallback_used']}  "
        f"field_errors={debug_info['field_errors']}"
    )

    return context_chunks, display_titles, False, debug_info


# ─────────────────────────────────────────────────────────────────────────────
#  GPT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def gpt_style_answer(q: str, context_chunks=None, history=None) -> str:
    ar       = is_ar(q)
    qtype    = detect_question_type(q)
    has_hist = bool(history)

    # ── Block 1: Language enforcement (first = highest attention weight) ───────
    if ar:
        lang_enforce = (
            "══ CRITICAL — LANGUAGE RULE — THIS OVERRIDES ALL OTHER INSTRUCTIONS ══\n"
            "Your entire response MUST be written in Arabic only.\n"
            "Do NOT use any English words, phrases, or sentences anywhere in your response.\n"
            "The ONLY permitted exception: a widely-known clinical acronym such as X-ray or CT scan "
            "may appear in English inside parentheses immediately after its Arabic name, "
            "on first mention only. Example: أشعة إكس (X-ray) — permitted once.\n"
            "Writing the word 'gauze' alone is NEVER permitted — always use قطعة شاش.\n"
            "Writing any English medical term without an Arabic equivalent first is NEVER permitted.\n"
            "══════════════════════════════════════════════════════════════════════\n\n"
        )
    else:
        lang_enforce = ""

    # ── Block 2: Conversation continuity ──────────────────────────────────────
    if has_hist:
        continuity = (
            "CONVERSATION CONTINUITY:\n"
            "The prior conversation turns are included below in the message list. "
            "The user's current message is a continuation of this conversation. "
            "It may be a follow-up question, a request for clarification, or a new angle "
            "on the same topic discussed before. "
            "Always interpret the current message in the context of what was already discussed. "
            "Do NOT treat it as a standalone, context-free query. "
            "Do NOT repeat information you already provided in this conversation "
            "unless the user explicitly asks you to repeat or re-explain it.\n\n"
        )
    else:
        continuity = ""

    # ── Block 3: RAG context (mandatory grounding) ────────────────────────────
    if context_chunks:
        ctx = (
            "══ REFERENCE MATERIAL — MANDATORY GROUNDING ══\n"
            "The following passages were retrieved specifically for this question. "
            "Your answer MUST be directly grounded in this material. "
            "Use the specific details, steps, timeframes, and clinical recommendations "
            "from these passages in your response. "
            "Do NOT replace this content with generic advice when it already covers the topic. "
            "Supplement with standard clinical dental knowledge only where the reference "
            "has clear gaps. Do NOT include home remedies or folk advice unless they appear "
            "in the reference material below.\n\n"
            + "\n---\n".join(context_chunks)
            + "\n══════════════════════════════════════════\n\n"
        )
    else:
        ctx = (
            "REFERENCE MATERIAL: None retrieved for this query.\n"
            "Answer from established clinical dental knowledge (standard protocols only). "
            "Do NOT include home remedies or folk advice.\n\n"
        )

    # ── Block 4: Format rules (Arabic and English parity) ─────────────────────
    if ar:
        if qtype == "instruction":
            fmt = (
                "FORMAT:\n"
                "• استخدم الرمز • فقط لكل نقطة — ممنوع الأرقام، ممنوع الشرطات.\n"
                "• كل نقطة: جملة واحدة كاملة وقابلة للتطبيق — محددة وسريرية، لا عامة ولا مبهمة.\n"
                "• عدد النقاط: من 4 إلى 6 فقط — لا أقل ولا أكثر.\n"
                "• ممنوع: عنوان للقائمة، جملة تمهيدية، جملة ختامية — "
                "ابدأ مباشرةً بأول نقطة بدون أي مقدمة.\n\n"
            )
        elif qtype == "symptom":
            fmt = (
                "FORMAT:\n"
                "• نثر مباشر فقط — ممنوع تماماً استخدام النقاط أو القوائم.\n"
                "• من 3 إلى 5 جمل. كل جملة تحمل معلومة سريرية حقيقية ومفيدة — "
                "لا حشو ولا تكرار.\n"
                "• اشرح السبب المحتمل وآلية حدوثه — لا تكتفِ بوصف ما قاله المريض.\n"
                "• اختم بجملة واحدة توصي بمراجعة طبيب الأسنان.\n\n"
            )
        else:
            fmt = (
                "FORMAT:\n"
                "• نثر مباشر فقط — ممنوع تماماً استخدام النقاط أو القوائم.\n"
                "• من 2 إلى 4 جمل، كل جملة دقيقة ومفيدة وتحمل معلومة حقيقية.\n"
                "• لا مقدمات عامة، لا تكرار للسؤال — اذهب مباشرةً للإجابة.\n\n"
            )
    else:
        if qtype == "instruction":
            fmt = (
                "FORMAT:\n"
                "• Use the • character only for each bullet. No numbers, no dashes.\n"
                "• Each bullet: one complete, actionable sentence — specific and clinical, "
                "not vague or generic.\n"
                "• Total bullets: 4 to 6. No fewer, no more.\n"
                "• No list heading, no introductory sentence, no closing sentence — "
                "begin directly with the first bullet, nothing before it.\n\n"
            )
        elif qtype == "symptom":
            fmt = (
                "FORMAT:\n"
                "• Plain prose only — no bullet points, no lists whatsoever.\n"
                "• 3 to 5 sentences. Each sentence must carry a distinct, useful clinical point — "
                "no padding, no repetition.\n"
                "• Explain the likely cause and the mechanism behind it — "
                "not just a restatement of the symptom.\n"
                "• End with one sentence recommending a dental consultation.\n\n"
            )
        else:
            fmt = (
                "FORMAT:\n"
                "• Plain prose only — no bullet points, no lists.\n"
                "• 2 to 4 sentences, each precise and informative.\n"
                "• No generic preamble, no question restatement — "
                "go directly to the answer.\n\n"
            )

    # ── Block 5: Core rules ────────────────────────────────────────────────────
    core = (
        "CORE RULES:\n"
        "• Explain likely causes and the clinical mechanism behind them.\n"
        "• You may explain: causes, clinical meaning, how conditions develop, "
        "what treatment generally involves, what a dental visit typically looks like.\n"
        "• You may NOT: prescribe for a specific patient, write a personalised treatment plan, "
        "or commit to a definitive diagnosis as certain.\n"
        "• Use 'typically', 'often', 'in most cases' for uncertain claims.\n"
        "• Do NOT claim insufficient information when standard clinical knowledge applies.\n"
        "• Do NOT ask the user any follow-up questions.\n"
        "• Do NOT open your response with filler phrases: 'Certainly!', 'Of course!', "
        "'Sure!', 'Great question!', 'بالتأكيد!', 'حسناً!', 'بكل سرور!' — "
        "start directly with the content.\n"
        "• Do NOT restate or paraphrase the user's question at the start of your answer.\n\n"
    )

    # ── Block 6: Scope ─────────────────────────────────────────────────────────
    scope = (
        "SCOPE: You are a dental health assistant only. "
        "If the question is clearly unrelated to oral or dental health, respond with exactly: "
        "'This is outside the scope of this oral health application.' (English) "
        "or 'هذا السؤال خارج نطاق تطبيق صحة الفم والأسنان.' (Arabic). Nothing more.\n\n"
    )

    # ── Block 7: Focus ─────────────────────────────────────────────────────────
    focus = (
        "FOCUS: Answer only what was asked — nothing more. "
        "Do not proactively mention x-rays, follow-up appointments, or additional procedures "
        "unless the user explicitly asked about them.\n\n"
    )

    # ── Block 8: Alarmism ──────────────────────────────────────────────────────
    alarmism = (
        "ALARMISM: Do not add 'if symptoms worsen, see a dentist' or similar escalation warnings "
        "unless the user explicitly describes uncontrolled bleeding, fever combined with dental pain, "
        "or difficulty breathing or swallowing. Omit all escalation language for other questions.\n\n"
    )

    # ── Block 9: Differential ──────────────────────────────────────────────────
    differential = (
        "DIFFERENTIAL: When symptoms are vague or could have multiple causes, briefly name the "
        "2–3 most likely possibilities before elaborating. Never commit to a single diagnosis "
        "from a vague description. Use language like 'this is often caused by...', "
        "'common causes include...'.\n\n"
    )

    # ── Block 10: Medication ───────────────────────────────────────────────────
    medication = (
        "MEDICATION: You may freely discuss antibiotics commonly used in dentistry, "
        "alternatives for penicillin allergy, general safe OTC dosage ranges for ibuprofen "
        "and paracetamol, medication side effects, IV vs oral antibiotic use, and "
        "weight-based dosage estimation for reference. "
        "Do NOT prescribe for a specific patient or write a personalised treatment plan.\n\n"
    )

    # ── Block 11: Language and terminology ────────────────────────────────────
    if ar:
        lang = (
            "TERMINOLOGY — use these exact Arabic terms. No substitutions, no English alternatives:\n"
            "قطعة شاش = gauze  |  حشوة = filling  |  علاج العصب = root canal  |  "
            "تاج = crown  |  تنظيف الجير = scaling  |  تبييض = whitening  |  "
            "خيط الأسنان = dental floss  |  زرعة = implant  |  خلع = extraction  |  "
            "ضرس العقل = wisdom tooth  |  لثة = gum  |  تسوس = decay or cavity  |  "
            "البلاك = plaque  |  جير الأسنان = tartar or calculus  |  "
            "تقليح = root planing or deep cleaning  |  "
            "طقم أسنان = denture  |  جسر = dental bridge  |  قشرة = veneer  |  "
            "تخدير أو بنج = anaesthesia  |  خراج = abscess  |  لب السن = dental pulp  |  "
            "مينا الأسنان = enamel  |  انحسار اللثة = gum recession\n\n"
            "TONE: Natural Gulf Arabic. A knowledgeable, warm dental professional speaking "
            "directly to a patient — clear and calm, not academic, not colloquial slang.\n\n"
        )
    else:
        lang = (
            "TONE: Plain, clear English for a non-dental adult. "
            "Warm and knowledgeable — like a trusted professional who explains clearly "
            "without being clinical or distant. Define technical terms briefly on first use.\n\n"
        )

    # ── Block 12: Forbidden patterns ──────────────────────────────────────────
    forbidden = (
        "STRICTLY FORBIDDEN — violating any of these is a hard failure:\n"
        "• Numbered or lettered lists (1. 2. 3. / a. b. c.)\n"
        "• Markdown headings (# ## ###) or bold text used as section titles\n"
        "• Hyphens as bullet markers (- item) — the only permitted bullet character is •\n"
        "• Summary or conclusion phrases: 'In summary', 'To summarize', 'Overall', "
        "'In conclusion', 'خلاصة', 'خلاصة القول', 'ختاماً', 'باختصار', 'في الختام'\n"
        "• Filler openers: 'Certainly!', 'Of course!', 'Great!', 'Sure!', "
        "'بالتأكيد', 'بكل سرور', 'حسناً', 'بالطبع'\n"
        "• Restating or paraphrasing the user's question in any form\n"
        "• Any text appended after the final content sentence\n"
        "Your response ends at the last relevant sentence. Nothing follows it.\n\n"
    )

    # ── Assemble in priority order ─────────────────────────────────────────────
    system = (
        lang_enforce    # 1. Language lock — must be first for Arabic
        + "You are ORA, a dental health assistant for general patients.\n"
          "Provide accurate, specific, clinically grounded dental information.\n\n"
        + core          # 2. Core behavioural rules
        + scope         # 3. Domain boundary
        + focus         # 4. Answer scope
        + fmt           # 5. Output structure
        + alarmism      # 6. Escalation control
        + differential  # 7. Differential guidance
        + medication    # 8. Medication rules
        + lang          # 9. Language / tone / terminology
        + forbidden     # 10. Hard prohibitions
        + continuity    # 11. Conversation context flag (if applicable)
        + ctx           # 12. RAG material last = highest recency weight
    )

    msgs = [{"role": "system", "content": system}]
    if history:
        for turn in history[-8:]:
            role    = turn.get("role", "")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": q})

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            max_tokens=MAX_GPT_TOKENS,
            temperature=0.3,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        log.error(f"GPT error: {e}")
        return (
            "عذرًا، حدث خطأ. يُنصح بمراجعة طبيب أسنان."
            if ar else
            "Sorry, an error occurred. Please consult a licensed dentist."
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION MANAGEMENT
#  In-memory store keyed by session UUID.
#  Each entry holds the conversation history and a last-active timestamp.
#  Sessions expire after SESSION_TTL_SECONDS of inactivity.
#  For production with multiple workers use Redis instead of this dict.
# ─────────────────────────────────────────────────────────────────────────────

_sessions: dict[str, dict] = {}


def _get_history(session_id: str) -> list:
    """
    Returns the stored history list for a session, or [] if it doesn't exist
    or has expired. Expired sessions are deleted on access (lazy cleanup).
    """
    entry = _sessions.get(session_id)
    if not entry:
        return []
    if time.time() - entry["last_active"] > SESSION_TTL_SECONDS:
        log.info(f"Session {session_id} expired — clearing.")
        del _sessions[session_id]
        return []
    return entry["history"]


def _save_history(session_id: str, history: list) -> None:
    """
    Persists history for a session and refreshes its TTL.
    Trims to SESSION_MAX_TURNS * 2 messages to cap memory usage.
    """
    _sessions[session_id] = {
        "history":     history[-(SESSION_MAX_TURNS * 2):],
        "last_active": time.time(),
    }


def _clear_history(session_id: str) -> None:
    _sessions.pop(session_id, None)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE QUERY DISPATCHER
#  Single function that owns the routing logic and history writes.
#  All code paths — dataset, GPT, social, refusal — write to history here.
#  Nothing bypasses this function when serving a real request.
# ─────────────────────────────────────────────────────────────────────────────

def dispatch(q: str, history: list) -> dict:
    """
    Routes a query through all filters and returns a result dict:
      answer   str   — the response text
      refs     list  — source document titles (may be empty)
      source   str   — which path answered: social | treatment_refusal |
                       scope_refusal | dataset:<idx> | gpt
      is_error bool  — True only if an unhandled exception occurred
      debug    dict  — RAG debug info (empty for non-GPT paths)
    """
    answer   = ""
    refs     = []
    source   = ""
    is_error = False
    debug    = {}

    try:
        # ── 1. Treatment / prescription refusal ──────────────────────────────
        if is_treatment_request(q):
            answer = refusal_treatment(q)
            source = "treatment_refusal"

        # ── 2. Social / conversational shortcut ───────────────────────────────
        elif is_social_exchange(q):
            answer = social_response(q)
            source = "social"

        # ── 3. Out-of-scope guard ─────────────────────────────────────────────
        # history_has_dental_context acts as a tiebreaker:
        # a follow-up like "how long does the bleeding last?" that arrives after
        # a dental exchange should never be refused as out-of-scope.
        elif is_out_of_scope(q) and not history_has_dental_context(history):
            answer = refusal_scope(q)
            source = "scope_refusal"

        else:
            # ── 4. Embed once, reuse for both dataset and RAG ─────────────────
            try:
                qv = embed(q)
            except Exception as e:
                log.error(f"Embedding failed: {e}")
                ar     = is_ar(q)
                answer = (
                    "عذرًا، حدث خطأ في المعالجة. يُنصح بمراجعة طبيب أسنان."
                    if ar else
                    "Sorry, a processing error occurred. Please consult a licensed dentist."
                )
                source   = "error"
                is_error = True
                return {
                    "answer": answer, "refs": refs,
                    "source": source, "is_error": is_error, "debug": debug,
                }

            # ── 5. Static dataset shortcut ────────────────────────────────────
            match, score, ar, idx = dataset_match(q, qv)
            if match and score >= SIM_THRESHOLD:
                answer = match["ar_a"] if ar else match["en_a"]
                source = f"dataset:{idx}"

            else:
                # ── 6. RAG + GPT ──────────────────────────────────────────────
                ctx, refs, off_topic, debug = rag_retrieve(qv)

                # off_topic from RAG means Pinecone top score < MIN_RELEVANCE.
                # Apply scope refusal only when there is also no dental signal in
                # the query AND no dental context in the conversation history.
                if (off_topic
                        and not has_dental_signal(q)
                        and not history_has_dental_context(history)):
                    answer = refusal_scope(q)
                    source = "scope_refusal"
                else:
                    answer = gpt_style_answer(q, ctx, history)
                    source = "gpt"

    except Exception as e:
        log.error(f"Unhandled dispatch error: {e}")
        ar     = is_ar(q)
        answer = (
            "عذرًا، حدث خطأ. يُنصح بمراجعة طبيب أسنان."
            if ar else
            "Sorry, an error occurred. Please consult a licensed dentist."
        )
        source   = "error"
        is_error = True

    return {
        "answer":   answer,
        "refs":     refs,
        "source":   source,
        "is_error": is_error,
        "debug":    debug,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FASTAPI APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:    str            = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str]  = None


class ChatResponse(BaseModel):
    session_id:   str
    answer:       str
    refs:         list[str]
    source:       str
    is_error:     bool
    history_turns: int   # number of completed turns in this session — useful for the frontend


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("ORA API ready.")
    yield
    log.info("ORA API shutting down.")


app = FastAPI(
    title="ORA Dental Assistant API",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Lightweight liveness check."""
    return {
        "status":  "ok",
        "model":   MODEL,
        "index":   PINECONE_INDEX,
        "sessions": len(_sessions),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint.

    Send a message and an optional session_id.
    On the first turn, omit session_id — the server creates one and returns it.
    On every subsequent turn, send back the same session_id to maintain context.

    The server writes every turn (including refusals and social replies) to the
    session history so follow-up questions always have full context.
    """
    q = req.message.strip()
    if not q:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())
    history    = _get_history(session_id)

    log.info(
        f"CHAT  session={session_id}  "
        f"history_turns={len(history) // 2}  "
        f"q={q[:80]!r}"
    )

    result = dispatch(q, history)

    # Write this turn to history for ALL response paths.
    # This is critical — if dataset or social replies are not stored here,
    # the next message arrives without context for what was said.
    history.append({"role": "user",      "content": q})
    history.append({"role": "assistant", "content": result["answer"]})
    _save_history(session_id, history)

    log.info(
        f"DONE  session={session_id}  source={result['source']}  "
        f"history_turns={len(history) // 2}  "
        f"refs={result['refs']}"
    )

    return ChatResponse(
        session_id    = session_id,
        answer        = result["answer"],
        refs          = result["refs"],
        source        = result["source"],
        is_error      = result["is_error"],
        history_turns = len(history) // 2,
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clears the conversation history for a session.
    Call this when the user explicitly starts a new conversation.
    """
    _clear_history(session_id)
    log.info(f"Session cleared: {session_id}")
    return {"cleared": session_id}


@app.get("/session/{session_id}/history")
async def get_session_history(session_id: str):
    """
    Returns the raw history for a session.
    Useful for debugging and frontend state recovery.
    """
    history = _get_history(session_id)
    return {
        "session_id": session_id,
        "turns":      len(history) // 2,
        "history":    history,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI  (python ora.py — for local testing only)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    session_id = str(uuid.uuid4())
    history: list = []
    print(f"ORA CLI  —  session {session_id}  —  type 'quit' to exit\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            sys.exit(0)

        if not q:
            continue
        if q.lower() in ("quit", "exit", "bye"):
            print("ORA: Take care!")
            sys.exit(0)

        result = dispatch(q, history)

        # Write to local history (mirrors what the server does)
        history.append({"role": "user",      "content": q})
        history.append({"role": "assistant", "content": result["answer"]})
        # Trim to cap
        history = history[-(SESSION_MAX_TURNS * 2):]

        print(f"\nORA [{result['source']}]: {result['answer']}")
        if result["refs"]:
            print(f"  Sources: {', '.join(result['refs'])}")
        if result["debug"]:
            print(f"  Debug: {result['debug']}")
        print()

# ─────────────────────────────────────────────────────────────────────────────
#  To run the API server:
#      pip install fastapi uvicorn[standard] openai pinecone-client
#      uvicorn ora:app --host 0.0.0.0 --port 8000 --reload
# ─────────────────────────────────────────────────────────────────────────────
