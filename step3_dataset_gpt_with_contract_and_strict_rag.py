import re
import math
import logging
from openai import OpenAI
from pinecone import Pinecone

logging.basicConfig(
    level=logging.INFO,
    format="[ORA %(levelname)s] %(message)s"
)
log = logging.getLogger("ora")

# ========= CONFIG =========
MODEL                = "gpt-5.2"
EMBED_MODEL          = "text-embedding-3-large"

SIM_THRESHOLD        = 0.70
PINECONE_INDEX       = "oraapp111"

TOP_K_RAW            = 5
TOP_K_FINAL          = 3
MIN_AUTHORITY        = 0.65
MAX_GPT_TOKENS       = 450
MIN_REF_DISPLAY      = 0.72

# [Fix 1/2] Lowered from 0.45 → 0.32.
# The previous threshold was blocking short but valid dental questions
# ("pain that comes and goes") and single-word follow-ups ("Is it dangerous?")
# that have naturally low vector similarity to specific documents
# but are still semantically within the dental domain.
MIN_RELEVANCE        = 0.32

PINECONE_TEXT_FIELD  = "chunk_text"

client = OpenAI()
pc     = Pinecone()
index  = pc.Index(PINECONE_INDEX)


# ========= DATASET =========
DATASET = [
    {
        "field": ["Periodontics"],
        "en_q": "My gums bleed when I brush, what should I do?",
        "en_a": (
            "Bleeding gums are usually a sign of gingival inflammation — the gum tissue "
            "is irritated by plaque, a sticky bacterial film that builds up on teeth. "
            "Mild cases cause redness and slight bleeding; advanced cases can involve swelling "
            "and recession. The key is consistent oral hygiene: brush twice daily with correct "
            "technique, and floss daily to clean between teeth where brushing cannot reach. "
            "If plaque is left undisturbed it hardens into calculus, which only a dentist can remove. "
            "A professional cleaning and checkup every six months helps prevent progression."
        ),
        "ar_q": "نزيف اللثة عادة ما يكون مؤشرا على وجود التهاب",
        "ar_a": (
            "نزيف اللثة في الغالب علامة على التهابها بسبب تراكم البلاك — طبقة لزجة من البكتيريا "
            "تتكوّن على الأسنان. في الحالات البسيطة يظهر احمرار ونزيف خفيف، وفي الحالات المتقدمة "
            "قد يصير فيه تورم. الحل هو الاهتمام بنظافة الفم: تفريش منتظم مرتين يومياً بطريقة صحيحة، "
            "واستخدام خيط الأسنان يومياً لتنظيف ما بين الأسنان. البلاك إذا بقي يتكلّس ويصير جيراً "
            "ما يزيله إلا التنظيف عند طبيب الأسنان. يُنصح بزيارة دورية كل ستة أشهر للفحص والتنظيف."
        )
    },
    {
        "field": ["Implant", "Periodontics"],
        "en_q": "I had a dental implant and notice bluish discoloration on my gum, is that normal?",
        "en_a": (
            "Bluish discoloration near an implant is common in patients with naturally thin gum "
            "tissue — when the tissue is thin enough, the metallic components show through. "
            "This is an anatomical variation, not a sign of infection or implant failure. "
            "It is primarily an aesthetic concern. A periodontist or implant specialist can assess "
            "whether soft tissue correction would be appropriate."
        ),
        "ar_q": "بعد ما سويت زرعة صار عندي لون أزرق في اللثة",
        "ar_a": (
            "اللون الأزرق بالقرب من الزرعة يصير في الغالب عند من تكون لثتهم رقيقة بطبيعتها، "
            "وهذا يخلي المكونات المعدنية للزرعة تبيّن من خلال النسيج. هذا تغيّر تشريحي طبيعي "
            "وما هو علامة على فشل الزرعة أو التهاب — في أغلب الأحيان هو مسألة مظهرية بس. "
            "لو كان يضايقك، تقدر تراجع طبيب متخصص في أمراض اللثة أو زراعة الأسنان ليقيّم "
            "إذا كان التدخل مناسب."
        )
    },
    {
        "field": ["Restorative Dentistry"],
        "en_q": "I had a filling and now I feel pain only when biting, what does it mean?",
        "en_a": (
            "Pain only when biting after a filling usually means the restoration is slightly too high — "
            "called high occlusion. The filling contacts the opposing tooth before the rest of the bite "
            "does, creating localized pressure when chewing. Unlike general sensitivity, this pain is "
            "specific to biting force. A quick adjustment at your dentist appointment resolves it."
        ),
        "ar_q": "بعد الحشوة صار عندي ألم عند العض فقط",
        "ar_a": (
            "الألم عند العض فقط بعد الحشوة يعني في الغالب إن الحشوة مرتفعة شوي عن مستواها الصحيح — "
            "يعني تلامس السن المقابل قبل بقية الأسنان، وهذا يسبب ضغطاً موضعياً أثناء المضغ. "
            "هذا يختلف عن الحساسية العامة لأن الألم مرتبط بقوة العض تحديداً. "
            "تعديل بسيط عند طبيب الأسنان يحل المشكلة."
        )
    },
    {
        "field": ["Endodontics"],
        "en_q": "I experienced severe tooth pain that disappeared without treatment. What does it mean?",
        "en_a": (
            "When severe tooth pain disappears on its own, it often indicates the pulp — "
            "the nerve-containing tissue inside the tooth — has lost vitality. "
            "As the nerve dies, pain signals stop, but the underlying bacterial infection continues. "
            "This is not recovery; the infection can spread beyond the root if left untreated. "
            "An evaluation is necessary even when there is no pain."
        ),
        "ar_q": "اختفى الألم الشديد في السن",
        "ar_a": (
            "اختفاء الألم من تلقاء نفسه ما يعني الشفاء. في الغالب يعني إن اللب — "
            "نسيج العصب داخل السن — فقد حيويته، فلما يموت العصب تتوقف الإشارات المؤلمة، "
            "بس العدوى البكتيرية تستمر. لو تُركت دون علاج يمكن تمتد العدوى خارج جذر السن. "
            "المراجعة ضرورية حتى في غياب الألم."
        )
    },
    {
        "field": ["Endodontics"],
        "en_q": "What does it mean that a tooth rots?",
        "en_a": (
            "The term 'rotting' is informal and often misunderstood. What actually happens is that "
            "bacteria invade the inner layers of the tooth — the dentin or pulp — usually following "
            "untreated decay or trauma. The tooth does not literally decompose, but ongoing bacterial "
            "activity causes progressive structural damage. The appropriate treatment depends on how "
            "far the infection has progressed."
        ),
        "ar_q": "هل تعني كلمة تعفن السن أن السن ميت؟",
        "ar_a": (
            "مصطلح التعفن مو دقيق طبياً. اللي يصير فعلاً هو إن البكتيريا تخترق الطبقات الداخلية "
            "للسن — الدنتين أو اللب — عادةً بسبب تسوس متقدم أو صدمة. السن ما تتحلل حرفياً، "
            "بس النشاط البكتيري يسبب تلفاً تدريجياً في بنيتها. "
            "العلاج المناسب يعتمد على مدى تقدم العدوى."
        )
    }
]


# ========= SCOPE & SAFETY =========
def is_ar(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


_NON_DENTAL_BLOCK = [
    # English
    "capital of", "weather forecast", "stock price", "bitcoin", "crypto",
    "movie", "film", "music", "song", "recipe", "cooking", "bake",
    "football", "basketball", "soccer", "rugby", "politics", "president",
    "prime minister", "election", "travel", "hotel", "flight", "airline",
    "restaurant", "penthouse", "real estate", "property investment",
    "porsche", "ferrari", "bmw", "toyota", "mercedes", "audi", "honda", "tesla",
    "car model", "vehicle specs", "phone", "iphone", "android",
    "laptop", "computer programming", "software development",
    # Arabic
    "عاصمة", "الطقس", "أسهم", "بيتكوين", "عملة رقمية", "فيلم",
    "موسيقى", "وصفة طبخ", "كرة القدم", "مباراة رياضية", "سياسة",
    "رئيس الدولة", "انتخابات", "سفر للخارج", "فندق", "طيران", "مطعم",
    "بورش", "فيراري", "جوال", "هاتف ذكي", "برمجة", "شقة فارهة", "عقار"
]

# [Fix 1/2] Expanded with pain/symptom terms and additional Arabic dental vocabulary.
# General symptom terms (pain, swelling, bleed) are included because in this application's
# context, they are almost always dental-related. Short follow-ups like
# "Is it dangerous?" are handled by the history context check, not by this list.
_DENTAL_SIGNALS = [
    # English - conditions and procedures
    "tooth", "teeth", "gum", "gums", "mouth", "oral", "dental", "dentist",
    "jaw", "bite", "biting", "cavity", "filling", "crown", "implant",
    "root canal", "braces", "plaque", "enamel", "dentin", "pulp",
    "extraction", "wisdom", "molar", "incisor", "premolar", "canine",
    "veneer", "whitening", "bleaching", "floss", "toothbrush",
    "gingiv", "periodon", "endodon", "orthodon", "calculus", "tartar",
    "abscess", "deep cleaning", "scaling", "anesthesia", "x-ray",
    "panoramic", "space maintainer", "retainer", "denture", "bridge",
    "alveolar", "pericoronitis", "caries", "pulpitis", "gingivitis",
    "periodontitis", "bone graft", "sinus lift", "fluoride", "sealant",
    "occlusion", "malocclusion", "tmj", "clench", "grind", "bruxism",
    # English - general symptoms valid in dental context
    "pain", "ache", "aching", "hurt", "hurts", "hurting",
    "swelling", "swollen", "swell", "bleed", "bleeding",
    "sensitive", "sensitivity", "numb", "numbness", "sore", "soreness",
    "discomfort", "throbbing",
    # Arabic - conditions and procedures
    "سن", "أسنان", "لثة", "فم", "طبيب الأسنان", "طبيب أسنان",
    "حشوة", "حشوات", "تاج", "زرعة", "علاج العصب", "تقويم",
    "تبييض", "جير", "ضرس", "نزيف اللثة", "حساسية الأسنان",
    "البلاك", "خيط الأسنان", "التهاب اللثة", "تسوس", "خلع",
    "تنظيف عميق", "تقليح", "تلميع", "حافظ مسافة", "مثبت",
    "طقم أسنان", "جسر أسنان", "تخدير", "التهاب حول التاج",
    "خراج", "كيس", "فك", "عظم السنخي", "تلاصق الأسنان",
    "قشرة", "ابتسامة هوليود", "رحى", "ناب", "قاطعة",
    "حكة اللثة", "انحسار اللثة", "فلور", "سيلنت",
    # Arabic - general symptoms valid in dental context
    "ألم", "يؤلم", "يؤلمني", "وجع", "تورم", "خدر",
    "نزيف", "حساسية", "إحساس", "أشعر", "لاحظت",
]


# [Fix 3] Now uses narrow regex patterns targeting only directive prescribing language.
# This approach is symmetric — the same function handles both Arabic and English input.
# Educational questions about medications (side effects, alternatives, dosage ranges,
# commonly used antibiotics) are no longer caught.
_PRESCRIPTION_PATTERNS = [
    # English — direct prescribing directive
    r"prescribe\s+me\b",
    r"write\s+(me\s+)?a\s+prescription",
    r"give\s+me\s+(a\s+specific\s+)?(prescription|medicine\s+for\s+my\s+case)",
    r"make\s+(me\s+)?a\s+(treatment|care)\s+plan\s+for\s+(me|my)",
    r"tell\s+me\s+exactly\s+what\s+(drug|medication|antibiotic)\s+to\s+take",
    r"diagnose\s+(my\s+case|what\s+i\s+have|me)\s+exactly",
    # Arabic — direct prescribing directive
    r"وصف\s+لي\s+دواء",
    r"وصّف\s+لي",
    r"اعطني\s+وصفة\s+طبية",
    r"أعطني\s+وصفة",
    r"اكتب\s+لي\s+وصفة",
    r"ابي\s+وصفة\s+طبية",
    r"أبغى\s+وصفة",
    r"شخّص\s+حالتي\s+بالضبط",
    r"شخص\s+حالتي",
]

def is_treatment_request(q: str) -> bool:
    ql = q.lower()
    return any(re.search(p, ql) for p in _PRESCRIPTION_PATTERNS)

def is_out_of_scope(q: str) -> bool:
    ql = q.lower()
    return any(b in ql for b in _NON_DENTAL_BLOCK)

def has_dental_signal(text: str) -> bool:
    tl = text.lower()
    return any(sig in tl for sig in _DENTAL_SIGNALS)

def history_has_dental_context(history: list) -> bool:
    """
    [Fix 2] Returns True if any prior turn contains dental content.
    When True, the current follow-up question inherits dental scope and must
    NOT be independently re-evaluated for scope. A short follow-up like
    "Is it dangerous?" has no dental keywords on its own but is implicitly
    dental because of what preceded it.
    """
    for turn in history:
        content = turn.get("content", "")
        if has_dental_signal(content):
            return True
    return False

def refusal_treatment(q: str) -> str:
    return (
        "ما أقدر أوصف أدوية أو أقدم تشخيصاً مباشراً. "
        "للحصول على العلاج المناسب، يُنصح بمراجعة طبيب أسنان."
        if is_ar(q) else
        "I cannot prescribe medication or provide a direct diagnosis. "
        "Please consult a licensed dentist."
    )

def refusal_scope(q: str) -> str:
    # [Fix 7] Short, neutral, non-defensive. One sentence only.
    return (
        "هذا السؤال خارج نطاق تطبيق صحة الفم والأسنان."
        if is_ar(q) else
        "This question is outside the scope of this oral health application."
    )


# ========= QUESTION TYPE =========
def detect_question_type(q: str) -> str:
    ql = q.lower()
    _instruction_signals = [
        "how to", "how do i", "what should i do", "aftercare",
        "after surgery", "after extraction", "after root canal", "after implant",
        "after filling", "after procedure", "steps", "instructions",
        "what to do after", "can i eat", "can i drink", "when can i",
        "كيف", "ماذا أفعل", "ماذا يجب", "تعليمات", "خطوات",
        "بعد العملية", "بعد الخلع", "بعد التركيب", "بعد الزرعة",
        "بعد الحشوة", "بعد التنظيف", "ايش أسوي", "كيف أعتني",
        "هل أقدر آكل", "هل أقدر أشرب", "متى أقدر", "ماذا أفعل بعد"
    ]
    _symptom_signals = [
        "pain", "hurt", "hurts", "bleed", "bleeding", "swell", "swelling",
        "ache", "sensitive", "sensitivity", "discomfort", "feel", "notice",
        "throbbing", "numb", "numbness",
        "ألم", "يؤلم", "يؤلمني", "نزيف", "تورم", "حساسية",
        "أشعر", "لاحظت", "وجع", "يحس", "خدر"
    ]
    for s in _instruction_signals:
        if s in ql:
            return "instruction"
    for s in _symptom_signals:
        if s in ql:
            return "symptom"
    return "informational"


# ========= EMBEDDING =========
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


# ========= INTENT SHORTCUTS =========
INTENT_PHRASES = {
    0: ["gum bleed", "bleeding gums", "نزيف اللثة", "اللثة تنزف", "لثتي تنزف"],
    1: ["blue gum", "implant color", "زرعة", "زرقة اللثة"],
    2: ["pain when biting", "after filling", "ألم عند العض", "بعد الحشوة"],
    3: ["pain disappeared", "اختفى الألم", "راح الألم"],
    4: ["tooth rotting", "تعفن السن", "السن ميت"]
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


# ========= RAG RETRIEVAL =========
def _extract_text(md: dict) -> str:
    for field in [PINECONE_TEXT_FIELD, "content", "chunk", "passage", "body"]:
        val = md.get(field, "")
        if val:
            log.debug(f"RAG text field='{field}' len={len(str(val))}")
            return str(val).strip()
    log.warning(f"RAG: no text found. Available fields: {list(md.keys())}")
    return ""

def rag_retrieve(qv: list, expected_fields=None):
    """
    Returns (context_chunks, display_titles, is_off_topic, debug_info).

    debug_info is a dict intended for server-side logging and the API's
    hidden _debug field. It gives you real-time confirmation that RAG
    content is being retrieved and injected — see the verification guide below.
    """
    res     = index.query(vector=qv, top_k=TOP_K_RAW, include_metadata=True)
    matches = res.get("matches", [])

    debug_info = {
        "pinecone_match_count": len(matches),
        "top_score": None,
        "chunks_injected": 0,
        "authority_filter_dropped": 0,
        "fallback_used": False,
    }

    if not matches:
        log.info("RAG: no matches from Pinecone")
        return [], [], False, debug_info

    top_score = float(matches[0].get("score", 0))
    debug_info["top_score"] = round(top_score, 4)
    log.info(f"RAG top_score={top_score:.3f} threshold={MIN_RELEVANCE}")

    if top_score < MIN_RELEVANCE:
        log.info("RAG: off-topic — score below MIN_RELEVANCE")
        return [], [], True, debug_info

    strong = []
    for m in matches:
        score = float(m.get("score", 0))
        md    = m.get("metadata", {})

        auth = float(md.get("authority_score", -1))
        if auth == -1:
            # [Fix 5] Warn if authority_score is missing entirely — this would silently
            # drop all chunks. If you see this warning consistently, the metadata schema
            # in Pinecone does not include this field and MIN_AUTHORITY should be disabled.
            log.warning(f"RAG: 'authority_score' missing. Fields present: {list(md.keys())}")

        if auth != -1 and auth < MIN_AUTHORITY:
            debug_info["authority_filter_dropped"] += 1
            continue

        specialty = str(md.get("specialty", "")).lower()
        if expected_fields and not any(f.lower() in specialty for f in expected_fields):
            continue

        text = _extract_text(md)
        strong.append({"title": md.get("title", ""), "text": text, "score": score})

    # [Fix 5] If authority/specialty filtering drops everything, fall back to raw results.
    # This prevents silent empty context — the model would otherwise answer from base
    # knowledge with no indication that RAG failed.
    if not strong and matches:
        log.warning("RAG: all chunks filtered — falling back to unfiltered top results")
        debug_info["fallback_used"] = True
        for m in matches[:TOP_K_FINAL]:
            text = _extract_text(m.get("metadata", {}))
            strong.append({
                "title": m.get("metadata", {}).get("title", ""),
                "text": text,
                "score": float(m.get("score", 0))
            })

    seen, unique = set(), []
    for s in strong:
        key = s["title"] or s["text"][:60]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    context_chunks = [s["text"] for s in unique if s["text"]][:TOP_K_FINAL]
    display_titles = [
        s["title"] for s in unique
        if s["title"] and s["score"] >= MIN_REF_DISPLAY
    ][:TOP_K_FINAL]

    debug_info["chunks_injected"] = len(context_chunks)
    log.info(
        f"RAG: chunks_injected={len(context_chunks)} "
        f"authority_dropped={debug_info['authority_filter_dropped']} "
        f"fallback={debug_info['fallback_used']} "
        f"refs_displayed={len(display_titles)}"
    )

    return context_chunks, display_titles, False, debug_info


# ========= GPT GENERATION =========
# [Fix 8] Pre-built constant — format prohibitions are now a fixed string,
# not constructed inside the function. This ensures they are always identical
# across language paths and question types (eliminates the symmetry risk).
_FORBIDDEN = (
    "STRICTLY FORBIDDEN — do not use any of these in your response:\n"
    "- Numbered lists (1. 2. 3.)\n"
    "- Markdown headings (## / ### / bold titles)\n"
    "- Bold text (**word** or __word__)\n"
    "- Asterisk bullets (* item)\n"
    "- Section-divider colons used as headers\n"
    "- Summary or conclusion paragraphs of any kind "
    "(e.g. 'In summary', 'To summarize', 'خلاصة', 'خلاصة القول', 'ختاماً', 'باختصار')\n"
    "Your answer must end naturally after the last relevant point — no closing remarks.\n"
)

def gpt_style_answer(q: str, context_chunks=None, history=None) -> str:
    ar    = is_ar(q)
    qtype = detect_question_type(q)

    # --- RAG context ---
    if context_chunks:
        ctx = (
            "\n\nREFERENCE MATERIAL — this is your primary source:\n"
            + "\n---\n".join(context_chunks)
            + "\n\n"
            "You may fill gaps using only clinical facts standard in accredited dental school "
            "curricula. Do NOT include home remedies (tea bags, clove oil as standalone treatment, "
            "saline rinse as the only post-op instruction, etc.) unless they appear in the reference "
            "material above. If unsure whether a fact meets this standard, omit it.\n"
        )
    else:
        ctx = (
            "\n\nNo reference material retrieved. Answer from established clinical dental "
            "knowledge — standard textbook protocols only. "
            "Do NOT include home remedies or folk-sourced advice.\n"
        )

    # --- Format ---
    if qtype == "instruction":
        fmt = "FORMAT: Hyphen bullet points ( - ) ONLY. Each point is one clear, actionable sentence.\n"
        length = "LENGTH: 4 to 6 bullet points. No more, no fewer.\n"
        disclaimer = ""
    elif qtype == "symptom":
        fmt = "FORMAT: Plain prose only. No bullet points.\n"
        length = "LENGTH: 3 to 5 sentences maximum.\n"
        disclaimer = "End with one sentence recommending a dental consultation.\n"
    else:
        fmt = "FORMAT: Plain prose only. No bullet points.\n"
        length = "LENGTH: 2 to 4 sentences maximum.\n"
        disclaimer = "End with one brief sentence recommending a dental consultation.\n"

    # --- Alarmism ---
    alarmism = (
        "ESCALATION WARNINGS: Do not append warnings like 'if pain worsens see a dentist' "
        "or 'if symptoms persist consult a professional'. Add urgent advice ONLY when the "
        "user explicitly describes: uncontrolled bleeding, fever alongside dental pain, "
        "or difficulty breathing or swallowing. For all other questions, omit escalation "
        "language entirely.\n"
    )

    # --- Language ---
    # [Fix 6/9] Arabic and English instructions are now symmetric in structure.
    # Both permit educational medication content. Both apply the same format rules.
    # Arabic tone is now anchored by a positive description of the target register
    # with concrete term examples, rather than a list of prohibitions.
    if ar:
        lang = (
            "LANGUAGE: Write in natural, professional Arabic with a Gulf register. "
            "The tone should feel like a knowledgeable dentist speaking plainly and "
            "warmly to a patient — not formal or academic (avoid 'حيث أن', 'إذ إن', "
            "'الجدير بالذكر', 'تجدر الإشارة'), and not slang. "
            "Natural Gulf expressions like 'بس', 'فعلاً', 'لما', 'يصير', 'فيه', "
            "'بدون ما', 'تقدر' are appropriate when they fit naturally — do not force them.\n"
            "Use these natural Arabic dental terms exactly as written:\n"
            "  قطعة شاش or شاش (gauze — NOT 'شاش ضاغط'),\n"
            "  حشوة (filling), علاج العصب (root canal),\n"
            "  تاج (crown), تنظيف الجير (scaling),\n"
            "  تبييض الأسنان (whitening), خيط الأسنان (floss),\n"
            "  زرعة (implant), البلاك (plaque), خلع (extraction).\n"
            "Do NOT write English dental terms mid-sentence as a style default. "
            "If an English term is genuinely helpful (the patient may have heard it at clinic), "
            "place it in parentheses after the Arabic on first use only.\n"
        )
    else:
        lang = (
            "LANGUAGE: Plain English for a non-dental adult. "
            "Briefly define any technical term the first time you use it. "
            "Warm, calm, professional tone — like a knowledgeable friend explaining clearly.\n"
        )

    # --- Diagnostic approach ---
    # [Fix 4] Prevents single-condition lock-in for vague symptoms.
    differential = (
        "DIFFERENTIAL APPROACH: When a symptom is vague or could stem from multiple causes, "
        "briefly state the 2-3 most likely possibilities before discussing them. "
        "Do not commit to one diagnosis from a vague description. "
        "For example, wisdom tooth pain could indicate pericoronitis, decay, "
        "impaction pressure on adjacent teeth, or referred pain — name the possibilities "
        "before elaborating. Use language like 'common causes include...', "
        "'this is often due to...', or 'the most likely explanation, though not the only one, is...'.\n"
    )

    # --- Medication education ---
    # [Fix 3] Explicit permission for educational medication content.
    # Symmetric — applies regardless of language.
    medication = (
        "MEDICATION EDUCATION: You MAY discuss the following without restriction:\n"
        "- Which antibiotics are commonly used for specific dental conditions\n"
        "- Alternatives for patients with penicillin allergy\n"
        "- General safe dosage ranges for OTC analgesics (ibuprofen, paracetamol)\n"
        "- Side effects of dental medications\n"
        "- When dentists use IV versus oral antibiotics\n"
        "- Weight-based dosage estimation (state it is for reference/verification only)\n"
        "You may NOT: prescribe medication for a specific patient's case, "
        "write a personal treatment plan, or state a definitive diagnosis.\n"
    )

    # --- Core system prompt ---
    system = (
        "You are ORA, a dental health assistant for general patients — not dental professionals.\n"
        "Give accurate, specific, and genuinely useful dental information in a clear human tone.\n"
        "\nCORE RULES:\n"
        "- Be specific. Explain the likely cause and the physiological mechanism behind it.\n"
        "- You MAY explain: likely causes, clinical meaning of a condition, how it develops, "
        "what treatment approaches generally exist, what a dentist visit for this typically involves.\n"
        "- You may NOT: prescribe specific medication, recommend dosages for a specific patient, "
        "or state a definitive diagnosis as certain.\n"
        "- Frame uncertain claims with 'typically', 'often', or 'in most cases'.\n"
        "- Do NOT claim insufficient information when general clinical dental knowledge exists.\n"
        "- Do NOT ask the user any follow-up questions.\n"
        "- If prior conversation is in the input, use it to understand context "
        "before answering the current message.\n"
        "\n"
        + fmt
        + length
        + disclaimer
        + alarmism
        + differential
        + medication
        + lang
        + _FORBIDDEN
        + ctx
    )

    msgs = [{"role": "system", "content": system}]
    if history:
        for turn in history:
            role    = turn.get("role", "")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": q})

    try:
        r = client.responses.create(
            model=MODEL,
            reasoning={"effort": "low"},
            input=msgs,
            max_output_tokens=MAX_GPT_TOKENS
        )
        return r.output_text.strip()

    except Exception as e:
        log.error(f"GPT error: {e}")
        return (
            "عذرًا، حدث خطأ في معالجة سؤالك. يُنصح بمراجعة طبيب أسنان مرخص."
            if ar else
            "Sorry, an error occurred. Please consult a licensed dentist."
        )


# ========= MAIN (CLI) =========
if __name__ == "__main__":
    q = input("> ").strip()

    if is_treatment_request(q):
        print(refusal_treatment(q))
        exit()

    if is_out_of_scope(q):
        print(refusal_scope(q))
        exit()

    qv = embed(q)
    match, score, ar, idx = dataset_match(q, qv)

    if match and score >= SIM_THRESHOLD:
        answer = match["ar_a"] if ar else match["en_a"]
        _, refs, _, debug = rag_retrieve(qv, match["field"])
    else:
        context_chunks, refs, is_off_topic, debug = rag_retrieve(qv)
        if is_off_topic and not has_dental_signal(q):
            print(refusal_scope(q))
            exit()
        answer = gpt_style_answer(q, context_chunks)

    print(answer)
    log.info(f"RAG debug: {debug}")
