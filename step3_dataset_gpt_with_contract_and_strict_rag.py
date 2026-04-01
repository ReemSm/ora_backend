import re
import math
from openai import OpenAI
from pinecone import Pinecone

# ========= CONFIG =========
MODEL                = "gpt-5.2"
EMBED_MODEL          = "text-embedding-3-large"

SIM_THRESHOLD        = 0.70
PINECONE_INDEX       = "oraapp111"

TOP_K_RAW            = 5       # [Fix 1] Reduced from 8: fewer Pinecone results → lower latency
TOP_K_FINAL          = 3
MIN_AUTHORITY        = 0.65
MAX_GPT_TOKENS       = 450     # [Fix 1/2] Reduced from 800: limits verbosity + reduces generation time
MIN_RELEVANCE        = 0.45    # [Fix 8/9] Below this score, query is likely off-topic
MIN_REF_DISPLAY      = 0.72    # [Fix 8]   Only show references above this confidence

PINECONE_TEXT_FIELD  = "text"

client = OpenAI()
pc     = Pinecone()
index  = pc.Index(PINECONE_INDEX)


# ========= DATASET =========
# [Fix 3] Arabic answers revised: professional warm register, no forced English mid-sentence,
# standard Arabic dental terms used throughout.
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
            "نزيف اللثة في الغالب علامة على التهابها، وهذا يحدث بسبب تراكم البلاك — "
            "طبقة لزجة من البكتيريا تتكوّن على الأسنان. في الحالات الخفيفة يظهر احمرار ونزيف بسيط، "
            "وفي الحالات المتقدمة قد يحدث تورم. الحل هو الحفاظ على نظافة الفم: التفريش مرتين يومياً "
            "بطريقة صحيحة، واستخدام خيط الأسنان يومياً لتنظيف ما بين الأسنان. "
            "إذا بقي البلاك يتكلّس ويصبح جيراً لا يُزال إلا بالتنظيف المهني عند طبيب الأسنان. "
            "يُنصح بزيارة دورية كل ستة أشهر للفحص والتنظيف."
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
            "اللون الأزرق بالقرب من الزرعة يحدث في الغالب عند من تكون لثتهم رقيقة بطبيعتها، "
            "مما يجعل المكونات المعدنية للزرعة تظهر من خلال النسيج. "
            "هذا تغيّر تشريحي طبيعي وليس علامة على فشل الزرعة أو التهاب، وهو في أغلب الأحيان "
            "مسألة جمالية. يمكن مراجعة طبيب متخصص في أمراض اللثة أو زراعة الأسنان إذا كان يُزعجك مظهرياً."
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
            "الألم عند العض فقط بعد الحشوة يعني في الغالب أن الحشوة مرتفعة قليلاً عن مستواها الصحيح، "
            "أي أنها تلامس السن المقابل قبل بقية الأسنان، مما يُحدث ضغطاً موضعياً أثناء المضغ. "
            "هذا يختلف عن الحساسية العامة لأن الألم مرتبط بقوة العض تحديداً. "
            "التعديل البسيط عند طبيب الأسنان يحل المشكلة."
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
            "اختفاء الألم الشديد من تلقاء نفسه لا يعني الشفاء. في الغالب يعني أن اللب — "
            "نسيج العصب داخل السن — فقد حيويته، فلما يموت العصب تتوقف الإشارات المؤلمة، "
            "لكن العدوى البكتيرية لا تزال مستمرة. إذا تُركت دون علاج يمكن أن تمتد العدوى "
            "خارج جذر السن. المراجعة ضرورية حتى في غياب الألم."
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
            "مصطلح التعفن غير دقيق طبياً. ما يحدث فعلياً هو اختراق البكتيريا للطبقات الداخلية "
            "للسن — الدنتين أو اللب — عادةً بسبب تسوس متقدم أو صدمة. السن لا تتحلل حرفياً، "
            "لكن النشاط البكتيري المستمر يُحدث تلفاً تدريجياً في بنيتها. "
            "العلاج المناسب يعتمد على مدى تقدم العدوى."
        )
    }
]


# ========= SCOPE & SAFETY =========
def is_ar(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))

# [Fix 9] Hard-block list — expanded to include car brands and general non-dental topics
# that the previous list missed (e.g. Porsche, phones, cooking).
_NON_DENTAL_BLOCK = [
    "capital of", "weather forecast", "stock price", "bitcoin", "crypto",
    "movie", "film", "music", "song", "recipe", "cooking", "bake",
    "football", "basketball", "soccer", "rugby", "politics", "president",
    "prime minister", "election", "travel", "hotel", "flight", "airline",
    "restaurant", "porsche", "ferrari", "bmw", "toyota", "mercedes",
    "audi", "honda", "tesla", "car model", "vehicle", "phone", "iphone",
    "android", "laptop", "computer programming", "software",
    # Arabic
    "عاصمة", "الطقس", "أسهم", "بيتكوين", "عملة رقمية", "فيلم",
    "موسيقى", "وصفة طبخ", "كرة القدم", "مباراة", "سياسة",
    "رئيس الدولة", "انتخابات", "سفر", "فندق", "طيران", "مطعم",
    "بورش", "فيراري", "سيارة", "جوال", "هاتف ذكي", "برمجة"
]

# [Fix 9] Positive dental signal list — used as secondary scope gate when
# Pinecone returns low scores. A question with no dental signal AND low
# retrieval score is treated as out-of-scope.
_DENTAL_SIGNALS = [
    "tooth", "teeth", "gum", "gums", "mouth", "oral", "dental", "dentist",
    "jaw", "bite", "biting", "cavity", "filling", "crown", "implant",
    "root canal", "braces", "plaque", "enamel", "dentin", "pulp",
    "extraction", "wisdom", "molar", "incisor", "veneer", "whitening",
    "floss", "toothache", "toothbrush", "bleeding gums", "sensitivity",
    "gingiv", "periodon", "endodon", "orthodon", "calculus", "tartar",
    # Arabic
    "سن", "أسنان", "لثة", "فم", "طبيب الأسنان", "حشوة", "تاج",
    "زرعة", "علاج العصب", "تقويم", "تبييض", "جير", "ضرس",
    "نزيف اللثة", "حساسية الأسنان", "البلاك", "خيط الأسنان",
    "التهاب اللثة", "تسوس", "خلع"
]

# [Fix 10] Prescription block is kept narrow — only direct medication requests.
# This avoids blocking legitimate informational questions about treatment options.
_PRESCRIPTION_BLOCK = [
    "prescribe", "write me a prescription", "which medication should i",
    "what antibiotic", "which antibiotic", "give me medicine", "what drug",
    "وصفة طبية", "اعطني دواء", "أي مضاد حيوي", "ما الدواء المناسب",
    "ما العلاج الدوائي", "شخّص حالتي بالضبط"
]

def is_treatment_request(q: str) -> bool:
    ql = q.lower()
    return any(x in ql for x in _PRESCRIPTION_BLOCK)

def is_out_of_scope(q: str) -> bool:
    ql = q.lower()
    return any(b in ql for b in _NON_DENTAL_BLOCK)

def has_dental_signal(q: str) -> bool:
    ql = q.lower()
    return any(sig in ql for sig in _DENTAL_SIGNALS)

def refusal_treatment(q: str) -> str:
    return (
        "لا يمكنني وصف أدوية أو تقديم تشخيص مباشر. يُنصح بمراجعة طبيب أسنان مرخص."
        if is_ar(q) else
        "I cannot prescribe medication or provide a direct diagnosis. Please consult a licensed dentist."
    )

def refusal_scope(q: str) -> str:
    return (
        "هذا السؤال خارج نطاق تطبيق صحة الفم والأسنان."
        if is_ar(q) else
        "This question is outside the scope of this oral health application."
    )


# ========= QUESTION TYPE =========
def detect_question_type(q: str) -> str:
    ql = q.lower()
    _instruction = [
        "how to", "how do i", "what should i do", "aftercare",
        "after surgery", "after extraction", "after root canal", "after implant",
        "after filling", "steps", "instructions", "what to do after",
        "can i eat", "can i drink", "when can i",
        "كيف", "ماذا أفعل", "ماذا يجب", "تعليمات", "خطوات",
        "بعد العملية", "بعد الخلع", "بعد التركيب", "بعد الزرعة",
        "بعد الحشوة", "ايش أسوي", "كيف أعتني", "هل أقدر آكل",
        "متى أقدر", "ماذا أفعل بعد"
    ]
    _symptom = [
        "pain", "hurt", "hurts", "bleed", "bleeding", "swell", "swelling",
        "ache", "sensitive", "sensitivity", "discomfort", "feel", "notice",
        "ألم", "يؤلم", "يؤلمني", "نزيف", "تورم", "حساسية",
        "أشعر", "لاحظت", "وجع", "يحس"
    ]
    for s in _instruction:
        if s in ql:
            return "instruction"
    for s in _symptom:
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

print("[ORA] Pre-computing dataset embeddings...")
_DS_EN_VECS = [embed(d["en_q"]) for d in DATASET]
_DS_AR_VECS = [embed(d["ar_q"]) for d in DATASET]
print(f"[ORA] Done — {len(DATASET)} entries cached.")


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
    """
    [Fix 8/10] Tries known field names in priority order.
    Logs the matched field name so you can verify PINECONE_TEXT_FIELD is correct.
    If you see the WARNING line, check the logged available fields and update the constant.
    """
    for field in [PINECONE_TEXT_FIELD, "content", "chunk", "passage", "body"]:
        val = md.get(field, "")
        if val:
            print(f"[ORA RAG] text field='{field}' len={len(str(val))}")
            return str(val).strip()
    print(f"[ORA RAG] WARNING: no text found. Available fields: {list(md.keys())}")
    return ""

def rag_retrieve(qv: list, expected_fields=None):
    """
    Returns (context_chunks, display_titles, is_off_topic).

    is_off_topic=True when the highest Pinecone score is below MIN_RELEVANCE.
    This catches queries that are semantically far from all dental content —
    such as questions about cars, general knowledge, etc. — even when they
    slip past the keyword blocklist.

    display_titles only includes references at or above MIN_REF_DISPLAY so that
    irrelevant or weakly-matched references are never shown to the user.
    """
    res     = index.query(vector=qv, top_k=TOP_K_RAW, include_metadata=True)
    matches = res.get("matches", [])

    if not matches:
        return [], [], False

    top_score = float(matches[0].get("score", 0))
    print(f"[ORA RAG] top_score={top_score:.3f} threshold={MIN_RELEVANCE}")

    # [Fix 9] Low top score → query is semantically off-topic
    if top_score < MIN_RELEVANCE:
        print("[ORA RAG] Score below MIN_RELEVANCE — flagging as off-topic")
        return [], [], True

    strong = []
    for m in matches:
        score = float(m.get("score", 0))
        md    = m.get("metadata", {})
        if float(md.get("authority_score", 0)) < MIN_AUTHORITY:
            continue
        specialty = str(md.get("specialty", "")).lower()
        if expected_fields and not any(f.lower() in specialty for f in expected_fields):
            continue
        text = _extract_text(md)
        strong.append({"title": md.get("title", ""), "text": text, "score": score})

    seen, unique = set(), []
    for s in strong:
        key = s["title"] or s["text"][:60]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    context_chunks = [s["text"] for s in unique if s["text"]][:TOP_K_FINAL]

    # [Fix 8] Only display high-confidence references to avoid mismatch
    display_titles = [
        s["title"] for s in unique
        if s["title"] and s["score"] >= MIN_REF_DISPLAY
    ][:TOP_K_FINAL]

    print(f"[ORA RAG] chunks={len(context_chunks)} refs_displayed={len(display_titles)}")
    return context_chunks, display_titles, False


# ========= GPT GENERATION =========
def gpt_style_answer(q: str, context_chunks=None, history=None) -> str:
    ar    = is_ar(q)
    qtype = detect_question_type(q)

    # --- RAG context block ---
    # [Fix 7] "supplement" is now strictly restricted to clinical curriculum facts.
    # The previous wording ("well-established general dental knowledge") allowed the
    # model to draw on its full training corpus, including folk remedies.
    # Now: only facts taught in accredited dental school programmes are permitted.
    if context_chunks:
        ctx = (
            "\n\nREFERENCE MATERIAL — this is your primary source:\n"
            + "\n---\n".join(context_chunks)
            + "\n\n"
            "You may supplement gaps using only clinical facts that are standard protocol "
            "in accredited dental school curricula — not home remedies, folk tips, or "
            "general internet advice. If unsure whether a fact meets this standard, omit it."
        )
    else:
        ctx = (
            "\n\nNo reference material retrieved. Answer from established clinical dental "
            "knowledge only — standard textbook protocols. "
            "Do NOT include home remedies (e.g. tea bags, clove oil as primary treatment, "
            "or other folk-sourced advice). Be specific and useful."
        )

    # --- Format rules ---
    # [Fix 4/5] Root cause of numbered lists, headings, and stars was the phrase
    # "use numbered steps or bullet points" — GPT treated all three as interchangeable.
    # Fix: specify the exact symbol to use, and explicitly forbid every other format element.
    _forbidden = (
        "STRICTLY FORBIDDEN in your response: "
        "numbered lists (1. 2. 3.), "
        "markdown headings (## or ### or ****Title****), "
        "bold text (**word**), "
        "asterisk bullets (* item), "
        "colons used as section dividers, "
        "and any summary or conclusion paragraph "
        "(e.g. 'In summary', 'To summarize', 'خلاصة', 'خلاصة القول', 'ختاماً').\n"
    )

    if qtype == "instruction":
        fmt = (
            "FORMAT: Use hyphen bullet points ( - ) ONLY. "
            "Each point is one clear, actionable sentence. " + _forbidden
        )
        length     = "LENGTH: 4 to 6 bullet points. No more.\n"
        disclaimer = ""
        # Instruction answers do not need a consultation reminder —
        # the user has already seen a dentist and needs post-care steps.

    elif qtype == "symptom":
        fmt        = "FORMAT: Plain prose only. " + _forbidden
        length     = "LENGTH: 3 to 5 sentences maximum.\n"
        disclaimer = (
            "After your explanation, end with exactly one sentence recommending "
            "a dental consultation for proper evaluation. This comes last.\n"
        )

    else:  # informational
        fmt        = "FORMAT: Plain prose only. " + _forbidden
        length     = "LENGTH: 2 to 4 sentences maximum.\n"
        disclaimer = (
            "End with one brief sentence recommending a dental consultation.\n"
        )

    # --- Alarmism rule ---
    # [Fix 6] Prevents the model from appending unsolicited escalation advice
    # (e.g. "if pain worsens see a dentist", "seek emergency care if...") to routine answers.
    alarmism = (
        "ESCALATION WARNINGS: Do NOT add advice such as 'if pain worsens see a dentist', "
        "'if symptoms persist consult a professional', or similar closing warnings. "
        "Add urgent advice ONLY if the user describes a genuinely urgent symptom: "
        "uncontrolled bleeding, fever alongside dental pain, or difficulty breathing "
        "or swallowing. For all other questions, omit escalation language entirely.\n"
    )

    # --- Language rules ---
    # [Fix 3] Arabic: previous version specified Gulf dialect vocabulary (يعني, الحين, عشان)
    # as explicit markers. This made every response sound overly casual regardless of topic.
    # Fix: specify register ("warm, professional"), provide the correct Arabic dental term
    # for each common concept, and restrict English to genuinely untranslatable cases only.
    if ar:
        lang = (
            "LANGUAGE: Write in warm, professional Arabic. "
            "Clear and natural — not formal bureaucratic Arabic, and not overly casual. "
            "The register should feel like a trusted, knowledgeable doctor "
            "speaking plainly to a patient.\n"
            "Use these standard Arabic dental terms:\n"
            "  حشوة (filling), علاج العصب (root canal), تاج (crown),\n"
            "  تنظيف الجير (scaling), تبييض الأسنان (whitening),\n"
            "  خيط الأسنان (floss), زرعة (implant), البلاك (plaque — "
            "accepted term in Arabic dental practice).\n"
            "Do NOT write English dental terms mid-sentence as a default style. "
            "If you include an English term, it must be in parentheses after the "
            "Arabic term, and only on the first use, and only when it genuinely helps "
            "a patient who may have heard the English term at their clinic.\n"
        )
    else:
        lang = (
            "LANGUAGE: Plain English for a non-dental adult. "
            "Briefly define any technical term the first time you use it. "
            "Warm, calm tone — like a knowledgeable friend explaining clearly.\n"
        )

    # --- Core system prompt ---
    # [Fix 10] Safety boundary is now explicit about what IS permitted,
    # not just what is forbidden. The previous version only said what not to do,
    # which caused the model to under-answer even legitimate informational questions.
    system = (
        "You are ORA, a dental health assistant for general patients — not professionals.\n"
        "Give accurate, specific, and genuinely useful dental information in a clear human tone.\n"
        "\nCORE RULES:\n"
        "- Be specific. Explain the likely cause and the physiological mechanism behind it.\n"
        "- You MAY explain: likely causes, what a condition means clinically, how it typically "
        "develops, what treatment approaches generally exist, and what a dentist visit for "
        "this condition usually involves.\n"
        "- You may NOT: prescribe specific medications, recommend dosages, or state a "
        "definitive diagnosis as certain.\n"
        "- Frame uncertain claims with 'typically', 'often', or 'in most cases'.\n"
        "- Do NOT claim insufficient information when general clinical dental knowledge exists.\n"
        "- Do NOT ask the user any follow-up questions.\n"
        "- If prior conversation is present in the input, use it to understand context "
        "before answering the current message.\n"
        "\n"
        + fmt
        + length
        + disclaimer
        + alarmism
        + lang
        + ctx
    )

    msgs = [{"role": "system", "content": system}]
    if history:
        for turn in history:
            role, content = turn.get("role", ""), turn.get("content", "")
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
        print(f"[ORA GPT ERROR] {e}")
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
        _, refs, _ = rag_retrieve(qv, match["field"])
    else:
        context_chunks, refs, is_off_topic = rag_retrieve(qv)
        # [Fix 9] Secondary scope gate: low RAG score + no dental vocabulary = off-topic
        if is_off_topic and not has_dental_signal(q):
            print(refusal_scope(q))
            exit()
        answer = gpt_style_answer(q, context_chunks)

    print(answer)
    if refs:
        print("\nReferences:")
        for r in refs:
            print(f"- {r}")
