import re
import math
from openai import OpenAI
from pinecone import Pinecone

# ========= CONFIG =========
MODEL = "gpt-5.2"
EMBED_MODEL = "text-embedding-3-large"

SIM_THRESHOLD = 0.70
PINECONE_INDEX = "oraapp111"

TOP_K_RAW = 8
TOP_K_FINAL = 3
MIN_AUTHORITY = 0.65

# Raised 600 → 800 to give Arabic follow-up responses more room.
# Reasoning tokens at "low" effort consume roughly 100-150 tokens,
# leaving ~650-700 for actual output — sufficient for structured answers.
MAX_GPT_TOKENS = 800

PINECONE_TEXT_FIELD = "chunk_text"

client = OpenAI()
pc = Pinecone()
index = pc.Index(PINECONE_INDEX)

# ========= DATASET =========
DATASET = [
    {
        "field": ["Periodontics"],
        "en_q": "My gums bleed when I brush, what should I do?",
        "en_a": "Bleeding gums are often a sign of gingival inflammation, which is when the gums become irritated due to plaque, a sticky layer of bacteria and food debris. In mild cases, you may notice slight bleeding and redness; in advanced cases, there can be swelling. Proper oral hygiene is essential to prevent this: brush at least twice daily (morning and bedtime) using correct technique and duration, and floss daily to clean between teeth where a toothbrush cannot reach. Plaque left on teeth can harden into calculus, which can only be removed professionally. Regular dental checkups and cleanings every six months help maintain gum health. Always consult a licensed dentist for personalized advice.",
        "ar_q": "نزيف اللثة عادة ما يكون مؤشرا على وجود التهاب",
        "ar_a": "يُعتبر نزيف اللثة في كثير من الأحيان علامة على التهاب اللثة، وهو ما يحدث عندما تتعرض اللثة للتهيج نتيجة تراكم البلاك، وهي طبقة لزجة من البكتيريا وفضلات الطعام. في الحالات الخفيفة، قد تلاحظ نزيفًا طفيفًا واحمرارًا؛ أما في الحالات المتقدمة، فقد يحدث تورم. إن الحفاظ على صحة الفم الجيدة أمر أساسي للوقاية من هذه المشكلة. يُنصح بتنظيف الأسنان بالفرشاة مرتين على الأقل يوميًا (في الصباح وقبل النوم) باستخدام الطريقة الصحيحة والتفريش لمدة كافية، بالإضافة إلى استخدام الخيط السني يوميًا لتنظيف ما بين الأسنان التي لا تستطيع فرشاة الأسنان الوصول إليها. إذا تُرك البلاك على الأسنان، فإنه يمكن أن يتصلب ليصبح جيرًا لا يمكن إزالته إلا عند طبيب الأسنان. تساعد الفحوصات الدورية كل ستة أشهر في الحفاظ على صحة اللثة. يجب دائمًا استشارة طبيب أسنان مرخص للحصول على نصائح تناسب حالتك."
    },
    {
        "field": ["Implant", "Periodontics"],
        "en_q": "I had a dental implant and notice bluish discoloration on my gum, is that normal?",
        "en_a": "Bluish discoloration near a dental implant often occurs in patients with a thin gingival biotype, where the gum tissue is naturally thin and slightly transparent. This is a normal anatomical variation and usually an aesthetic concern rather than a medical problem. Evaluation by a periodontist or implant specialist is recommended to assess whether any aesthetic corrections might be desired. Always consult a licensed dentist or specialist for proper assessment.",
        "ar_q": "بعد ما سويت زرعة صار عندي لون أزرق في اللثة",
        "ar_a": "اللون الأزرق اللي تشوفه بالقرب من الزرعة (implant) غالبًا يحصل عند الناس اللي عندهم لثة رقيعة بطبيعتها — يعني اللثة رفيعة لدرجة إنها تخلي لون الزرعة يظهر من تحتها. هذا الشيء طبيعي تشريحيًا وما يعني إن في مشكلة طبية، بس يمكن يكون مزعج من الناحية الجمالية. إذا كان هذا اللون يضايقك، يُنصح بزيارة طبيب متخصص في أمراض اللثة أو زراعة الأسنان عشان يقيّم إذا في حل جمالي مناسب."
    },
    {
        "field": ["Restorative Dentistry"],
        "en_q": "I had a filling and now I feel pain only when biting, what does it mean?",
        "en_a": "Pain that occurs only when biting after a filling typically indicates that the restoration is slightly high, a condition called high occlusion. This happens when the filling contacts the opposing tooth before the rest of the teeth, creating pressure during chewing. Unlike generalized sensitivity, this pain is limited to biting. Adjustment by a dentist resolves the issue. Always consult a licensed dentist for proper evaluation.",
        "ar_q": "بعد الحشوة صار عندي ألم عند العض فقط",
        "ar_a": "الألم اللي يحصل فقط عند العض بعد الحشوة غالبًا معناه إن الحشوة طالعة شوي عالي — يعني تلامس السن المقابل قبل باقي الأسنان، وهذا يسبب ضغط وقت المضغ. هذا النوع من الألم يختلف عن الحساسية العادية لأنه محصور في العض بس. الحل بسيط: طبيب الأسنان يعدّل ارتفاع الحشوة وينتهي الموضوع."
    },
    {
        "field": ["Endodontics"],
        "en_q": "I experienced severe tooth pain that disappeared without treatment. What does it mean?",
        "en_a": "The disappearance of severe tooth pain may indicate that the pulp inside the tooth has lost vitality. Because the pulp contains nerves, pain can subside even while bacterial infection continues. This does not mean the tooth is healthy and can lead to further complications if untreated. Evaluation by a licensed dentist is necessary.",
        "ar_q": "اختفى الألم الشديد في السن",
        "ar_a": "اختفاء الألم الشديد فجأة ما يعني إن السن تعافى — بالعكس، هذا أحيانًا يكون علامة إن العصب (اللب) داخل السن فقد حيويته. لما يموت العصب، الألم يروح لأنه ما عاد في أعصاب تحس، لكن المشكلة ما زالت موجودة وتحتاج تقييم. إذا تركت بدون علاج قد تصير مضاعفات. يُنصح بزيارة طبيب الأسنان حتى لو ما تحس بألم الحين."
    },
    {
        "field": ["Endodontics"],
        "en_q": "What does it mean that a tooth 'rots'?",
        "en_a": "The term 'tooth rotting' is informal and often misleading. It usually refers to a tooth affected by bacterial infection of the dentin or pulp, commonly following decay or trauma. The tooth is not literally rotting, but ongoing infection can cause structural damage over time. Evaluation by a licensed dentist is important to determine the appropriate management.",
        "ar_q": "هل تعني كلمة تعفن السن أن السن ميت؟",
        "ar_a": "كلمة 'تعفن' مصطلح شعبي وليس طبيًا، وأحيانًا يكون مضلل. اللي يصير فعلًا هو إن تكون فيه بكتيريا داخل السن، سواء في الطبقة الداخلية (الدنتين) أو في اللب (العصب)، وهذا يحصل عادةً بسبب تسوس أو ضربة. السن ما يتعفن حرفيًا، لكن إذا تركت البكتيريا تشتغل بدون علاج، ممكن تضر ببنية السن مع الوقت. الأفضل تراجع طبيب الأسنان عشان يشوف الوضع."
    }
]

# ========= SCOPE + SAFETY =========
def is_ar(text):
    return bool(re.search(r"[\u0600-\u06FF]", text))

def is_treatment_request(q):
    ql = q.lower()
    blocked = [
        "prescribe", "write prescription", "which medication",
        "which antibiotic", "give me medication",
        "وصفة", "اعطني دواء", "اي مضاد حيوي", "شخص حالتي"
    ]
    return any(x in ql for x in blocked)

def is_out_of_scope(q):
    ql = q.lower()
    banned = [
        "capital of", "weather", "stock", "bitcoin", "movie", "music",
        "recipe", "football", "basketball", "politics", "president",
        "travel", "hotel", "flight", "restaurant",
        "عاصمة", "طقس", "مباراة", "سياسة", "فيلم", "موسيقى", "سفر", "مطعم"
    ]
    return any(b in ql for b in banned)

def refusal_treatment(q):
    return "عذرًا، لا يمكنني تقديم تشخيص أو خطة علاجية. يُنصح بمراجعة طبيب أسنان مرخص." if is_ar(q) \
           else "Sorry, I cannot provide diagnosis or treatment plans. Please consult a licensed dentist."

def refusal_scope(q):
    return "عذرًا، هذا السؤال خارج نطاق تطبيق صحة الفم والأسنان." if is_ar(q) \
           else "Sorry, this question is outside the scope of this oral health application."

# ========= QUESTION TYPE DETECTION =========
# [New — Fix 2/9] Drives conditional formatting in the system prompt.
# instruction → numbered steps are required and appropriate
# symptom     → prose explaining cause, mechanism, and meaning
# informational → prose explanation
def detect_question_type(q):
    ql = q.lower()
    instruction_signals = [
        "how to", "how do i", "what should i do", "aftercare",
        "after surgery", "after extraction", "after root canal",
        "after implant", "after filling", "steps", "instructions",
        "كيف", "ماذا أفعل", "ماذا يجب", "تعليمات", "خطوات",
        "بعد العملية", "بعد الخلع", "بعد التركيب", "بعد الزرعة",
        "بعد عملية", "بعد الحشوة", "ايش أسوي"
    ]
    symptom_signals = [
        "pain", "hurt", "hurts", "bleed", "bleeding", "swell", "swelling",
        "ache", "sensitivity", "sensitive", "feel", "notice", "discomfort",
        "ألم", "يؤلم", "يؤلمني", "نزيف", "تورم", "حساسية",
        "أشعر", "لاحظت", "يحس", "وجع", "موجوع"
    ]
    for sig in instruction_signals:
        if sig in ql:
            return "instruction"
    for sig in symptom_signals:
        if sig in ql:
            return "symptom"
    return "informational"

# ========= INTENT PHRASES =========
INTENT_PHRASES = {
    0: ["gum bleed", "bleeding gums", "نزيف اللثة", "اللثة تنزف", "لثتي تنزف"],
    1: ["blue gum", "implant", "زرعة", "زرقة اللثة"],
    2: ["pain when biting", "after filling", "ألم عند العض", "بعد الحشوة"],
    3: ["pain disappeared", "اختفى الألم", "راح الألم"],
    4: ["tooth rotting", "rotting tooth", "تعفن السن", "السن ميت"]
}

# ========= EMBEDDING =========
def embed(text):
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

print("[ORA] Pre-computing dataset embeddings...")
_DATASET_EN_VECS = [embed(d["en_q"]) for d in DATASET]
_DATASET_AR_VECS = [embed(d["ar_q"]) for d in DATASET]
print(f"[ORA] Done — {len(DATASET)} entries cached.")

# ========= DATASET MATCH =========
def dataset_match(q, qv=None):
    q_norm = q.lower().strip()
    ar = is_ar(q)

    for idx, phrases in INTENT_PHRASES.items():
        for p in phrases:
            if p in q_norm:
                return DATASET[idx], 1.0, ar, idx

    if qv is None:
        qv = embed(q)

    vecs = _DATASET_AR_VECS if ar else _DATASET_EN_VECS
    best, score, best_idx = None, -1, -1
    for i, dv in enumerate(vecs):
        s = cosine(qv, dv)
        if s > score:
            score, best, best_idx = s, DATASET[i], i

    return best, score, ar, best_idx

# ========= RAG RETRIEVAL =========
def _extract_text(md):
    """
    [Fix 10] Added diagnostic logging so you can see exactly which metadata
    field is being matched — or whether none matched at all.
    If you see "No text found" warnings, check your actual Pinecone field names
    by inspecting the logged available fields and updating PINECONE_TEXT_FIELD.
    """
    for field in [PINECONE_TEXT_FIELD, "chunk_text", "title", "source_type", "authority_score", "source_path"]:
        val = md.get(field, "")
        if val:
            print(f"[ORA RAG] Text matched on field='{field}' len={len(str(val))}")
            return str(val).strip()
    print(f"[ORA RAG] WARNING — no text found. Fields present: {list(md.keys())}")
    return ""

def rag_retrieve(qv, expected_fields=None):
    res = index.query(vector=qv, top_k=TOP_K_RAW, include_metadata=True)

    strong = []
    for m in res.get("matches", []):
        md = m.get("metadata", {})
        if float(md.get("authority_score", 0)) < MIN_AUTHORITY:
            continue
        specialty = str(md.get("specialty", "")).lower()
        if expected_fields and not any(f.lower() in specialty for f in expected_fields):
            continue
        title = md.get("title", "")
        text = _extract_text(md)
        strong.append({"title": title, "text": text})

    seen = set()
    unique = []
    for s in strong:
        key = s["title"] if s["title"] else s["text"][:60]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    context_chunks = [s["text"] for s in unique if s["text"]][:TOP_K_FINAL]
    display_titles = [s["title"] for s in unique if s["title"]][:TOP_K_FINAL]

    print(f"[ORA RAG] {len(context_chunks)} usable chunks / {len(display_titles)} titles retrieved")
    return context_chunks, display_titles

# ========= GPT =========
def gpt_style_answer(q, context_chunks=None, history=None):
    """
    Changes from previous version:
    - [Fix 1/3]  "No bullet points" rule removed — replaced with conditional formatting
    - [Fix 2]    "One explanation only" removed — replaced with depth requirement
    - [Fix 4]    RAG grounding relaxed from "only" to "primary source + supplement"
    - [Fix 6]    Symptom questions now explicitly required to explain cause + mechanism
    - [Fix 7]    history parameter added — passed as prior turns into the input list
    - [Fix 8]    "Consult a dentist" is now conditional and must follow a complete answer
    - [Fix 9]    All rules that conflict with useful output removed or inverted
    - [Fix 5/ar] Arabic rules tightened: specific terms listed, MSA explicitly forbidden
    """
    ar = is_ar(q)
    qtype = detect_question_type(q)

    # --- RAG context block ---
    # [Fix 4] Changed from "only reference material" to "primary + supplement".
    # The old wording caused GPT to hedge when RAG content was sparse or partially
    # relevant, because it had no fallback. Now it can fill gaps with established
    # dental knowledge, which prevents artificial under-answering.
    if context_chunks:
        context_section = (
            "\n\nReference material — treat as your primary source. "
            "You may supplement gaps using well-established general dental knowledge. "
            "Do not fabricate specific clinical claims unsupported by either:\n\n"
            + "\n---\n".join(context_chunks)
        )
    else:
        context_section = (
            "\n\nNo reference material retrieved. "
            "Answer entirely from well-established general dental knowledge. "
            "Be specific and genuinely useful — do not hedge or claim insufficient information."
        )

    # --- Format rules (question-type-aware) ---
    # [Fix 2/9] Previously: "No bullet points" was unconditional and broke instruction answers.
    # Now: format is chosen based on what the question actually needs.
    if qtype == "instruction":
        format_rules = (
            "FORMAT: This is an aftercare or how-to question. "
            "Use numbered steps or bullet points — this is required here. "
            "Steps should be brief, specific, and actionable.\n"
        )
    elif qtype == "symptom":
        format_rules = (
            "FORMAT: This is a symptom question. Write in natural flowing prose. "
            "You must cover: (1) the most likely cause, (2) the mechanism — why it happens "
            "physiologically, (3) what it typically means for the patient. "
            "Be specific. A vague or overly cautious answer does not help the user.\n"
        )
    else:
        format_rules = (
            "FORMAT: This is an informational question. "
            "Write in clear, natural flowing prose. Be complete and specific.\n"
        )

    # --- Disclaimer rule ---
    # [Fix 8] Previously unconditional and often replaced or shortened the answer.
    # Now explicit: the disclaimer is one sentence, comes last, never truncates content.
    disclaimer_rule = (
        "DISCLAIMER: After giving a complete and informative answer, add exactly one brief "
        "sentence advising consultation with a licensed dentist for personal evaluation. "
        "This sentence comes last. It must never replace, shorten, or appear before the answer.\n"
    )

    # --- Language rules ---
    # [Fix 5] Arabic rules now explicitly name which terms stay in English.
    # Previous version was vague ("keep common dental terms") which caused
    # inconsistent handling — e.g. plaque being translated to اللويحة الجرثومية.
    if ar:
        lang_rules = (
            "\nLANGUAGE — Arabic:\n"
            "- Write in natural, conversational Arabic — Gulf/Modern spoken style.\n"
            "- NOT formal MSA. NOT translated English sentences.\n"
            "- Sound like a knowledgeable friend, not a medical text.\n"
            "- The following dental terms must appear in English exactly as written, "
            "used naturally inside Arabic sentences: "
            "plaque, scaling, crown, implant, filling, root canal, veneer, "
            "whitening, floss, block, calculus, gingivitis, periodontitis.\n"
            "- Do NOT translate these terms into Arabic. Examples of correct usage:\n"
            "  'البلاك (plaque) يتراكم على الأسنان'\n"
            "  'عملية root canal ما تكون مؤلمة في الغالب'\n"
            "  'الـ scaling هو تنظيف الجير عند طبيب الأسنان'\n"
            "- A short Arabic clarification in parentheses is allowed the FIRST time a term "
            "appears, only if it genuinely helps a layperson understand it.\n"
            "- Forbidden: 'اللويحة الجرثومية', 'قلح الأسنان' (for calculus), "
            "or any formal MSA rendering of the listed terms.\n"
            "- Use direct, everyday phrasing: 'يعني', 'يصير', 'لما', 'عشان', 'الحين'.\n"
            "- Avoid formal openers like 'يُعتبر', 'يتضمن', 'وبما أن', 'تجدر الإشارة'.\n"
        )
    else:
        lang_rules = (
            "\nLANGUAGE — English:\n"
            "- Clear, plain English for a non-dental adult audience.\n"
            "- Briefly explain any dental term the first time you use it.\n"
            "- Warm, calm, conversational tone — like a knowledgeable friend.\n"
        )

    system = (
        "You are ORA, a dental health assistant for everyday people — not professionals.\n"
        "Your role is to give genuinely useful, accurate, specific dental information "
        "in a calm, human tone. Think of yourself as a knowledgeable friend who happens "
        "to know dentistry well — not a liability-conscious chatbot.\n"
        "\nCORE RULES:\n"
        "- Be specific and informative. Vague or overly cautious answers fail the user.\n"
        "- Explain the most likely cause with enough clinical detail that the user "
        "actually understands what is happening in their mouth.\n"
        "- Do NOT claim insufficient information when dental knowledge is available to answer.\n"
        "- Do NOT hedge unnecessarily or add unprompted apologies or limitations.\n"
        "- You may describe treatments as general possibilities — never as instructions.\n"
        "- Frame clinical claims as 'likely', 'often', or 'typically' — not as a diagnosis.\n"
        "- Do NOT ask the user follow-up questions.\n"
        "- If prior conversation messages are present, use them to understand context "
        "and answer the current question accordingly — do not treat the conversation as isolated.\n"
        "\n"
        + format_rules
        + disclaimer_rule
        + lang_rules
        + context_section
    )

    # [Fix 7] Build input with conversation history.
    # Prior turns give GPT the context it needs to correctly interpret follow-up questions.
    messages = [{"role": "system", "content": system}]
    if history:
        for turn in history:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": q})

    try:
        r = client.responses.create(
            model=MODEL,
            reasoning={"effort": "low"},
            input=messages,
            max_output_tokens=MAX_GPT_TOKENS
        )
        return r.output_text.strip()

    except Exception as e:
        print(f"[ORA GPT ERROR] {e}")
        if ar:
            return "عذرًا، حدث خطأ أثناء معالجة سؤالك. يُنصح بمراجعة طبيب أسنان مرخص."
        return "Sorry, an error occurred while processing your question. Please consult a licensed dentist."

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
        fields = match["field"]
        _, refs = rag_retrieve(qv, fields)
    else:
        context_chunks, refs = rag_retrieve(qv, expected_fields=None)
        answer = gpt_style_answer(q, context_chunks)

    print(answer)
    if refs:
        print("\nReferences:")
        for ref in refs:
            print(f"- {ref}")
