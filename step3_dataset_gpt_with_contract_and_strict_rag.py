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

# [Fix 1] Raised from 220 → 600.
# At 220, reasoning tokens consumed most of the budget, leaving ~100-150 tokens
# for actual output. Arabic text is especially token-heavy — this was the
# primary cause of cut-off responses.
MAX_GPT_TOKENS = 600

# [Fix 2] The metadata field name in Pinecone that holds the document text.
# ⚠️ Check your actual Pinecone metadata — common names: "chunk_text", "title", "source_type", "authority_score", "source_path"
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

# [Fix 3] Pre-compute all dataset embeddings once when the module loads.
# Previously: up to 5 embed() calls per request inside dataset_match().
# Now: 10 embed() calls total at startup, 0 during query time for dataset comparison.
# Server startup will take a few extra seconds — this is expected and normal.
print("[ORA] Pre-computing dataset embeddings...")
_DATASET_EN_VECS = [embed(d["en_q"]) for d in DATASET]
_DATASET_AR_VECS = [embed(d["ar_q"]) for d in DATASET]
print(f"[ORA] Done — {len(DATASET)} entries cached.")

# ========= DATASET MATCH =========
# [Fix 3] Now accepts qv (pre-computed query vector) to avoid redundant embed() calls.
# qv is computed once in the caller and passed here — no re-embedding.
def dataset_match(q, qv=None):
    q_norm = q.lower().strip()
    ar = is_ar(q)

    # Intent phrase matching first — zero embedding cost
    for idx, phrases in INTENT_PHRASES.items():
        for p in phrases:
            if p in q_norm:
                return DATASET[idx], 1.0, ar, idx

    # Use pre-computed dataset vectors; compute qv only if not passed in
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
    Try common Pinecone metadata field names for document text.
    Falls back through alternatives if PINECONE_TEXT_FIELD is not found.
    ⚠️ If retrieval returns empty chunks, check your actual Pinecone field names.
    """
    for field in [PINECONE_TEXT_FIELD, "chunk_text", "title", "source_type", "authority_score", "source_path"]:
        val = md.get(field, "")
        if val:
            return str(val).strip()
    return ""

# [Fix 2] Replaces rag_refs().
# Previously: returned only titles, document content never reached GPT.
# Now: returns both context_chunks (injected into GPT prompt) and display_titles (shown to user).
# Also accepts pre-computed qv to avoid a second embed() call.
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

    # Deduplicate by title (or by first 60 chars of text if no title)
    seen = set()
    unique = []
    for s in strong:
        key = s["title"] if s["title"] else s["text"][:60]
        if key not in seen:
            seen.add(key)
            unique.append(s)

    context_chunks = [s["text"] for s in unique if s["text"]][:TOP_K_FINAL]
    display_titles = [s["title"] for s in unique if s["title"]][:TOP_K_FINAL]

    return context_chunks, display_titles

# ========= GPT =========
# [Fix 2] Now accepts context_chunks from rag_retrieve() — GPT answer is grounded
#         in actual retrieved documents instead of relying on model memory.
# [Fix 1] Uses the corrected MAX_GPT_TOKENS = 600.
# Arabic prompt now enforces natural conversational tone, not formal MSA.
def gpt_style_answer(q, context_chunks=None):
    ar = is_ar(q)

    # [Fix 2] Build context block from retrieved Pinecone documents
    if context_chunks:
        context_section = (
            "\n\nBase your answer on the following reference material only. "
            "Do not add facts that are not supported here:\n\n"
            + "\n---\n".join(context_chunks)
        )
    else:
        context_section = (
            "\n\nNo reference material was retrieved. "
            "Answer using only well-established general dental knowledge. "
            "Do not invent specific clinical details."
        )

    # Language-specific prompt rules
    if ar:
        lang_rules = (
            "\n\nLANGUAGE — Arabic response required:\n"
            "- Write in natural, conversational Arabic. NOT formal MSA. NOT translated sentences.\n"
            "- Sound like a knowledgeable friend explaining something clearly and simply.\n"
            "- Keep common English dental terms as-is: plaque, scaling, block, crown, implant, "
            "filling, whitening, root canal, veneer, etc.\n"
            "- After each English term, add a short Arabic explanation in parentheses if helpful.\n"
            "  Correct: 'البلاك (plaque) هو طبقة بكتيريا تتكون على الأسنان'\n"
            "  Wrong: 'اللويحة السنية البكتيرية هي طبقة لزجة'\n"
            "- Avoid stiff formal phrases like 'يُعتبر', 'يتضمن', 'يُلاحظ', 'وبما أن'.\n"
            "- Use direct, everyday phrasing instead.\n"
        )
    else:
        lang_rules = (
            "\n\nLANGUAGE — English response required:\n"
            "- Write in clear, plain English for a general (non-dental) audience.\n"
            "- Explain any dental term immediately after using it.\n"
        )

    system = (
        "You are ORA, a dental health assistant for general users — not dental students or professionals.\n"
        "Your tone is calm, clear, and reassuring — like a knowledgeable friend, not a textbook.\n"
        "\nSTRICT RULES:\n"
        "- Give one most likely explanation only.\n"
        "- No bullet points.\n"
        "- No alarmist or exaggerated language.\n"
        "- No follow-up questions.\n"
        "- You may mention treatments only as general possibilities — never as a decision or instruction.\n"
        "- Do not diagnose the patient.\n"
        "- Use plain, practical language that any adult understands.\n"
        "- If reference material is provided below, base your answer on it. Do not invent facts.\n"
        "- End with a brief note advising the user to consult a licensed dentist.\n"
        + lang_rules
        + context_section
    )

    try:
        r = client.responses.create(
            model=MODEL,
            reasoning={"effort": "low"},
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": q}
            ],
            max_output_tokens=MAX_GPT_TOKENS
        )
        return r.output_text.strip()

    except Exception as e:
        # [Fix 1] Graceful fallback — prevents a GPT API error from crashing the endpoint
        print(f"[ORA GPT ERROR] {e}")
        if ar:
            return "عذرًا، حدث خطأ أثناء معالجة سؤالك. يُنصح بمراجعة طبيب أسنان مرخص."
        return "Sorry, an error occurred while processing your question. Please consult a licensed dentist."

# ========= MAIN =========
if __name__ == "__main__":
    q = input("> ").strip()

    if is_treatment_request(q):
        print(refusal_treatment(q))
        exit()

    if is_out_of_scope(q):
        print(refusal_scope(q))
        exit()

    # [Fix 3] Single embed() call — reused by both dataset_match and rag_retrieve
    qv = embed(q)
    match, score, ar, idx = dataset_match(q, qv)

    if match and score >= SIM_THRESHOLD:
        answer = match["ar_a"] if ar else match["en_a"]
        fields = match["field"]
        # Answer is pre-written; retrieve refs for display only
        _, refs = rag_retrieve(qv, fields)
    else:
        # [Fix 2] Retrieve context first, then ground GPT with it
        context_chunks, refs = rag_retrieve(qv, expected_fields=None)
        answer = gpt_style_answer(q, context_chunks)

    print(answer)
    if refs:
        print("\nReferences:")
        for ref in refs:
            print(f"- {ref}")
