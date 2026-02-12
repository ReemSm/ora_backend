import re
import math
from openai import OpenAI
from pinecone import Pinecone

# ========= CONFIG =========
MODEL = "gpt-5.2"
EMBED_MODEL = "text-embedding-3-large"

SIM_THRESHOLD = 0.70
PINECONE_INDEX = "oraapp111"

TOP_K_RAW = 8            # strict gate: retrieve 8
TOP_K_FINAL = 3          # max refs shown
MIN_AUTHORITY = 0.55     # authority gate

MAX_GPT_TOKENS = 220     # hard verbosity cap

# ---- STRICT RAG TIGHTENERS (added) ----
PINECONE_SCORE_MIN = 0.82     # require strong vector match (reduces random refs)
LOCAL_RERANK_MIN = 0.80       # require strong local semantic match
REQUIRE_MIN_REFS = 2          # keep your ≥2 refs gate, explicit

client = OpenAI()
pc = Pinecone()
index = pc.Index(PINECONE_INDEX)

# ========= DATASET =========
DATASET = [
    {
        "field": ["Periodontics"],
        "en_q": "My gums bleed when I brush, what should I do?",
        "en_a": "Bleeding gums are often a sign of gingival inflammation, which is when the gums become irritated due to plaque, a sticky layer of bacteria and food debris. In mild cases, you may notice slight bleeding and redness; in advanced cases, there can be swelling. Proper oral hygiene is essential to prevent this: brush at least twice daily (morning and bedtime) using correct technique and duration, and floss daily to clean between teeth where a toothbrush cannot reach. Plaque left on teeth can harden into calculus, which can only be removed professionally. Regular dental checkups and cleanings every six months help maintain gum health. Educating patients on the cause and prevention empowers them to take control of their oral health. Always consult a licensed dentist for personalized advice.",
        "ar_q": "نزيف اللثة عادة ما يكون مؤشرا على وجود التهاب",
        "ar_a": "يُعتبر نزيف اللثة في كثير من الأحيان علامة على التهاب اللثة، وهو ما يحدث عندما تتعرض اللثة للتهيج نتيجة تراكم البلاك، وهي طبقة لزجة من البكتيريا وفضلات الطعام. في الحالات الخفيفة، قد تلاحظ نزيفًا طفيفًا واحمرارًا؛ أما في الحالات المتقدمة، فقد يحدث تورم. إن الحفاظ على صحة الفم الجيدة أمر أساسي للوقاية من هذه المشكلة. يُنصح بتنظيف الأسنان بالفرشاة مرتين على الأقل يوميًا (في الصباح وقبل النوم) باستخدام الطريقة الصحيحة والتفريش لمدة كافية، بالإضافة إلى استخدام الخيط السني يوميًا لتنظيف ما بين الأسنان التي لا تستطيع فرشاة الأسنان الوصول إليها. إذا تُرك البلاك على الأسنان، فإنه يمكن أن يتصلب ليصبح جيرًا، الذي لا يمكن إزالته إلا بواسطة أدوات خاصة عند طبيب الأسنان. تساعد الفحوصات الدورية وتنظيف الأسنان كل ستة أشهر في الحفاظ على اللثة. كما أن تثقيف المرضى حول الأسباب وطرق الوقاية يُمكنهم من السيطرة على صحتهم الفموية. يجب دائمًا استشارة طبيب أسنان مرخص للحصول على نصائح شخصية تناسب احتياجاتك الفردية."
    },
    {
        "field": ["Implant", "Periodontics"],
        "en_q": "I had a dental implant and notice bluish discoloration on my gum, is that normal?",
        "en_a": "Bluish discoloration near a dental implant often occurs in patients with a thin gingival biotype, where the gum tissue is naturally thin and slightly transparent. This is a normal anatomical variation and usually an aesthetic concern rather than a medical problem. Understanding gingival biotypes can help patients appreciate natural differences in gum appearance. Evaluation by a periodontist or implant specialist is recommended to assess whether any aesthetic corrections might be desired. Always consult a licensed dentist or specialist for proper assessment.",
        "ar_q": "بعد ما سويت زرعة صار عندي لون أزرق في اللثة",
        "ar_a": "التغير في لون اللثة للون الأزرق بالقرب من المنطقة التي تمت زراعة الأسنان فيها، يحدث غالبًا لدى المرضى الذين يمتلكون نوعًا رقيقًا من اللثة، حيث تكون أنسجة اللثة رقيقة بطبيعتها وأقرب إلى أن تكون شفافة إلى حد ما. يُعتبر هذا التغيير تشريحيًا طبيعيًا وعادةً ما يمثل مسألة جمالية أكثر من كونه مشكلة طبية. يساعد فهم أنواع اللثة المرضى على تقدير الاختلافات الطبيعية في مظهر اللثة. يُوصى بإجراء تقييم من قبل طبيب متخصص في أمراض اللثة أو زراعة الأسنان لتحديد ما إذا كانت هناك حاجة لأي تصحيحات جمالية. من المهم دائمًا استشارة طبيب أسنان مرخص أو متخصص للحصول على تقييم دقيق."
    },
    {
        "field": ["Restorative Dentistry"],
        "en_q": "I had a filling and now I feel pain only when biting, what does it mean?",
        "en_a": "Pain that occurs only when biting after a filling typically indicates that the restoration is slightly high, a condition called high occlusion. This happens when the filling contacts the opposing tooth before the rest of the teeth, creating pressure during chewing. Unlike generalized sensitivity, this pain is limited to biting. Adjustment by a dentist resolves the issue. Always consult a licensed dentist for proper evaluation.",
        "ar_q": "بعد الحشوة صار عندي ألم عند العض فقط",
        "ar_a": "الألم الذي يحدث فقط عند العض بعد وضع حشوة على السن، عادةً ما يدل على أن الحشوة مرتفعة قليلاً، وهي حالة تُعرف بارتفاع الإطباق. يحدث ذلك عندما تلامس الحشوة السن المقابل قبل باقي الأسنان، مما يسبب ضغطًا أثناء المضغ. على عكس الحساسية العامة، يقتصر هذا الألم على العض فقط. من الضروري استشارة طبيب أسنان مرخص لتقييم الحالة وإجراء التعديل اللازم."
    },
    {
        "field": ["Endodontics"],
        "en_q": "I experienced severe tooth pain that disappeared without treatment. What does it mean?",
        "en_a": "The disappearance of severe tooth pain may indicate that the pulp inside the tooth has lost vitality. Because the pulp contains nerves, pain can subside even while bacterial infection continues. This does not mean the tooth is healthy and can lead to further complications if untreated. Evaluation by a licensed dentist is necessary.",
        "ar_q": "اختفى الألم الشديد في السن",
        "ar_a": "اختفاء ألم الأسنان الشديد قد يشير إلى أن اللب داخل السن فقد حيويته. وبما أن اللب يحتوي على الأعصاب، فقد يختفي الألم رغم استمرار المشكلة. لا يعني غياب الألم أن السن سليم، وقد يؤدي ذلك إلى مضاعفات إذا لم يتم علاجه. يُنصح باستشارة طبيب أسنان مرخص للتقييم."
    },
    {
        "field": ["Endodontics"],
        "en_q": "What does it mean that a tooth ‘rots’?",
        "en_a": "The term ‘tooth rotting’ is informal and often misleading. It usually refers to a tooth affected by bacterial infection of the dentin or pulp, commonly following decay or trauma. The tooth is not literally rotting, but ongoing infection can cause structural damage over time. Evaluation by a licensed dentist is important to determine the appropriate management.",
        "ar_q": "هل تعني كلمة تعفن السن أن السن ميت؟",
        "ar_a": "مصطلح تعفن الأسنان هو مصطلح غير طبي وقد يكون مضللاً. غالبًا ما يُقصد به وجود عدوى بكتيرية في السن، تحدث عادةً بعد تسوس أو إصابة. لا يعني ذلك أن السن يتعفن حرفيًا، لكن العدوى المستمرة قد تؤدي إلى تلف بنية السن مع الوقت. يُنصح باستشارة طبيب أسنان مرخص للتقييم المناسب."
    }
]

# ========= INTENT PHRASES =========
INTENT_PHRASES = {
    0: ["gum bleed", "bleeding gums", "نزيف اللثة", "اللثة تنزف", "لثتي تنزف"],
    1: ["blue gum", "implant", "زرعة", "زرقة اللثة"],
    2: ["pain when biting", "after filling", "ألم عند العض", "بعد الحشوة"],
    3: ["pain disappeared", "اختفى الألم", "راح الألم"],
    4: ["tooth rotting", "rotting tooth", "تعفن السن", "السن ميت"]
}

AR_RE = re.compile(r"[\u0600-\u06FF]")

def is_ar(text):
    return bool(AR_RE.search(text))

def embed(text):
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def dataset_match(q):
    q_norm = q.lower().strip()
    ar = is_ar(q)

    for idx, phrases in INTENT_PHRASES.items():
        for p in phrases:
            if p in q_norm:
                return DATASET[idx], 1.0, ar, idx

    qv = embed(q)
    best, score, best_idx = None, -1, -1
    for i, d in enumerate(DATASET):
        dv = embed(d["ar_q"] if ar else d["en_q"])
        s = cosine(qv, dv)
        if s > score:
            score, best, best_idx = s, d, i

    return best, score, ar, best_idx

# ========= SCOPE / POLICY GATES (added) =========
# One-line only. English + Arabic. No extra justification.
SCOPE_APOLOGY = "Sorry, this is outside the scope of this dental app. / عذرًا، هذا خارج نطاق تطبيق الأسنان."
CLINICAL_APOLOGY = "Sorry, I can’t help with diagnosis, treatment plans, or prescriptions. / عذرًا، لا يمكنني المساعدة في التشخيص أو خطة العلاج أو الوصفات."

_DENTAL_HINTS = [
    "tooth", "teeth", "gum", "gums", "gingiva", "oral", "mouth", "tongue", "jaw",
    "braces", "orthodont", "implant", "filling", "crown", "bridge", "root canal",
    "caries", "decay", "enamel", "dentin", "pulp", "periodont", "endodont",
    "تسوس", "سن", "أسنان", "ضرس", "لثة", "فم", "لسان", "فك", "تقويم", "حشوة", "تلبيسة", "تاج", "عصب", "زرعة"
]

_DISALLOWED_CLINICAL = [
    "diagnos", "diagnosis", "what do i have", "what is my condition",
    "treatment plan", "plan", "prescribe", "prescription", "antibiotic",
    "dose", "dosage", "mg", "medication", "drug", "painkiller",
    "وصفة", "وصفه", "روشتة", "روشته", "جرعة", "جرعه", "تشخيص", "شخّص", "خطة علاج", "علاج", "مضاد", "مضاد حيوي"
]

def is_dental_scope(q: str) -> bool:
    qn = q.lower()
    return any(h in qn for h in _DENTAL_HINTS) or is_ar(q) and any(h in q for h in _DENTAL_HINTS)

def is_disallowed_clinical_request(q: str) -> bool:
    qn = q.lower()
    return any(k in qn for k in _DISALLOWED_CLINICAL) or (is_ar(q) and any(k in q for k in _DISALLOWED_CLINICAL))

# ========= STRICT RAG GATE =========
def rag_refs(query_text, expected_fields):
    qv = embed(query_text)
    res = index.query(vector=qv, top_k=TOP_K_RAW, include_metadata=True)

    # Phase 1: hard filters (authority + specialty + pinecone similarity)
    filtered = []
    for m in res.get("matches", []):
        md = m.get("metadata", {})
        if float(md.get("authority_score", 0)) < MIN_AUTHORITY:
            continue

        # require strong pinecone similarity to avoid "random" refs
        try:
            if float(m.get("score", 0)) < PINECONE_SCORE_MIN:
                continue
        except Exception:
            continue

        specialty = str(md.get("specialty", "")).lower()
        if expected_fields and not any(f.lower() in specialty for f in expected_fields):
            continue

        title = md.get("title")
        if not title or not isinstance(title, str) or len(title.strip()) < 6:
            continue

        # keep some text to rerank locally (no backend changes)
        hint_text = f"{title} {md.get('specialty','')}".strip()
        filtered.append((title.strip(), hint_text))

    # Phase 2: local semantic rerank to enforce relevance deterministically
    ranked = []
    if filtered:
        for title, hint_text in filtered:
            hv = embed(hint_text)
            s = cosine(qv, hv)
            if s >= LOCAL_RERANK_MIN:
                ranked.append((s, title))

    ranked.sort(key=lambda x: x[0], reverse=True)

    # strict gate: require ≥2 strong refs
    strong_titles = []
    for _, t in ranked:
        if t not in strong_titles:
            strong_titles.append(t)

    if len(strong_titles) < REQUIRE_MIN_REFS:
        return []

    return strong_titles[:TOP_K_FINAL]

# ========= GPT FALLBACK (CONTRACT-LOCKED) =========
def gpt_style_answer(q):
    system = (
        "Write in the exact style, tone, and length of the provided dental Q&A dataset.\n"
        "STRICT:\n"
        "- One most likely explanation only.\n"
        "- No bullet points.\n"
        "- No warnings or alarmist language.\n"
        "- No follow-up questions.\n"
        "- No treatment plans or prescriptions.\n"
        "- Plain language biological explanation.\n"
        "- Do not mention missing data, missing references, or say anything like 'no relevant reference'.\n"
        "- End by advising evaluation by a licensed dentist.\n"
    )

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

# ========= MAIN =========
if __name__ == "__main__":
    q = input("> ").strip()

    # Out-of-scope (non-dental): one-line bilingual apology, nothing else
    if not is_dental_scope(q):
        print("\n--- ANSWER ---\n")
        print(SCOPE_APOLOGY)
        raise SystemExit(0)

    # Disallowed clinical requests: one-line bilingual apology, nothing else
    if is_disallowed_clinical_request(q):
        print("\n--- ANSWER ---\n")
        print(CLINICAL_APOLOGY)
        raise SystemExit(0)

    match, score, ar, idx = dataset_match(q)

    if match and score >= SIM_THRESHOLD:
        answer = match["ar_a"] if ar else match["en_a"]
        rag_query = match["ar_q"] if ar else match["en_q"]
        fields = match["field"]
    else:
        answer = gpt_style_answer(q)
        rag_query = q
        fields = None

    refs = rag_refs(rag_query, fields)

    print("\n--- ANSWER ---\n")
    print(answer)

    if refs:
        print("\nReferences:")
        for r in refs:
            print(f"- {r}")