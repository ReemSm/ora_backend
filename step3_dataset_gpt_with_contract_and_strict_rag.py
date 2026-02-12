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
MIN_AUTHORITY = 0.40   # relaxed so references appear again

MAX_GPT_TOKENS = 220

client = OpenAI()
pc = Pinecone()
index = pc.Index(PINECONE_INDEX)

# ========= POLITE REFUSALS =========
SCOPE_REFUSAL = "Sorry, this request is outside the scope of this dental application. / عذرًا، هذا الطلب خارج نطاق هذا التطبيق المتخصص في طب الأسنان."
CLINICAL_REFUSAL = "Sorry, I cannot assist with diagnosis, treatment plans, or prescriptions. / عذرًا، لا يمكنني المساعدة في التشخيص أو خطط العلاج أو الوصفات الطبية."

# ========= SIMPLE SCOPE DETECTION =========
DENTAL_TERMS = [
    "tooth","teeth","gum","gums","implant","filling","crown","bridge","root canal",
    "caries","decay","mouth","oral","jaw","brace","orthodont",
    "سن","أسنان","لثة","زرعة","حشوة","تلبيسة","تاج","عصب","فم","فك","تقويم","تسوس"
]

CLINICAL_TERMS = [
    "diagnosis","diagnose","treatment plan","prescribe","prescription","dosage","dose",
    "تشخيص","خطة علاج","وصفة","جرعة","دواء","مضاد"
]

def is_dental(q):
    ql = q.lower()
    return any(t in ql for t in DENTAL_TERMS) or any(t in q for t in DENTAL_TERMS)

def is_clinical(q):
    ql = q.lower()
    return any(t in ql for t in CLINICAL_TERMS) or any(t in q for t in CLINICAL_TERMS)

# ========= DATASET =========
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

# ========= HELPERS =========
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

# ========= DATASET MATCH =========
def dataset_match(q):
    ar = is_ar(q)
    qv = embed(q)

    best, score = None, -1
    for d in DATASET:
        dv = embed(d["ar_q"] if ar else d["en_q"])
        s = cosine(qv, dv)
        if s > score:
            score, best = s, d

    return best, score, ar

# ========= STRICT RAG =========
def rag_refs(query_text, expected_fields):
    qv = embed(query_text)
    res = index.query(vector=qv, top_k=TOP_K_RAW, include_metadata=True)

    strong = []
    for m in res.get("matches", []):
        md = m.get("metadata", {})

        if float(md.get("authority_score", 0)) < MIN_AUTHORITY:
            continue

        specialty = str(md.get("specialty", "")).lower()
        if expected_fields and not any(f.lower() in specialty for f in expected_fields):
            continue

        title = md.get("title")
        if title and title not in strong:
            strong.append(title)

    return strong[:TOP_K_FINAL]   # no empty message ever

# ========= GPT FALLBACK =========
def gpt_style_answer(q):
    system = (
        "Write in the exact style and tone of a formal dental educational dataset.\n"
        "Rules:\n"
        "- One explanation only\n"
        "- No bullet points\n"
        "- No casual tone\n"
        "- No treatment or prescriptions\n"
        "- End advising consultation with a licensed dentist\n"
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

    # scope enforcement
    if not is_dental(q):
        print("\n--- ANSWER ---\n")
        print(SCOPE_REFUSAL)
        exit()

    if is_clinical(q):
        print("\n--- ANSWER ---\n")
        print(CLINICAL_REFUSAL)
        exit()

    match, score, ar = dataset_match(q)

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
