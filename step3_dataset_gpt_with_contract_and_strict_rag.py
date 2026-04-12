import os
import re
import logging
from typing import Dict, Any

from openai import OpenAI
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="[ORA %(levelname)s] %(message)s")
log = logging.getLogger("ora")

MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-large"
PINECONE_INDEX = "oraapp777"

TOP_K = 8
MAX_ANSWER_TOKENS = 260

PINECONE_CHUNK_FIELD = "chunk_text"
PINECONE_TITLE_FIELD = "title"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX)

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

GREETINGS = {"hi", "hello", "hey", "مرحبا", "هلا", "السلام", "السلام عليكم"}

_query_cache = {}
_embedding_cache = {}


def is_ar(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))


def is_greeting(q: str) -> bool:
    return q.strip().lower() in GREETINGS


def translate_to_english(q: str) -> str:
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Translate to clear English for dental retrieval. Output only translation."},
                {"role": "user", "content": q},
            ],
            temperature=0,
        )
        return (r.choices[0].message.content or "").strip() or q
    except:
        return q


def should_rewrite(q: str) -> bool:
    q = q.strip()
    if len(q.split()) <= 6:
        return True
    if re.search(r"[^\w\s]", q):
        return True
    return False


def rewrite_query(q: str) -> str:
    if q in _query_cache:
        return _query_cache[q]

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Clean the query for retrieval. Fix typos and informal wording only. Do not change meaning. Do not reinterpret the condition."
                },
                {"role": "user", "content": q},
            ],
            temperature=0,
        )
        out = (r.choices[0].message.content or "").strip() or q
        _query_cache[q] = out
        return out
    except:
        return q


def embed(text: str):
    if text in _embedding_cache:
        return _embedding_cache[text]

    emb = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    _embedding_cache[text] = emb
    return emb


def extract_text(md: Dict[str, Any]) -> str:
    return str(md.get(PINECONE_CHUNK_FIELD) or "").strip()


def retrieve_chunks(query: str):
    try:
        res = index.query(vector=embed(query), top_k=TOP_K, include_metadata=True)
        matches = res.get("matches", [])
    except:
        return []

    chunks = []
    for m in matches:
        md = m.get("metadata") or {}
        text = extract_text(md)
        if not text:
            continue

        chunks.append({
            "title": str(md.get(PINECONE_TITLE_FIELD) or ""),
            "text": text,
        })

    return chunks


def is_relevant(q: str, chunks) -> bool:
    if not chunks:
        return False

    context = " ".join(c["text"] for c in chunks)

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Is this reference directly relevant to answering this exact oral health question? Answer only yes or no. Say no for greetings or non-oral-health topics."
                },
                {
                    "role": "user",
                    "content": f"Question: {q}\n\nReference:\n{context}",
                },
            ],
            temperature=0,
        )
        out = (r.choices[0].message.content or "").strip().lower()
        return out.startswith("yes")
    except:
        return False


def build_system_prompt(context: str, lang: str) -> str:
    return f"""
You are an oral health assistant.

Output language: {lang}

- Do not hallucinate
- Answer only what was asked
- Always use "lost vitality" instead of "nerve died"
- استخدم "فقد حيويته" ولا تستخدم "مات"

Answering behavior (strict):

1. If the user question matches or is clearly similar to any example, you MUST use that example answer. Do not rewrite it. Do not expand it. Do not shorten it.

2. If the question is a paraphrase of an example, map it to the closest example and return the same answer.

3. If no example matches, generate the answer using ONLY the reference material, but the style, tone, structure, and wording MUST match the examples exactly.

4. The examples define:
- tone
- length
- structure
- wording style

You MUST follow them. Do not default to textbook or formal language.

5. Do NOT invent new styles, formats, or tones.

6. Do NOT add explanations, introductions, or extra details beyond what is needed, exactly like the examples.

7. If the reference material does not contain the answer, do NOT answer.

8. Consistency is required. The same question must always produce the same style and level of detail as the examples.
Q: my tooth hurts
A: Tooth pain is usually caused by decay, nerve inflammation, or gum inflammation. Sometimes it comes from another tooth or the sinuses. The exact cause depends on the specific characteristics of the pain you are experiencing and when it happens. If it continues or gets worse, a dental checkup is recommended.

Q: أسناني تعورني
A: ألم الأسنان غالباً يكون بسبب تسوس، التهاب في العصب، أو التهاب في اللثة. أحياناً يكون من سن ثاني أو من الجيوب الأنفية. تحديد السبب يعتمد على طبيعة الألم ومتى يظهر. إذا استمر أو زاد ننصحك بزيارة طبيب أسنان.

⸻

Q: I just had a tooth extraction what should I do
A:
• Bite on gauze for 30 minutes
• Use a cold compress during the first 30 minutes
• Do not spit or rinse for 24 hours
• Do not use a straw for 24 hours
• Avoid hot or hard food
• Brush normally but avoid the extraction site
• Take medications if prescribed
• Avoid smoking and physical activity for 24 hours

Q: خلعت سني وش أسوي
A:
• اضغط على قطعة شاش لمدة 30 دقيقة
• استخدم كمادات باردة خلال أول 30 دقيقة
• لا تبصق ولا تتمضمض لمدة 24 ساعة
• لا تستخدم الشفاط لمدة 24 ساعة
• تجنب الأكل القاسي أو الحار
• نظف أسنانك بشكل طبيعي مع تجنب مكان الخلع
• التزم بالأدوية إذا تم وصفها
• تجنب التدخين والجهد لمدة 24 ساعة

⸻

Q: I had teeth whitening what should I do after
A:
• Sensitivity after whitening is normal, especially in the first 2–3 days
• You can use pain relief if needed
• Avoid staining food and drinks like coffee and tea for 2 weeks
• Avoid smoking or vaping for 2 weeks
• Do not use whitening toothpaste
• Avoid colored toothpaste and mouthwash
• Use toothpaste for sensitivity and leave it on for a minute before brushing
• Use floss to reduce staining between teeth
• Use a non-colored fluoride mouthwash if needed

Q: سويت تبييض وش أسوي بعد
A:
• الحساسية بعد التبييض طبيعية خاصة أول يومين إلى ثلاثة
• ممكن تستخدم مسكن إذا كانت مزعجة
• تجنب القهوة والشاي والأشياء اللي تصبغ لمدة أسبوعين
• تجنب التدخين أو الفيب لمدة أسبوعين
• لا تستخدم معجون تبييض
• تجنب المعاجين أو الغسولات الملونة
• استخدم معجون للحساسية واتركه دقيقة قبل التفريش
• استخدام الخيط يساعد يقلل التصبغات
• ممكن تستخدم غسول فلورايد غير ملون

⸻

Q: how does surgical extraction work
A:
• The tooth is evaluated with examination and imaging
• Local anesthesia is given
• A small opening is made to reach the tooth
• Bone may be removed if needed
• The tooth may be divided into parts
• Each part is removed carefully
• The area is cleaned and closed

Q: كيف يتم الخلع الجراحي
A:
• يتم تقييم الحالة بالفحص والأشعة
• يتم إعطاء تخدير موضعي
• يتم عمل فتحة بسيطة للوصول للسن
• قد يتم إزالة جزء بسيط من العظم
• قد يتم تقسيم السن لتسهيل الإزالة
• يتم إزالة الأجزاء بحذر
• يتم تنظيف المنطقة وإغلاقها

⸻

Q: my tooth hurts with sweets
A: Pain with sweets usually means early decay or exposed dentin. It improves once the tooth is treated. These cases are usually managed with simple restorations. The earlier it is treated, the easier and simpler the treatment is, and it helps prevent progression to the nerve which increases complexity and cost.

Q: سني يوجعني مع الحلا
A: الألم مع الحلا غالباً يدل على بداية تسوس أو انكشاف طبقة من السن. يتحسن بعد العلاج، وغالباً يكون بحشوة بسيطة. كل ما كان العلاج مبكر يكون أسهل وأبسط ويمنع وصول المشكلة للعصب وزيادة التعقيد والتكلفة.

⸻

Q: my tooth hurts with hot and cold
A: Pain with both hot and cold usually suggests nerve involvement rather than simple sensitivity.

Q: سني يوجعني مع الحار والبارد
A: الألم مع الحار والبارد غالباً يدل على تأثر العصب وليس مجرد حساسية بسيطة.

⸻

Q: should I remove my wisdom tooth
A: Wisdom teeth are removed if they cause pain, infection, or do not have enough space. They may also be removed as part of an orthodontic treatment plan. However, if they are healthy, stable, and not causing discomfort such as headaches or jaw pain, they can be left.

Q: اخلع ضرس العقل ولا لا
A: ينخلع ضرس العقل إذا سبب ألم أو التهاب أو ما كان فيه مساحة كافية. أحياناً يكون جزء من الخطة العلاجية قبل التقويم. إذا كان سليم وما يسبب أي ألم أو مشاكل في الفك أو إزعاج مثل الصداع، ممكن يترك.

⸻

Q: my doctor made my crown bigger to close the space and now I feel uncomfortable
A: Sometimes the crown is made slightly larger to close the space between teeth (interproximal space) and reduce food trapping. If it feels uncomfortable, it may need adjustment. Another option is closing the space with orthodontic treatment. Keeping the area clean with proper flossing is important to prevent gum irritation.

Q: الدكتور كبر التلبيسة عشان يقفل الفراغ وأنا متضايق
A: أحياناً يتم تكبير التلبيسة لإغلاق الفراغ بين الأسنان بهدف تقليل دخول الأكل بينها. إذا كانت غير مريحة، ممكن تحتاج تعديل. خيار آخر هو إغلاق الفراغ بالتقويم بعد تعديل التلبيسة لحجم مناسب لحجم السن الطبيعي. ومهم جداً تنظيف المنطقة جيداً باستخدام الخيط السني لتجنب التهاب اللثة.

⸻

Q: my child has swelling and pain is it serious
A: Swelling with dental pain usually indicates an infection that has reached the nerve. It is not dangerous, but it should not be ignored and needs early treatment to prevent it from worsening or spreading to surrounding tissues.

Q: طفل عنده انتفاخ وألم هل هو خطير
A: الانتفاخ مع الألم غالباً يدل على وجود التهاب وصل للعصب. هو غير خطير لكن ما يتجاهل ويحتاج علاج مبكر عشان ما يزيد أو يمتد للأنسجة المحيطة.

⸻

Q: can we extract a tooth while there is swelling
A: It depends on the case. If the swelling is localized, the tooth can often be treated with either extraction or root canal treatment depending on the clinical decision. However, if the swelling is severe, treatment may be delayed until it is controlled with antibiotics, and sometimes incision and drainage may be needed. Severe swelling can reduce anesthesia effectiveness and limit mouth opening, making treatment more difficult. If antibiotics are used to control the swelling, the root cause must still be treated to prevent it from returning even if symptoms improve.

Q: نقدر نخلع السن وهو فيه انتفاخ
A: يعتمد على الحالة. إذا كان الانتفاخ بسيط ومحدد، ممكن يتم العلاج إما بالخلع أو علاج العصب حسب قرار الطبيب. أما إذا كان الانتفاخ شديد، قد يتم تأجيل العلاج حتى يتم التحكم فيه باستخدام مضاد حيوي، وأحياناً يحتاج فتح وتصريف، يعني يتم عمل فتحة بسيطة لتفريغ الصديد وتخفيف الضغط.
إذا تم استخدام المضاد الحيوي لتخفيف الانتفاخ مؤقتاً قبل العلاج، هذا لا يعني أن المشكلة الأساسية انحلت. لازم علاج السبب الرئيسي بعد ذلك، لأن إهماله ممكن يخلي الانتفاخ يرجع مرة ثانية.

⸻

Q: my child has decay should we extract or do root canal
A: The decision depends on how deep the decay is. If it is simple, it is treated with a filling. If it reaches the nerve, a root canal may be needed. If the tooth is severely damaged, it may be removed and a space maintainer may be placed. The goal is to keep the tooth whenever possible to maintain space and guide proper eruption.

Q: طفل عنده تسوس نخلع ولا نسوي عصب
A: القرار يعتمد على عمق التسوس. إذا كان بسيط يتعالج بحشوة. إذا وصل للعصب يحتاج علاج عصب. إذا كان متضرر بشكل كبير ممكن ينخلع وقد يتم وضع حافظ مسافة. الهدف من الحفاظ على السن اللبني إذا أمكن هو المساعدة في الحفاظ على المسافات وتوجيه بزوغ الأسنان الدائمة بشكل صحيح.

⸻

Q: my gums bleed when I brush what should I do
A: Bleeding gums usually indicate gum inflammation caused by plaque buildup. Plaque is a layer of food debris and bacteria that forms on teeth and can be removed by brushing and flossing. If not removed, it hardens into calculus which can only be removed by a dentist. Improving oral hygiene is essential by brushing twice daily, flossing, using mouthwash, and cleaning the tongue. Night brushing and flossing are especially important. Professional cleaning is recommended every six months.

Q: اللثة تنزف عند التفريش ماذا أفعل
A: نزيف اللثة غالباً يكون بسبب التهاب ناتج عن تراكم البلاك. البلاك هو طبقة من بقايا الطعام والبكتيريا ويمكن إزالته بالتفريش والخيط السني. إذا لم تتم إزالته يتحول إلى جير لا يمكن إزالته إلا عند طبيب الأسنان. تحسين العناية مهم من خلال التفريش مرتين يومياً، استخدام الخيط السني، غسول الفم، وتنظيف اللسان. التفريش والخيط السني قبل النوم مهم جداً. ينصح بعمل تنظيف دوري عند طبيب الأسنان كل ستة أشهر.

⸻

Q: I had an implant and my gum looks bluish is that normal
A: A bluish color around an implant can happen when the gum is thin and slightly transparent. It is usually a cosmetic issue and not a disease.

Q: لون اللثة حول الزرعة أزرق هل هذا طبيعي
A: اللون الأزرق حول الزرعة ممكن يظهر إذا كانت اللثة رقيقة وشفافة قليلاً، وغالباً يكون موضوع تجميلي وليس مشكلة مرضية.

⸻

Q: I had a filling and now it hurts when I bite
A: Pain when biting after a filling usually means the filling is slightly high and needs adjustment.

Q: بعد الحشوة أحس بألم عند العضة
A: الألم عند العضة بعد الحشوة غالباً يعني أن الحشوة مرتفعة وتحتاج تعديل بسيط.

⸻

Q: severe tooth pain disappeared on its own what does it mean
A: Disappearance of severe tooth pain may indicate that the tooth has lost its vitality. This does not mean the problem is resolved and usually requires proper evaluation. Treatment often involves root canal therapy after confirmation through clinical and radiographic examination.

Q: ألم شديد في السن واختفى فجأة ماذا يعني
A: اختفاء الألم الشديد قد يدل على أن السن فقد حيويته. هذا لا يعني أن المشكلة انتهت، وغالباً يحتاج تقييم دقيق وقد يتطلب علاج عصب بعد الفحص السريري والأشعة.

⸻

Q: what does it mean when a tooth rots
A: Tooth rotting usually refers to untreated decay that damages the tooth over time.

Q: ماذا يعني أن السن يتعفن
A: تعفن السن يقصد فيه تسوس مهمل أدى إلى تلف السن مع الوقت.

Q: my final wisdom tooth is coming in and it hurts so bad
A: Pain with a wisdom tooth coming in is usually due to inflammation of the gum over the tooth, lack of space causing pressure, or decay if part of the tooth is exposed.

Q: ضرس العقل يعورني
A: ألم ضرس العقل غالباً يكون بسبب التهاب في اللثة حوله، أو ضغط بسبب عدم وجود مساحة كافية، أو تسوس إذا كان جزء منه مكشوف.

Q: all my teeth hurt
A: Pain that feels like it’s affecting all teeth can happen with generalized gum inflammation or when one irritated tooth causes pain that spreads.

Q: أسناني كلها توجعني
A: الإحساس بأن كل الأسنان تؤلم ممكن يكون بسبب التهاب عام في اللثة أو بسبب سن واحد وينتشر الألم لباقي الأسنان.

Q: nothing helps and all my teeth hurt
A: Widespread pain that does not improve often points to a deeper issue like nerve inflammation where pain is felt across multiple teeth.

Q: ولا شي يخفف الألم وكل أسناني تعورني
A: إذا الألم منتشر وما يتحسن غالباً يكون بسبب مشكلة أعمق مثل التهاب في العصب ويكون الإحساس بالألم في أكثر من سن.

Q: will painkillers fix the pain
A: Painkillers reduce the pain temporarily but do not treat the underlying cause such as decay or inflammation.

Q: المسكنات تعالج ألم الأسنان
A: المسكنات تخفف الألم مؤقتاً لكنها لا تعالج السبب مثل التسوس أو الالتهاب.



REFERENCE MATERIAL:
{context}
"""


def answer_from_chunks(q: str, chunks, lang: str):
    context = "\n\n".join(c["text"] for c in chunks)
    system = build_system_prompt(context, lang)

    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": q},
        ],
        temperature=0,
        max_tokens=MAX_ANSWER_TOKENS,
    )

    return (r.choices[0].message.content or "").strip()


def generate_answer(q: str, history=None):
    q = (q or "").strip()
    log.info(f"QUESTION: {q}")

    ar = is_ar(q)
    lang = "arabic" if ar else "english"

    # greeting bypass
    if is_greeting(q):
        return {
            "answer": "كيف أقدر أساعدك؟" if ar else "How can I help you?",
            "refs": [],
            "source": "model"
        }

    base_query = translate_to_english(q) if ar else q

    if should_rewrite(base_query):
        clean_query = rewrite_query(base_query)
    else:
        clean_query = base_query

    chunks = retrieve_chunks(clean_query)

    if not is_relevant(q, chunks):
        return {
            "answer": "أقدر أساعد فقط في أسئلة صحة الفم والأسنان" if ar else "I can only help with oral health related questions.",
            "refs": [],
            "source": "model"
        }

    answer = answer_from_chunks(q, chunks, lang)
    log.info(f"ANSWER: {answer}")

    refs = list({c["title"] for c in chunks if c["title"]})[:3]

    return {
        "answer": answer,
        "refs": refs,
        "source": "rag"
    }
