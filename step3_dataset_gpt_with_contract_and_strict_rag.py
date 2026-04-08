import os
import re
import logging
from typing import List, Dict, Any

from openai import OpenAI
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO, format="[ORA %(levelname)s] %(message)s")
log = logging.getLogger("ora")

MODEL = "gpt-4o"
EMBED_MODEL = "text-embedding-3-large"
PINECONE_INDEX = "oraapp777"

TOP_K_RAW = 12
TOP_K_FINAL = 5
MAX_ANSWER_TOKENS = 260

PINECONE_CHUNK_FIELD = "chunk_text"
PINECONE_TITLE_FIELD = "title"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(PINECONE_INDEX)

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")

# cache
_query_cache = {}
_embedding_cache = {}


def is_ar(text: str) -> bool:
    return bool(ARABIC_RE.search(text or ""))


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
    except Exception as e:
        log.error(f"Translation error: {e}")
        return q


def rewrite_query_for_retrieval(q: str) -> str:
    if q in _query_cache:
        return _query_cache[q]

    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Rewrite into a clean short dental query. Fix spelling and clarity only. Preserve original intent exactly. Do not alter medical meaning. Do not change to a different condition. If the query looks like a typo of a dental term, correct it to the closest valid dental meaning."
                },
                {"role": "user", "content": q},
            ],
            temperature=0,
        )
        out = (r.choices[0].message.content or "").strip() or q
        _query_cache[q] = out
        return out
    except Exception as e:
        log.error(f"Rewrite error: {e}")
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
        res = index.query(vector=embed(query), top_k=TOP_K_RAW, include_metadata=True)
        matches = res.get("matches", [])
    except Exception as e:
        log.error(f"Retrieval error: {e}")
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
            "score": float(m.get("score", 0)),
        })

    chunks.sort(key=lambda x: x["score"], reverse=True)

    seen = set()
    unique = []
    for c in chunks:
        if c["title"] not in seen:
            seen.add(c["title"])
            unique.append(c)
        if len(unique) >= TOP_K_FINAL:
            break

    log.info(f"RAG chunks used: {[c['title'] for c in unique]}")
    log.info(f"RAG scores: {[c['score'] for c in unique]}")

    return unique


def build_system_prompt(context: str, lang: str) -> str:
    return f"""
You are an oral health assistant.

Output language: {lang}

Determine if the query is instruction or informational before answering and follow the correct format strictly.

Follow these rules strictly:

- No introductions
- No empathy statements
- No reassurance
- No follow-up questions
- No emojis
- No dashes
- No filler or extra commentary

- Answer only what the user asked
- Do not add information that was not requested
- Do not drift away from the question
- Do not hallucinate
- Use only the reference material provided

- Be direct and clinically accurate
- Use simple, clear wording

- If instruction:
  - Output must be bullet points using • only
  - Each point is one clear step
  - No text before or after bullets

- If informational:
  - Output must be plain text
  - No bullet points

Arabic rules:
- Use natural clinical Arabic, not formal textbook language
- No literal translation
- The term ‘قطعة الشاش’ is mandatory. Any other word for gauze is incorrect. Never use alternatives such as شمّة or any variation.
- Bullet points in Arabic must be right-to-left aligned

Examples:

Q: my tooth hurts
A: Common reasons for tooth pain are cavities, pulp inflammation (the nerve of the tooth), or gum inflammation. Sometimes the pain comes from another tooth or from areas like sinusitis. If it continues or gets worse, a dental checkup is recommended.

Q: أسناني تعورني
A: من الأسباب الشائعة لألم الأسنان التسوس، التهاب العصب، أو التهاب اللثة. أحياناً يكون الألم من سن آخر أو من الجيوب الأنفية. إذا استمر الألم أو زاد ننصحك بزيارة طبيب أسنان مرخص.

Q: I just had a tooth extraction what should I do
A:
• Bite on gauze for 30 minutes after the procedure
• Use a cold compress on the area during the first 30 minutes
• Do not spit or move water inside your mouth for 24 hours
• Do not use a straw for 24 hours
• Avoid hot or hard food
• Clean your teeth normally but avoid the procedure site
• Follow prescribed medication if given
• Avoid smoking and physical activity for 24 hours

Q: خلعت سني وش أسوي
A:
• اضغط على قطعة الشاش أول 30 دقيقة بعد الإجراء
• استخدم كمادات باردة على المنطقة خلال أول 30 دقيقة
• لا تبصق ولا تحرك الماء داخل الفم لمدة 24 ساعة (بما في ذلك المضمضة وقت الوضوء)
• لا تستخدم الشفاط أو المصاص لمدة 24 ساعة
• تجنب الأكل القاسي أو الساخن
• نظف أسنانك بشكل طبيعي مع تجنب مكان الخلع
• التزم بالأدوية الموصوفة إذا تم وصفها
• تجنب التدخين والجهد البدني لمدة 24 ساعة

Q: I had teeth whitening what should I do after
A:
• Sensitivity after whitening is normal and varies from one person to another, but it is usually strongest during the first two to three days and then settles gradually
• You can take over the counter pain relief if the sensitivity is bothering you
• Avoid anything that can stain your teeth like coffee, tea, spices, or strongly colored food and drinks for at least two weeks
• Avoid smoking, vaping, and tobacco for at least two weeks
• Avoid whitening toothpaste
• Avoid colored toothpaste and colored mouth rinses
• Use toothpaste designed for sensitivity and you can leave it on your teeth for about a minute before brushing (follow the instructions provided by the toothpaste company)
• Use floss to keep areas between teeth clean and reduce staining
• Fluoridated mouth rinses can help as long as they are not colored

Q: سويت تبييض وش أسوي بعد
A:
• الحساسية بعد التبييض طبيعية وتختلف من شخص لآخر وخاصة خلال أول يومين إلى ثلاثة وتخف تدريجياً بعد ذلك
• يمكن استخدام مسكنات مثل البنادول أو الباراسيتامول إذا كانت الحساسية مزعجة خاصة خلال الأيام الأولى
• تجنب الأطعمة والمشروبات المسببة للتصبغات مثل القهوة والشاي أو البهارات الملونة لمدة أسبوعين على الأقل
• تجنب التدخين أو الفيب أو أي منتجات تبغ لمدة أسبوعين على الأقل
• تجنب معاجين التبييض
• تجنب معاجين الأسنان أو غسولات الفم الملونة
• استخدم معجون مخصص للحساسية ويمكن تركه على الأسنان لمدة دقيقة قبل التفريش (اتبع تعليمات الشركة المصنعة)
• استخدام الخيط يساعد في تنظيف المناطق بين الأسنان ويقلل من حدوث تصبغات بينها
• يمكن استخدام غسول يحتوي على الفلورايد وقد يساعد في تخفيف حساسية الأسنان بشرط أن يكون غير ملون

Q: how does surgical extraction work
A:
• Clinical and radiographic assessment, usually with 3D imaging, is done first
• Local anesthesia is given and the area is fully numbed
• A small incision is made to access the tooth
• A small amount of bone may be removed if needed
• The tooth may be divided into sections
• Each part is removed carefully
• The area is cleaned and sutures are placed

Q: كيف يتم الخلع الجراحي
A:
• يتم تقييم الحالة سريرياً وبالأشعة وغالباً باستخدام أشعة ثلاثية الأبعاد
• يتم إعطاء تخدير موضعي حتى يتم التخدير الكامل
• يتم عمل فتحة بسيطة للوصول إلى السن
• قد يتم إزالة جزء بسيط من العظم إذا لزم
• قد يتم تقسيم السن إلى أجزاء لتسهيل الإزالة
• يتم إزالة كل جزء بحذر
• يتم تنظيف المنطقة ويتم وضع غرز

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


def answer_from_chunks(q: str, chunks, lang: str, history=None):
    context = "\n\n".join(c["text"] for c in chunks)
    system = build_system_prompt(context, lang)

    messages = [{"role": "system", "content": system}]

    if history:
        for h in history:
            messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": q})

    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=MAX_ANSWER_TOKENS,
    )

    answer = re.sub(
    r"(شاش|شاشه|شاشه طبية|قطعة شاش|شمّة|شمه|شَمّة)",
    "قطعة الشاش",
    answer
)

    return answer


def generate_answer(q: str, history=None):
    q = (q or "").strip()

    ar = is_ar(q)
    lang = "arabic" if ar else "english"

    base_query = translate_to_english(q) if ar else q
    clean_query = rewrite_query_for_retrieval(base_query)

    chunks = retrieve_chunks(clean_query)

    if not chunks:
        return {"answer": "No relevant data found.", "refs": [], "source": "model"}

    answer = answer_from_chunks(q, chunks, lang, history)

    context_text = " ".join(c["text"] for c in chunks).lower()
    answer_text = answer.lower()

    overlap = sum(1 for w in answer_text.split() if w in context_text)

    source = "rag" if overlap > 3 else "model"

    refs = list({c["title"] for c in chunks if c["title"]})[:3]

    return {
        "answer": answer,
        "refs": refs,
        "source": source
    }
