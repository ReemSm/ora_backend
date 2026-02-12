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

MAX_GPT_TOKENS = 220

client = OpenAI()
pc = Pinecone()
index = pc.Index(PINECONE_INDEX)

# ========= DATASET =========
# (your dataset stays exactly the same — not shown shortened here for brevity)
# KEEP YOUR EXISTING DATASET BLOCK UNCHANGED

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

# ========= EMBEDDING =========
def embed(text):
    return client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb) if na and nb else 0.0

# ========= SCOPE & SAFETY =========
DENTAL_KEYWORDS = [
    "tooth","teeth","gum","gums","dental","dentist","implant","filling",
    "cavity","decay","root canal","pulp","bleeding","occlusion",
    "سن","أسنان","اللثة","لثة","طبيب أسنان","زرعة","حشوة",
    "تسوس","قناة الجذر","لب السن","نزيف","إطباق"
]

def in_scope_dental(q):
    ql = q.lower()
    return any(k in ql for k in DENTAL_KEYWORDS)

def is_treatment_request(q):
    ql = q.lower()
    return any(x in ql for x in [
        "treatment plan", "prescription", "medication", "what should i take",
        "give me medicine", "dosage", "plan for me",
        "خطة علاج", "وصفة", "دواء", "ايش اخذ", "ماذا آخذ"
    ])

def refuse_treatment(q):
    if is_ar(q):
        return "عذرًا، لا يمكنني تقديم خطة علاج أو وصفة طبية. يُرجى استشارة طبيب أسنان مرخص."
    return "Sorry, I cannot provide treatment plans or prescriptions. Please consult a licensed dentist."

def refuse_out_of_scope(q):
    if is_ar(q):
        return "هذا السؤال خارج نطاق هذا التطبيق المتخصص بصحة الفم والأسنان."
    return "This question is outside the scope of this oral health application."

# ========= DATASET MATCH =========
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
        if title:
            strong.append(title)

    if len(strong) < 2:
        return []

    return list(dict.fromkeys(strong))[:TOP_K_FINAL]

# ========= GPT FALLBACK =========
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
        "- End by advising evaluation by a licensed dentist.\n"
        "Always use formal professional language. Never mirror slang or informal user phrasing.\n"
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

    # 🔒 Treatment refusal
    if is_treatment_request(q):
        answer = refuse_treatment(q)
        print("\n--- ANSWER ---\n")
        print(answer)
        exit()

    # 🔒 Scope refusal
    if not in_scope_dental(q):
        answer = refuse_out_of_scope(q)
        print("\n--- ANSWER ---\n")
        print(answer)
        exit()

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
