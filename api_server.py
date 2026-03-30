from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import step3_dataset_gpt_with_contract_and_strict_rag as rag

app = FastAPI()

# --- CORS (required for Base44 browser calls) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    query: str

@app.post("/ask")
def ask(req: AskRequest):
    q = req.query.strip()

    # 🔒 1) BLOCK treatment / diagnosis / prescription
    if rag.is_treatment_request(q):
        return {
            "answer": rag.refusal_treatment(q),
            "references": []
        }

    # 🔒 2) BLOCK out-of-scope (non-dental)
    if rag.is_out_of_scope(q):
        return {
            "answer": rag.refusal_scope(q),
            "references": []
        }

    # [Fix 3] Compute query embedding exactly once.
    # This single vector is reused by both dataset_match() and rag_retrieve()
    # — previously each function triggered its own embed() call.
    qv = rag.embed(q)
    match, score, ar, idx = rag.dataset_match(q, qv)

    if match and score >= rag.SIM_THRESHOLD:
        # Dataset answer is pre-written — use it directly.
        # Still run rag_retrieve() for display references.
        answer = match["ar_a"] if ar else match["en_a"]
        fields = match["field"]
        _, refs = rag.rag_retrieve(qv, fields)

    else:
        # [Fix 2] GPT fallback: retrieve document content FIRST, then pass it
        # into gpt_style_answer() so GPT answers from real sources, not memory.
        context_chunks, refs = rag.rag_retrieve(qv, expected_fields=None)
        answer = rag.gpt_style_answer(q, context_chunks)

    return {
        "answer": answer,
        "references": refs
    }
