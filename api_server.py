from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import step3_dataset_gpt_with_contract_and_strict_rag as rag

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HistoryTurn(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class AskRequest(BaseModel):
    query: str
    history: Optional[List[HistoryTurn]] = []

@app.post("/ask")
def ask(req: AskRequest):
    q       = req.query.strip()
    history = [h.dict() for h in (req.history or [])]

    # --- Safety + scope gates (always run first) ---
    if rag.is_treatment_request(q):
        return {"answer": rag.refusal_treatment(q), "references": []}

    if rag.is_out_of_scope(q):
        return {"answer": rag.refusal_scope(q), "references": []}

    qv = rag.embed(q)

    # --- Follow-up routing (history present) ---
    # When a conversation history exists, bypass dataset matching entirely.
    # Pre-written dataset answers cannot interpret follow-up questions —
    # they would return canned text with no awareness of what was said before.
    # GPT with full history handles this correctly.
    if history:
        context_chunks, refs, is_off_topic = rag.rag_retrieve(qv)
        if is_off_topic and not rag.has_dental_signal(q):
            return {"answer": rag.refusal_scope(q), "references": []}
        answer = rag.gpt_style_answer(q, context_chunks, history=history)
        return {"answer": answer, "references": refs}

    # --- First-message routing (no history) ---
    match, score, ar, idx = rag.dataset_match(q, qv)

    if match and score >= rag.SIM_THRESHOLD:
        answer = match["ar_a"] if ar else match["en_a"]
        _, refs, _ = rag.rag_retrieve(qv, match["field"])
    else:
        context_chunks, refs, is_off_topic = rag.rag_retrieve(qv)
        if is_off_topic and not rag.has_dental_signal(q):
            return {"answer": rag.refusal_scope(q), "references": []}
        answer = rag.gpt_style_answer(q, context_chunks)

    return {"answer": answer, "references": refs}
