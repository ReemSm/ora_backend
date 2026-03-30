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
    role: str       # must be "user" or "assistant"
    content: str

class AskRequest(BaseModel):
    query: str
    history: Optional[List[HistoryTurn]] = []

@app.post("/ask")
def ask(req: AskRequest):
    q = req.query.strip()

    # Serialize history to plain dicts for the RAG module
    history = [h.dict() for h in (req.history or [])]

    if rag.is_treatment_request(q):
        return {"answer": rag.refusal_treatment(q), "references": []}

    if rag.is_out_of_scope(q):
        return {"answer": rag.refusal_scope(q), "references": []}

    qv = rag.embed(q)

    # [Fix 7] If conversation history exists, bypass dataset matching entirely.
    #
    # Why: pre-written dataset answers cannot interpret follow-up questions.
    # A follow-up like "is that serious?" or "what happens if I ignore it?"
    # needs the prior context to give a coherent answer. Routing it through
    # a dataset match would either return the wrong canned answer or fall
    # through to GPT anyway — but without history, making it useless.
    #
    # When history is present: always go to GPT with full context.
    # When no history (first message): use dataset matching as normal.
    if history:
        context_chunks, refs = rag.rag_retrieve(qv, expected_fields=None)
        answer = rag.gpt_style_answer(q, context_chunks, history=history)
    else:
        match, score, ar, idx = rag.dataset_match(q, qv)

        if match and score >= rag.SIM_THRESHOLD:
            answer = match["ar_a"] if ar else match["en_a"]
            fields = match["field"]
            _, refs = rag.rag_retrieve(qv, fields)
        else:
            context_chunks, refs = rag.rag_retrieve(qv, expected_fields=None)
            answer = rag.gpt_style_answer(q, context_chunks)

    return {"answer": answer, "references": refs}
