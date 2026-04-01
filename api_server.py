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
    role: str
    content: str

class AskRequest(BaseModel):
    query: str
    history: Optional[List[HistoryTurn]] = []

@app.post("/ask")
def ask(req: AskRequest):
    q       = req.query.strip()
    history = [h.dict() for h in (req.history or [])]

    # ── Gate 1: Prescription check (always runs, no exceptions) ──────────────
    if rag.is_treatment_request(q):
        return {
            "answer": rag.refusal_treatment(q),
            "references": [],
            "_debug": {"gate": "prescription_block"}
        }

    # ── Gate 2: Scope check — with critical fix for follow-ups ───────────────
    # [Fix 2] The previous version ran is_out_of_scope BEFORE checking history,
    # which meant every follow-up question was scope-checked in isolation.
    # "Is it dangerous?" fails every dental keyword test and was incorrectly blocked.
    #
    # Corrected order:
    # A) If history exists AND contains dental context → skip scope check entirely.
    #    The conversation is already established as dental.
    # B) If history exists but contains NO dental context → treat as new question
    #    and apply normal scope checks (prevents misuse via empty history injection).
    # C) No history → apply full scope checks normally.
    dental_context_from_history = history and rag.history_has_dental_context(history)

    if not dental_context_from_history:
        if rag.is_out_of_scope(q):
            return {
                "answer": rag.refusal_scope(q),
                "references": [],
                "_debug": {"gate": "keyword_scope_block"}
            }

    # ── Embed once, reuse everywhere ─────────────────────────────────────────
    qv = rag.embed(q)

    # ── Follow-up routing (history with dental context) ──────────────────────
    if dental_context_from_history:
        context_chunks, refs, is_off_topic, debug = rag.rag_retrieve(qv)

        # For follow-ups, only block on RAG score if ALSO missing dental signals
        # in BOTH current question AND history (belt-and-suspenders check).
        if is_off_topic and not rag.has_dental_signal(q) and not dental_context_from_history:
            return {
                "answer": rag.refusal_scope(q),
                "references": [],
                "_debug": {**debug, "gate": "rag_score_block_followup"}
            }

        answer = rag.gpt_style_answer(q, context_chunks, history=history)
        return {
            "answer": answer,
            "references": refs,
            "_debug": {**debug, "gate": "followup_gpt"}
        }

    # ── First-message routing ─────────────────────────────────────────────────
    match, score, ar, idx = rag.dataset_match(q, qv)

    if match and score >= rag.SIM_THRESHOLD:
        answer = match["ar_a"] if ar else match["en_a"]
        _, refs, _, debug = rag.rag_retrieve(qv, match["field"])
        return {
            "answer": answer,
            "references": refs,
            "_debug": {**debug, "gate": "dataset_match", "score": round(score, 4)}
        }

    context_chunks, refs, is_off_topic, debug = rag.rag_retrieve(qv)

    if is_off_topic and not rag.has_dental_signal(q):
        return {
            "answer": rag.refusal_scope(q),
            "references": [],
            "_debug": {**debug, "gate": "rag_score_block"}
        }

    answer = rag.gpt_style_answer(q, context_chunks)
    return {
        "answer": answer,
        "references": refs,
        "_debug": {**debug, "gate": "gpt_generation"}
    }
