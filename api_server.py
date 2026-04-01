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

    # ── Gate 1: Direct prescription / treatment plan request ──────────────────
    # Always runs first, no exceptions.
    if rag.is_treatment_request(q):
        return {
            "answer":     rag.refusal_treatment(q),
            "references": [],
            "_debug":     {"gate": "prescription_block"},
        }

    # ── Gate 2: Social / conversational exchange ──────────────────────────────
    # Short-circuits before any embedding or RAG call.
    # Handles: greetings, thanks, "I have a question", "go ahead" prompts, etc.
    if rag.is_social_exchange(q):
        return {
            "answer":     rag.social_response(q),
            "references": [],
            "_debug":     {"gate": "social_exchange"},
        }

    # ── Context signals ───────────────────────────────────────────────────────
    dental_in_history = rag.history_has_dental_context(history)
    dental_in_query   = rag.has_dental_signal(q)
    hard_blocked      = rag.is_out_of_scope(q)

    # ── Gate 3: Hard non-dental block ─────────────────────────────────────────
    # Blocked only when ALL THREE conditions are true:
    #   • query contains a hard non-dental keyword
    #   • query contains no dental signal
    #   • history contains no dental context
    #
    # This means:
    #   "حافظ مسافة"                 → dental_in_query=True  → NOT blocked
    #   "Is it dangerous?" + history → dental_in_history=True → NOT blocked
    #   "What can I do in this case?" + history → same → NOT blocked
    #   "Best Porsche model"          → blocked
    if hard_blocked and not dental_in_query and not dental_in_history:
        return {
            "answer":     rag.refusal_scope(q),
            "references": [],
            "_debug": {
                "gate":               "hard_scope_block",
                "dental_in_query":    dental_in_query,
                "dental_in_history":  dental_in_history,
            },
        }

    # ── Embed once — reused for dataset match and RAG ─────────────────────────
    qv = rag.embed(q)

    # ── Dataset shortcut ──────────────────────────────────────────────────────
    # Skipped when dental_in_history is True: follow-ups need GPT + history,
    # not a pre-written dataset answer.
    if not dental_in_history:
        match, score, ar, idx = rag.dataset_match(q, qv)
        if match and score >= rag.SIM_THRESHOLD:
            answer = match["ar_a"] if ar else match["en_a"]
            _, refs, _, debug = rag.rag_retrieve(qv)
            return {
                "answer":     answer,
                "references": refs,
                "_debug": {
                    **debug,
                    "gate":  "dataset_match",
                    "score": round(score, 4),
                },
            }

    # ── RAG retrieval ─────────────────────────────────────────────────────────
    context_chunks, refs, is_off_topic, debug = rag.rag_retrieve(qv)

    # ── Gate 4: RAG off-topic ─────────────────────────────────────────────────
    # Only blocks when Pinecone signals off-topic AND no dental context anywhere.
    # dental_in_query=True or dental_in_history=True → always proceed to GPT.
    if is_off_topic and not dental_in_query and not dental_in_history:
        return {
            "answer":     rag.refusal_scope(q),
            "references": [],
            "_debug": {
                **debug,
                "gate":               "rag_score_block",
                "dental_in_query":    dental_in_query,
                "dental_in_history":  dental_in_history,
            },
        }

    # ── GPT generation ────────────────────────────────────────────────────────
    # history is always passed — GPT uses it to understand follow-up context.
    answer = rag.gpt_style_answer(q, context_chunks, history=history)
    return {
        "answer":     answer,
        "references": refs,
        "_debug": {
            **debug,
            "gate":               "gpt_generation",
            "dental_in_query":    dental_in_query,
            "dental_in_history":  dental_in_history,
        },
    }
