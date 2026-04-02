from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import step3_dataset_gpt_with_contract_and_strict_rag as rag


# ─────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────
class HistoryTurn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    query: str
    history: Optional[List[HistoryTurn]] = []


# ─────────────────────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.post("/ask")
def ask(req: AskRequest):
    try:
        q = (req.query or "").strip()
        history = [h.dict() for h in (req.history or [])]

        if not q:
            return {
                "answer": "Empty query.",
                "references": [],
                "_debug": {"error": "empty_query"},
                "source": "error",
            }

        result = rag.generate_answer(q, history=history)

        return {
            "answer": result.get("answer", ""),
            "references": result.get("refs", []),
            "_debug": result.get("debug", {}),
            "source": result.get("source", "unknown"),
        }

    except Exception as e:
        return {
            "answer": "Server error.",
            "references": [],
            "_debug": {"error": str(e)},
            "source": "server_error",
        }


# ─────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "ORA backend"}
