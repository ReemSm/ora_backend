from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import step3_dataset_gpt_with_contract_and_strict_rag as rag


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MAX_QUERY_LENGTH = 500
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CONTENT_LENGTH = 500
REQUEST_TIMEOUT_SECONDS = 12

ALLOWED_ORIGINS = [
    "https://your-frontend-domain.com",
]

executor = ThreadPoolExecutor(max_workers=10)


# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[API %(levelname)s] %(message)s")
log = logging.getLogger("api")


# ─────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────
class HistoryTurn(BaseModel):
    role: str
    content: str

    @validator("role")
    def validate_role(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("Invalid role")
        return v

    @validator("content")
    def validate_content(cls, v):
        v = (v or "").strip()
        if len(v) > MAX_HISTORY_CONTENT_LENGTH:
            raise ValueError("History content too long")
        return v


class AskRequest(BaseModel):
    query: str
    history: Optional[List[HistoryTurn]] = None

    @validator("query")
    def validate_query(cls, v):
        v = (v or "").strip()
        if not v:
            raise ValueError("Empty query")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError("Query too long")
        return v


# ─────────────────────────────────────────────────────────────
# UTIL
# ─────────────────────────────────────────────────────────────
def truncate_history(history: List[HistoryTurn]):
    if not history:
        return []

    trimmed = history[-MAX_HISTORY_TURNS:]
    return [{"role": h.role, "content": h.content.strip()} for h in trimmed]


async def run_with_timeout(q, history):
    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(executor, rag.generate_answer, q, history),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


# ─────────────────────────────────────────────────────────────
# MAIN ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.post("/ask")
async def ask(req: AskRequest, request: Request):
    start_time = time.time()

    try:
        q = req.query.strip()
        history = truncate_history(req.history or [])

        log.info(f"Incoming query: {q[:120]}")

        result = await run_with_timeout(q, history)

        latency = round((time.time() - start_time) * 1000, 2)

        return {
            "answer": result.get("answer", ""),
            "references": result.get("refs", []),
            "source": result.get("source", "unknown"),
            "latency_ms": latency,
        }

    except asyncio.TimeoutError:
        log.error("Request timed out")
        return {
            "answer": "Request timed out.",
            "references": [],
            "source": "timeout",
        }

    except Exception as e:
        log.exception("Server error")
        return {
            "answer": "Server error.",
            "references": [],
            "source": "server_error",
        }


# ─────────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "ORA backend"}
