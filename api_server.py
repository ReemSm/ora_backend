import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Literal

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import step3_dataset_gpt_with_contract_and_strict_rag as rag


MAX_QUERY_LENGTH = 500
MAX_HISTORY_TURNS = 6
MAX_HISTORY_CONTENT_LENGTH = 500
REQUEST_TIMEOUT_SECONDS = 15

ALLOWED_ORIGINS = ["*"]

executor = ThreadPoolExecutor(max_workers=10)

logging.basicConfig(level=logging.INFO, format="[API %(levelname)s] %(message)s")
log = logging.getLogger("api")


app = FastAPI(title="ORA Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("History content cannot be empty")
        if len(v) > MAX_HISTORY_CONTENT_LENGTH:
            raise ValueError("History content too long")
        return v


class AskRequest(BaseModel):
    query: str
    history: Optional[List[HistoryTurn]] = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("Empty query")
        if len(v) > MAX_QUERY_LENGTH:
            raise ValueError("Query too long")
        return v


class AskResponse(BaseModel):
    answer: str
    references: List[str]
    source: str
    request_id: str
    latency_ms: float


def normalize_history(history: List[HistoryTurn] | None) -> List[dict]:
    if not history:
        return []
    trimmed = history[-MAX_HISTORY_TURNS:]
    return [{"role": item.role, "content": item.content} for item in trimmed]


async def run_generate_answer(query: str, history: List[dict]) -> dict:
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(executor, rag.generate_answer, query, history),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()

    try:
        query = req.query
        history = normalize_history(req.history)

        log.info(f"[{request_id}] /ask query={query[:120]!r} history_turns={len(history)}")

        result = await run_generate_answer(query, history)

        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        log.info(
            f"[{request_id}] completed source={result.get('source', 'unknown')} latency_ms={latency_ms}"
        )

        return AskResponse(
            answer=result.get("answer", ""),
            references=result.get("refs", []),
            source=result.get("source", "unknown"),
            request_id=request_id,
            latency_ms=latency_ms,
        )

    except asyncio.TimeoutError:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        log.error(f"[{request_id}] timeout latency_ms={latency_ms}")
        return AskResponse(
            answer="Request timed out.",
            references=[],
            source="timeout",
            request_id=request_id,
            latency_ms=latency_ms,
        )

    except Exception:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        log.exception(f"[{request_id}] server_error latency_ms={latency_ms}")
        return AskResponse(
            answer="Server error.",
            references=[],
            source="server_error",
            request_id=request_id,
            latency_ms=latency_ms,
        )


@app.get("/")
def root():
    return {"status": "ok", "service": "ORA backend"}
