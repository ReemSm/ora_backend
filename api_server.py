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

    match, score, ar, idx = rag.dataset_match(q)

    if match and score >= rag.SIM_THRESHOLD:
        answer = match["ar_a"] if ar else match["en_a"]
        rag_query = match["ar_q"] if ar else match["en_q"]
        fields = match["field"]
    else:
        answer = rag.gpt_style_answer(q)
        rag_query = q
        fields = None

    refs = rag.rag_refs(rag_query, fields)

    return {
        "answer": answer,
        "references": refs
    }
