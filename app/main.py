from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

load_dotenv()

from app.llm_providers import get_llm_provider
from app.rag import ask as rag_ask
from app.vector_store import load_vector_store

logger = logging.getLogger(__name__)

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["store"] = load_vector_store()
    try:
        app_state["provider"] = get_llm_provider()
    except RuntimeError as exc:
        logger.warning("LLM provider not configured: %s", exc)
        app_state["provider"] = None
    yield
    app_state.clear()


app = FastAPI(
    title="PDF RAG API",
    description="Answers questions about Document.pdf using retrieval-augmented generation.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("question must not be empty or whitespace-only")
        return value.strip()


class Source(BaseModel):
    page_number: int | None
    source: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    if app_state.get("provider") is None:
        raise HTTPException(
            status_code=503,
            detail="LLM provider is not configured on the server. Set LLM_API_KEY.",
        )

    try:
        result = rag_ask(
            request.question,
            store=app_state["store"],
            provider=app_state["provider"],
        )
    except Exception:
        logger.exception("RAG pipeline failed for question: %r", request.question)
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")

    return AskResponse(**result)
