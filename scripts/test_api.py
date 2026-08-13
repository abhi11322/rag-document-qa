"""API-level tests for the FastAPI RAG endpoints, using FastAPI's TestClient.

Requires the Chroma index to be built and LLM_API_KEY to be set (see
.env.example).

Run from the project root:
    python scripts/test_api.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app

VALID_QUESTIONS = [
    "What are the three stages of RAG?",
    "What is query expansion in RAG?",
]
OUT_OF_DOMAIN_QUESTION = "What is the capital of France?"
INVALID_QUESTIONS = ["", "   "]


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        raise AssertionError(message)


def main() -> None:
    with TestClient(app) as client:
        print("=== GET /health ===")
        resp = client.get("/health")
        check(resp.status_code == 200, "GET /health returns 200")
        check(resp.json() == {"status": "ok"}, "GET /health returns {'status': 'ok'}")
        print()

        print("=== POST /ask — valid in-document questions ===")
        for question in VALID_QUESTIONS:
            resp = client.post("/ask", json={"question": question})
            print(f"Q: {question}")
            check(resp.status_code == 200, f"HTTP 200 for {question!r}")
            body = resp.json()
            check(bool(body.get("answer")), "answer is present and non-empty")
            check(len(body.get("sources", [])) > 0, "sources list is non-empty")
            check(
                all("page_number" in s for s in body["sources"]),
                "each source has page_number",
            )
            print(f"  answer: {body['answer'][:150]}...")
            print(f"  sources: {body['sources']}")
            print()

        print("=== POST /ask — out-of-document question ===")
        resp = client.post("/ask", json={"question": OUT_OF_DOMAIN_QUESTION})
        check(resp.status_code == 200, "HTTP 200 for out-of-document question")
        body = resp.json()
        print(f"Q: {OUT_OF_DOMAIN_QUESTION}")
        print(f"  answer: {body['answer']}")
        check(
            "could not be found" in body["answer"].lower(),
            "grounded refusal preserved for out-of-document question",
        )
        print()

        print("=== POST /ask — invalid input ===")
        for question in INVALID_QUESTIONS:
            resp = client.post("/ask", json={"question": question})
            check(
                400 <= resp.status_code < 500,
                f"HTTP 4xx for invalid question {question!r} (got {resp.status_code})",
            )
        print()

    print("All API tests passed.")


if __name__ == "__main__":
    main()
