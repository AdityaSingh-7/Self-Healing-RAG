"""
routers/evaluate.py — RAGAS Evaluation Endpoint

WHAT THIS DOES:
Runs evaluation metrics on your RAG pipeline to measure quality.
Upload a test dataset (questions + expected answers) and get scores.

METRICS:
- Faithfulness: Does the answer stick to the retrieved context? (no hallucination)
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are retrieved chunks actually relevant?
- Context Recall: Did we find ALL relevant chunks?

USAGE:
POST /evaluate/run with a list of test questions + ground truth answers.
The system runs each question through the pipeline and scores the results.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.embedder import EmbeddingService
from app.services.vectorstore import VectorStoreService
from app.services.llm import LLMService


router = APIRouter(tags=["Evaluation"])


class EvalQuestion(BaseModel):
    """A single evaluation test case."""

    question: str = Field(description="The test question")
    ground_truth: str = Field(description="The expected correct answer")
    # Optional: provide expected source docs for context recall
    expected_contexts: list[str] = Field(
        default=[],
        description="Expected relevant text passages (for context recall)",
    )


class EvalRequest(BaseModel):
    """Request to run evaluation."""

    test_cases: list[EvalQuestion] = Field(
        min_length=1,
        max_length=100,
        description="List of test cases to evaluate",
    )
    top_k: int = Field(default=5, ge=1, le=20)
    recency_weight: float = Field(default=0.2, ge=0.0, le=1.0)


class EvalResult(BaseModel):
    """Results for a single test case."""

    question: str
    generated_answer: str
    ground_truth: str
    retrieved_contexts: list[str]
    scores: dict[str, float]


class EvalResponse(BaseModel):
    """Full evaluation results."""

    overall_scores: dict[str, float] = Field(
        description="Average scores across all test cases"
    )
    per_question: list[EvalResult] = Field(
        description="Detailed results per test case"
    )
    num_questions: int
    model: str


@router.post("/run", response_model=EvalResponse)
async def run_evaluation(request: EvalRequest):
    """
    Run RAG evaluation on a set of test questions.

    For each question:
    1. Retrieve context chunks
    2. Generate answer via LLM
    3. Score faithfulness, relevancy, precision, recall

    Returns per-question and aggregate scores.
    """
    user_id = "default_user"
    embedder = EmbeddingService()
    store = VectorStoreService()
    llm = LLMService()

    results = []
    total_scores = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
    }

    for test_case in request.test_cases:
        # Step 1: Retrieve
        query_embedding = embedder.embed_text(test_case.question)
        search_results = store.search(
            query_embedding=query_embedding,
            user_id=user_id,
            top_k=request.top_k,
            recency_weight=request.recency_weight,
        )

        retrieved_texts = [r["text"] for r in search_results]

        # Step 2: Generate answer
        answer = await llm.generate(
            test_case.question, search_results, history=None
        )

        # Step 3: Score (simplified RAGAS-style metrics)
        scores = _calculate_scores(
            question=test_case.question,
            answer=answer,
            ground_truth=test_case.ground_truth,
            retrieved_contexts=retrieved_texts,
            expected_contexts=test_case.expected_contexts,
        )

        results.append(
            EvalResult(
                question=test_case.question,
                generated_answer=answer,
                ground_truth=test_case.ground_truth,
                retrieved_contexts=retrieved_texts[:3],  # Trim for response size
                scores=scores,
            )
        )

        # Accumulate for averages
        for key in total_scores:
            total_scores[key] += scores.get(key, 0.0)

    # Calculate averages
    n = len(request.test_cases)
    overall_scores = {k: round(v / n, 4) for k, v in total_scores.items()}

    return EvalResponse(
        overall_scores=overall_scores,
        per_question=results,
        num_questions=n,
        model=llm.model,
    )


def _calculate_scores(
    question: str,
    answer: str,
    ground_truth: str,
    retrieved_contexts: list[str],
    expected_contexts: list[str],
) -> dict[str, float]:
    """
    Calculate simplified RAGAS-style metrics.

    In production, you'd use the actual RAGAS library which uses LLM-as-judge.
    This is a simplified version using text overlap for demonstration.

    For full RAGAS integration, install `ragas` and `datasets` packages
    and use ragas.evaluate() with the proper dataset format.
    """
    # Faithfulness: How much of the answer is supported by context?
    # Simplified: word overlap between answer and retrieved contexts
    context_combined = " ".join(retrieved_contexts).lower()
    answer_words = set(answer.lower().split())
    context_words = set(context_combined.split())
    if answer_words:
        faithfulness = len(answer_words & context_words) / len(answer_words)
    else:
        faithfulness = 0.0

    # Answer Relevancy: Does the answer address the question?
    # Simplified: word overlap between answer and question
    question_words = set(question.lower().split()) - {"what", "how", "is", "the", "a", "an", "do", "does", "can", "are"}
    if question_words:
        relevancy = len(answer_words & question_words) / len(question_words)
        relevancy = min(relevancy, 1.0)
    else:
        relevancy = 0.0

    # Context Precision: Are retrieved chunks relevant to the question?
    # Simplified: overlap between question keywords and contexts
    if context_words and question_words:
        precision = len(context_words & question_words) / max(len(question_words), 1)
        precision = min(precision, 1.0)
    else:
        precision = 0.0

    # Context Recall: Did we retrieve chunks covering the ground truth?
    # Simplified: overlap between ground truth and retrieved contexts
    ground_words = set(ground_truth.lower().split())
    if ground_words:
        recall = len(ground_words & context_words) / len(ground_words)
    else:
        recall = 0.0

    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevancy": round(relevancy, 4),
        "context_precision": round(precision, 4),
        "context_recall": round(recall, 4),
    }


@router.get("/sample-dataset")
async def get_sample_dataset():
    """
    Returns a sample evaluation dataset template.
    Use this as a starting point for creating your own test cases.
    """
    return {
        "description": "Sample evaluation dataset. Replace with your own questions based on uploaded documents.",
        "test_cases": [
            {
                "question": "What is the company's parental leave policy?",
                "ground_truth": "16 weeks at full pay for all employees.",
                "expected_contexts": [],
            },
            {
                "question": "How many PTO days do employees receive?",
                "ground_truth": "20 days per year, with up to 5 days carrying over.",
                "expected_contexts": [],
            },
            {
                "question": "What is the process for requesting remote work?",
                "ground_truth": "Submit a request through the HR portal, approved by direct manager within 5 business days.",
                "expected_contexts": [],
            },
        ],
    }
