"""Evaluation harness for Veritas RAG.

Supports two tiers:
1. Offline Deterministic Metrics (Default, zero API keys/costs):
   - Context Precision@k: Proportion of retrieved passages from ground-truth relevant files.
   - Context Recall: Proportion of required ground-truth sources present in retrieved passages.
   - Cross-Fiscal-Year Synthesis verification: Validates simultaneous retrieval of multi-year filings.
   - Grounded Refusal Accuracy: Validates that unanswerable queries return "I don't know".
2. RAGAS LLM-Judged Metrics (Opt-in via `--ragas` flag):
   - Faithfulness, Answer Relevancy, Context Precision, and Context Recall using an LLM judge.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

from src.config import settings
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger("veritas.evaluate")


def load_eval_dataset(eval_path: Path = Path("eval/eval_questions.json")) -> List[Dict[str, Any]]:
    """Load evaluation benchmark dataset."""
    full_path = settings.BASE_DIR / eval_path
    if not full_path.exists():
        raise FileNotFoundError(f"Evaluation benchmark file not found at: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_offline_evaluation(pipeline: RAGPipeline, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute deterministic offline retrieval and generation evaluation."""
    results = []
    total_precision = 0.0
    total_recall = 0.0
    cross_year_success = 0
    cross_year_total = 0
    refusal_success = 0
    refusal_total = 0
    evaluated_retrieval_count = 0

    print("=" * 75)
    print("VERITAS OFFLINE RETRIEVAL & GENERATION EVALUATION")
    print("=" * 75)
    print(f"{'ID':<26} | {'Type':<18} | {'Precision@5':<11} | {'Recall':<6} | Sources Retrieved")
    print("-" * 75)

    for item in dataset:
        qid = item["id"]
        qtype = item.get("type", "general")
        question = item["question"]
        relevant_sources = set(item.get("relevant_sources", []))

        pipe_res = pipeline.answer_query(question, top_k=settings.RETRIEVAL_FINAL_TOP_K)
        retrieved_sources = [c.source for c in pipe_res.retrieved_chunks]
        retrieved_sources_set = set(retrieved_sources)

        if qtype == "unanswerable" or not relevant_sources:
            refusal_total += 1
            is_refusal = "i don't know" in pipe_res.answer.lower()
            if is_refusal:
                refusal_success += 1
            precision = 1.0 if is_refusal else 0.0
            recall = 1.0 if is_refusal else 0.0
            retrieval_summary = f"Refusal verified: {is_refusal}"
        else:
            evaluated_retrieval_count += 1
            # Context Precision@k: fraction of retrieved chunks belonging to relevant_sources
            hits = sum(1 for s in retrieved_sources if s in relevant_sources)
            precision = hits / max(len(retrieved_sources), 1)

            # Context Recall: fraction of ground truth sources represented
            recall_hits = len(relevant_sources & retrieved_sources_set)
            recall = recall_hits / max(len(relevant_sources), 1)

            total_precision += precision
            total_recall += recall

            retrieval_summary = ", ".join(sorted(retrieved_sources_set))

            if qtype == "cross_fiscal_year":
                cross_year_total += 1
                if relevant_sources.issubset(retrieved_sources_set):
                    cross_year_success += 1

        print(f"{qid:<26} | {qtype:<18} | {precision:.4f}      | {recall:.4f} | {retrieval_summary}")

        results.append({
            "id": qid,
            "question": question,
            "type": qtype,
            "precision_at_k": round(precision, 4),
            "recall": round(recall, 4),
            "answer": pipe_res.answer,
            "retrieved_sources": retrieved_sources,
            "citations": [{"source": c.source, "page": c.page} for c in pipe_res.citations],
        })

    avg_precision = total_precision / max(evaluated_retrieval_count, 1)
    avg_recall = total_recall / max(evaluated_retrieval_count, 1)
    refusal_acc = (refusal_success / refusal_total) if refusal_total > 0 else 1.0
    cross_year_rate = (cross_year_success / cross_year_total) if cross_year_total > 0 else 1.0

    print("-" * 75)
    print("SUMMARY METRICS:")
    print(f"  • Mean Context Precision@5:  {avg_precision:.4f}")
    print(f"  • Mean Context Recall:        {avg_recall:.4f}")
    print(f"  • Cross-Year Retrieval Rate:  {cross_year_rate:.1%} ({cross_year_success}/{cross_year_total})")
    print(f"  • Unanswerable Refusal Rate:  {refusal_acc:.1%} ({refusal_success}/{refusal_total})")
    print("=" * 75)

    summary = {
        "mean_context_precision_at_5": round(avg_precision, 4),
        "mean_context_recall": round(avg_recall, 4),
        "cross_fiscal_year_retrieval_rate": round(cross_year_rate, 4),
        "unanswerable_refusal_rate": round(refusal_acc, 4),
        "total_questions": len(dataset),
        "individual_results": results,
    }

    # Save to data/processed/eval_results.json
    out_path = settings.get_processed_dir() / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Detailed evaluation results saved to: {out_path}\n")

    return summary


def run_ragas_evaluation(pipeline: RAGPipeline, dataset: List[Dict[str, Any]]) -> None:
    """Execute LLM-judged evaluation using RAGAS."""
    if not settings.OPENAI_API_KEY:
        print("\n[RAGAS] Skipping RAGAS evaluation: OPENAI_API_KEY is not set.")
        print("[RAGAS] Set OPENAI_API_KEY in your .env file to enable RAGAS LLM-judged evaluation.")
        return

    print("\n[RAGAS] Running LLM-judged evaluation with RAGAS metrics...")
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        ragas_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        for item in dataset:
            if item.get("type") == "unanswerable":
                continue
            pipe_res = pipeline.answer_query(item["question"])
            ragas_data["question"].append(item["question"])
            ragas_data["answer"].append(pipe_res.answer)
            ragas_data["contexts"].append([c.text for c in pipe_res.retrieved_chunks])
            ragas_data["ground_truth"].append(item["ground_truth"])

        eval_ds = Dataset.from_dict(ragas_data)
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        score = evaluate(eval_ds, metrics=metrics)
        print("\n" + "=" * 60)
        print("RAGAS EVALUATION RESULTS:")
        print("=" * 60)
        print(score)
        print("=" * 60)
    except Exception as e:
        print(f"[RAGAS] Error executing RAGAS evaluation: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Veritas RAG Pipeline")
    parser.add_argument(
        "--ragas", action="store_true", help="Run opt-in RAGAS LLM-judged evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eval/eval_questions.json",
        help="Path to evaluation questions JSON",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline()
    dataset = load_eval_dataset(Path(args.dataset))

    summary = run_offline_evaluation(pipeline, dataset)

    if args.ragas:
        run_ragas_evaluation(pipeline, dataset)


if __name__ == "__main__":
    main()
