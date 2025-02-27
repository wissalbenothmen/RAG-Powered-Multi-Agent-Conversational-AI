"""
Overview:
This module evaluates a RAG-powered AI system's API using Q&A pairs from a CSV.
It measures latency, semantic similarity, faithfulness, and retrieval metrics.

Technical Details:
- Input: CSV file with question-response pairs, sampled randomly.
- Evaluation: Queries API, computes metrics, and saves results to JSON.
- Metrics: Uses SentenceTransformer for embeddings, cosine similarity for scoring.
- Configuration: API endpoint and CSV file are predefined constants.
"""
import json
import csv
import time
import random
from datetime import datetime
import requests
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Metrics storage lists
latencies = []
semantic_similarities = []
faithfulness_scores = []
precision_scores = []
recall_scores = []
ndcg_scores = []

# Configuration constants
API_URL = "http://127.0.0.1:5000/chat"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json"}
CSV_FILE = "eval_questions_answers.csv"
NUM_EXAMPLES = 10
RESULTS_FILE = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_qa_pairs(csv_file: str) -> list:
    """Load and sample valid Q&A pairs from a CSV file.

    Args:
        csv_file (str): Path to CSV with 'Question' and 'Response' columns.

    Returns:
        list: List of (question, answer) tuples.
    """
    qa_pairs = []
    with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, fieldnames=["Question", "Response"])
        next(reader)  # Skip header row
        for row in reader:
            question = row["Question"].strip() if row["Question"] else ""
            response = row["Response"].strip() if row["Response"] else ""
            if question and response:
                qa_pairs.append((question, response))
    if len(qa_pairs) < NUM_EXAMPLES:
        raise ValueError(
            "CSV must have at least %d valid Q&A pairs, found %d",
            NUM_EXAMPLES,
            len(qa_pairs)
        )
    return random.sample(qa_pairs, NUM_EXAMPLES)


def query_api(question: str) -> tuple:
    """Send a question to the API and measure latency.

    Args:
        question (str): Query to send to the API.

    Returns:
        tuple: (API response dict or None, latency in seconds).
    """
    payload = {"user_input": question}
    start_time = time.time()
    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        latency = time.time() - start_time
        if response.status_code == 200:
            print("Raw API response: %s", response.text)
            return response.json(), latency
        print("API error: Status %d - %s", response.status_code, response.text)
        return None, latency
    except requests.RequestException as e:
        print("Request failed: %s", str(e))
        return None, time.time() - start_time


def cosine_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts.

    Args:
        text1 (str): First text string.
        text2 (str): Second text string.

    Returns:
        float: Cosine similarity score between 0 and 1.
    """
    emb1 = embedding_model.encode(text1, convert_to_tensor=True)
    emb2 = embedding_model.encode(text2, convert_to_tensor=True)
    # Suppress E1102 false positive; cosine_similarity is callable
    cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()  # pylint: disable=no-member
    return max(0.0, min(1.0, cos_sim))


def compute_faithfulness(answer: str, sources: list) -> float:
    """Compute faithfulness score based on answer and sources.

    Args:
        answer (str): Generated answer text.
        sources (list): List of source dictionaries or pseudo-sources.

    Returns:
        float: Faithfulness score between 0 and 1.
    """
    if not sources or not isinstance(sources, list) or not answer:
        return 0.0
    import re
    # Split answer by source citations
    source_segments = re.split(r"\[Source: [^\]]+\]", answer)
    source_segments = [seg.strip() for seg in source_segments if seg.strip()]
    if not source_segments:
        return cosine_similarity(answer, "")
    # Average similarity across segments
    faithfulness = np.mean(
        [cosine_similarity(answer, seg) for seg in source_segments]
    )
    return faithfulness


def compute_retrieval_metrics(sources: list, ground_truth: str) -> tuple:
    """Compute precision, recall, and NDCG for retrieval evaluation.

    Args:
        sources (list): List of retrieved source dictionaries.
        ground_truth (str): Expected answer text.

    Returns:
        tuple: (precision, recall, ndcg) scores between 0 and 1.
    """
    if not sources or not isinstance(sources, list) or not ground_truth:
        return 0.0, 0.0, 0.0
    answer = sources[0].get("answer", "") if sources and "answer" in sources[0] else ""
    if answer and "[Source:" in answer:
        import re
        source_segments = re.split(r"\[Source: [^\]]+\]", answer)
        source_segments = [seg.strip() for seg in source_segments if seg.strip()]
    else:
        source_segments = [answer] if answer else [""]
    num_sources = len(sources)
    relevance = [
        1 if cosine_similarity(seg, ground_truth) > 0.7 else 0
        for seg in source_segments[:num_sources]
    ]
    precision = np.mean(relevance) if relevance else 0.0
    recall = min(
        sum(relevance) / max(1, min(5, len(ground_truth.split()))),
        1.0
    )
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))
    idcg = sum(1 / np.log2(idx + 2) for idx in range(min(len(relevance), 5)))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return precision, recall, ndcg


def save_results(qa_pairs: list, api_responses: list) -> None:
    """Save evaluation results to a JSON file.

    Args:
        qa_pairs (list): List of (question, ground_truth) tuples.
        api_responses (list): List of API response dictionaries or None.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "examples": [
            {
                "question": question,
                "ground_truth": ground_truth,
                "answer": resp.get("answer", "") if resp else "",
                "sources_count": len(resp.get("sources", [])) if resp else 0,
                "semantic_similarity": semantic_similarities[i],
                "faithfulness": faithfulness_scores[i],
                "precision": precision_scores[i],
                "recall": recall_scores[i],
                "ndcg": ndcg_scores[i],
                "latency": latencies[i],
            }
            for i, ((question, ground_truth), resp) in enumerate(zip(qa_pairs, api_responses))
        ],
        "summary": {
            "avg_semantic_similarity": float(np.mean(semantic_similarities)),
            "avg_faithfulness": float(np.mean(faithfulness_scores)),
            "avg_precision": float(np.mean(precision_scores)),
            "avg_recall": float(np.mean(recall_scores)),
            "avg_ndcg": float(np.mean(ndcg_scores)),
            "avg_latency": float(np.mean(latencies)),
            "total_examples": len(qa_pairs),
        },
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("Results saved to %s", RESULTS_FILE)


def run_evaluation() -> None:
    """Run the evaluation process on the API with sampled Q&A pairs."""
    print("Starting evaluation...\nAssuming API at %s", API_URL)
    print("Start API with 'python app.py' if not running.\n")

    try:
        response = requests.get("http://127.0.0.1:5000/", timeout=5)
        if response.status_code != 200:
            print("Warning: API root not verified. Proceeding...")
    except requests.RequestException:
        print("Warning: API not reachable. Ensure it’s running.")
        return

    try:
        qa_pairs = load_qa_pairs(CSV_FILE)
        print("Loaded %d Q&A pairs from %s\n", len(qa_pairs), CSV_FILE)
    except Exception as e:
        print("Failed to load CSV: %s", str(e))
        return

    api_responses = []
    for i, (question, ground_truth) in enumerate(qa_pairs, 1):
        print("--- Evaluating Example %d/%d ---", i, NUM_EXAMPLES)
        print("Question: %s", question)

        response, latency = query_api(question)
        api_responses.append(response)

        if response and response.get("status") == "success":
            answer = response.get("answer", "")
            sources = response.get("sources", [])
            print("Answer: %s", answer)
            print("Ground Truth: %s", ground_truth)
            print("Sources Retrieved: %d", len(sources))

            sim = cosine_similarity(answer, ground_truth)
            faith = compute_faithfulness(answer, [{"answer": answer}] + sources)
            prec, rec, ndcg = compute_retrieval_metrics(
                [{"answer": answer}] + sources,
                ground_truth
            )

            latencies.append(latency)
            semantic_similarities.append(sim)
            faithfulness_scores.append(faith)
            precision_scores.append(prec)
            recall_scores.append(rec)
            ndcg_scores.append(ndcg)

            print("Semantic Similarity: %.2f", sim)
            print("Faithfulness: %.2f", faith)
            print("Precision: %.2f", prec)
            print("Recall: %.2f", rec)
            print("NDCG: %.2f", ndcg)
            print("Latency: %.2f seconds", latency)
        else:
            print("No valid response; assigning zero metrics")
            latencies.append(latency)
            semantic_similarities.append(0.0)
            faithfulness_scores.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            ndcg_scores.append(0.0)
            print("Latency: %.2f seconds", latency)

        print("-" * 50)
        if i < NUM_EXAMPLES:
            print("Waiting 10 seconds before next request...")
            time.sleep(10)

    print("\n=== Summary of Evaluation ===")
    print("Avg Semantic Similarity: %.2f", np.mean(semantic_similarities))
    print("Avg Faithfulness: %.2f", np.mean(faithfulness_scores))
    print("Avg Precision: %.2f", np.mean(precision_scores))
    print("Avg Recall: %.2f", np.mean(recall_scores))
    print("Avg NDCG: %.2f", np.mean(ndcg_scores))
    print("Avg Latency: %.2f seconds", np.mean(latencies))
    print("Total Examples: %d", len(qa_pairs))
    save_results(qa_pairs, api_responses)


if __name__ == "__main__":
    run_evaluation()

#end-of-file
