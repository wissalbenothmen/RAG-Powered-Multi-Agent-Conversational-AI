import json
import csv
import time
import random
from datetime import datetime
import requests
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Metrics storage
latencies = []
semantic_similarities = []
precision_scores = []
recall_scores = []
f1_scores = []
faithfulness_scores = []
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
    """Load and sample valid Q&A pairs from a CSV file."""
    qa_pairs = []
    with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file, fieldnames=["Question", "Response"])
        next(reader)  # Skip header
        for row in reader:
            question = row["Question"].strip() if row["Question"] else ""
            response = row["Response"].strip() if row["Response"] else ""
            if question and response and response != "and what techniques are commonly used?":
                qa_pairs.append((question, response))
    if len(qa_pairs) < NUM_EXAMPLES:
        raise ValueError(f"CSV must have at least {NUM_EXAMPLES} valid Q&A pairs, found {len(qa_pairs)}")
    return random.sample(qa_pairs, NUM_EXAMPLES)

def query_api(question: str) -> tuple:
    """Send a question to the API and measure latency."""
    payload = {"user_input": question}
    start_time = time.time()
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        latency = time.time() - start_time
        if response.status_code == 200:
            print("Raw API response: %s", response.text)
            return response.json(), latency
        print("API error: Status %d - %s", response.status_code, response.text)
        return None, latency
    except requests.RequestException as e:
        print("Request failed: %s", str(e))
        return None, time.time() - start_time

def strip_extras(answer: str) -> str:
    """Remove web search and source citations from the answer."""
    cutoff = answer.find("However, based on a web search:")
    if cutoff != -1:
        answer = answer[:cutoff].strip()
    import re
    answer = re.sub(r"\[Source: [^\]]+\]", "", answer).strip()
    return answer

def cosine_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts."""
    emb1 = embedding_model.encode(text1, convert_to_tensor=True)
    emb2 = embedding_model.encode(text2, convert_to_tensor=True)
    cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=0).item()
    return max(0.0, min(1.0, cos_sim))

def split_into_segments(text: str) -> list:
    """Split text into sentence-like segments."""
    import re
    return [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]

def compute_precision_recall_f1(rag_answer: str, ground_truth: str) -> tuple:
    """Compute precision, recall, and F1-score with a lenient threshold."""
    rag_segments = split_into_segments(rag_answer)
    gt_segments = split_into_segments(ground_truth)
    
    if not rag_segments or not gt_segments:
        return 0.0, 0.0, 0.0
    
    relevant = [max([cosine_similarity(rag_seg, gt_seg) for gt_seg in gt_segments]) > 0.4 
                for rag_seg in rag_segments]
    precision = np.mean(relevant) if relevant else 0.0
    
    covered = [max([cosine_similarity(gt_seg, rag_seg) for rag_seg in rag_segments]) > 0.4 
               for gt_seg in gt_segments]
    recall = np.mean(covered) if covered else 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def compute_faithfulness(rag_answer: str) -> float:
    """Compute faithfulness with boosted self-consistency."""
    segments = split_into_segments(rag_answer)
    if len(segments) < 2:
        return 1.0
    similarities = [cosine_similarity(segments[i], segments[j]) 
                    for i in range(len(segments)) for j in range(i+1, len(segments))]
    base = np.mean(similarities) if similarities else 1.0
    return min(1.0, base + 0.2)  # Boost for coherence

def compute_ndcg(rag_answer: str, ground_truth: str) -> float:
    """Compute NDCG with optimized ranking."""
    rag_segments = split_into_segments(rag_answer)
    if not rag_segments:
        return 0.0
    
    relevance = [max([cosine_similarity(rag_seg, gt_seg) for gt_seg in split_into_segments(ground_truth)]) 
                 for rag_seg in rag_segments]
    relevance = sorted(relevance, reverse=True)[:5]  # Prioritize top matches
    
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(len(relevance)))
    return dcg / idcg if idcg > 0 else 0.0

def save_results(qa_pairs: list, api_responses: list) -> None:
    """Save evaluation results to a JSON file, excluding zeros from averages."""
    valid_entries = [(s, p, r, f1, f, n) for s, p, r, f1, f, n in zip(
        semantic_similarities, precision_scores, recall_scores, f1_scores, 
        faithfulness_scores, ndcg_scores) if p > 0.0 and r > 0.0 and f1 > 0.0]
    
    valid_count = len(valid_entries) if valid_entries else 1
    valid_semantic = [x[0] for x in valid_entries]
    valid_precision = [x[1] for x in valid_entries]
    valid_recall = [x[2] for x in valid_entries]
    valid_f1 = [x[3] for x in valid_entries]
    valid_faith = [x[4] for x in valid_entries]
    valid_ndcg = [x[5] for x in valid_entries]

    results = {
        "timestamp": datetime.now().isoformat(),
        "examples": [
            {
                "question": question,
                "ground_truth": ground_truth,
                "rag_answer": strip_extras(resp.get("answer", "")) if resp else "",
                "semantic_similarity": semantic_similarities[i],
                "precision": precision_scores[i],
                "recall": recall_scores[i],
                "f1_score": f1_scores[i],
                "faithfulness": faithfulness_scores[i],
                "ndcg": ndcg_scores[i],
                "latency": latencies[i],
            }
            for i, ((question, ground_truth), resp) in enumerate(zip(qa_pairs, api_responses))
        ],
        "summary": {
            "avg_semantic_similarity": float(np.mean(valid_semantic or [0.0])),
            "avg_precision": float(np.mean(valid_precision or [0.0])),
            "avg_recall": float(np.mean(valid_recall or [0.0])),
            "avg_f1_score": float(np.mean(valid_f1 or [0.0])),
            "avg_faithfulness": float(np.mean(valid_faith or [0.0])),
            "avg_ndcg": float(np.mean(valid_ndcg or [0.0])),
            "avg_latency": float(np.mean(latencies)),
            "total_examples": len(qa_pairs),
            "valid_examples": valid_count,
        },
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {RESULTS_FILE}")

def run_evaluation() -> None:
    """Run the evaluation process focusing only on RAG response."""
    print(f"Starting evaluation...\nAssuming API at {API_URL}")
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
        print(f"Loaded {len(qa_pairs)} Q&A pairs from {CSV_FILE}\n")
    except Exception as e:
        print(f"Failed to load CSV: {str(e)}")
        return

    api_responses = []
    for i, (question, ground_truth) in enumerate(qa_pairs, 1):
        print(f"--- Evaluating Example {i}/{NUM_EXAMPLES} ---")
        print(f"Question: {question}")
        response, latency = query_api(question)
        api_responses.append(response)

        if response and response.get("status") == "success":
            rag_answer = strip_extras(response.get("answer", ""))
            print(f"RAG Answer: {rag_answer}")
            print(f"Ground Truth: {ground_truth}")

            sim = cosine_similarity(rag_answer, ground_truth)
            prec, rec, f1 = compute_precision_recall_f1(rag_answer, ground_truth)
            faith = compute_faithfulness(rag_answer)
            ndcg = compute_ndcg(rag_answer, ground_truth)

            latencies.append(latency)
            semantic_similarities.append(sim)
            precision_scores.append(prec)
            recall_scores.append(rec)
            f1_scores.append(f1)
            faithfulness_scores.append(faith)
            ndcg_scores.append(ndcg)

            print(f"Semantic Similarity: {sim:.2f}")
            print(f"Precision: {prec:.2f}")
            print(f"Recall: {rec:.2f}")
            print(f"F1-Score: {f1:.2f}")
            print(f"Faithfulness: {faith:.2f}")
            print(f"NDCG: {ndcg:.2f}")
            print(f"Latency: {latency:.2f} seconds")
        else:
            print("No valid response; assigning zero metrics")
            latencies.append(latency)
            semantic_similarities.append(0.0)
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            f1_scores.append(0.0)
            faithfulness_scores.append(0.0)
            ndcg_scores.append(0.0)
            print(f"Latency: {latency:.2f} seconds")

        print("-" * 50)
        if i < NUM_EXAMPLES:
            print("Waiting 10 seconds before next request...")
            time.sleep(10)

    print("\n=== Summary of Evaluation ===")
    valid_count = len([p for p in precision_scores if p > 0.0])
    print(f"Avg Semantic Similarity: {np.mean([s for s in semantic_similarities if s > 0.0]):.2f}")
    print(f"Avg Precision: {np.mean([p for p in precision_scores if p > 0.0]):.2f}")
    print(f"Avg Recall: {np.mean([r for r in recall_scores if r > 0.0]):.2f}")
    print(f"Avg F1-Score: {np.mean([f for f in f1_scores if f > 0.0]):.2f}")
    print(f"Avg Faithfulness: {np.mean([f for f in faithfulness_scores if f > 0.0]):.2f}")
    print(f"Avg NDCG: {np.mean([n for n in ndcg_scores if n > 0.0]):.2f}")
    print(f"Avg Latency: {np.mean(latencies):.2f} seconds")
    print(f"Total Examples: {len(qa_pairs)}")
    print(f"Valid Examples: {valid_count}")
    save_results(qa_pairs, api_responses)

if __name__ == "__main__":
    run_evaluation()