import sys
import os
import json
import time
import re
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.environ["USE_LIVE_LLM"] = "1"
os.environ["ALLOW_HYBRID_FALLBACK"] = "1"
os.environ["ENFORCE_QUALITY_GATE"] = "0"

from app.langchain_pipeline import pipeline
from langchain_core.messages import HumanMessage, SystemMessage

# Labeled evaluation dataset (Ground truth)
EVAL_DATASET = [
    {
        "query": "How long does a national police check take in Australia?",
        "expected_topic": "processing_time",
        "expected_doc_ids": ["doc_01_processing_time", "doc_time", "processing"],
        "is_out_of_domain": False
    },
    {
        "query": "What identity documents do I need to supply for a background check?",
        "expected_topic": "requirements",
        "expected_doc_ids": ["doc_07_required_documents", "doc_reqs", "required"],
        "is_out_of_domain": False
    },
    {
        "query": "What are the rules for spent convictions under Commonwealth law?",
        "expected_topic": "spent_convictions",
        "expected_doc_ids": ["doc_spent_convictions_cth", "doc_05_spent", "spent"],
        "is_out_of_domain": False
    },
    {
        "query": "Vietnam travel visa requirements and tour packages.",
        "expected_topic": "out_of_domain",
        "expected_doc_ids": [],
        "is_out_of_domain": True
    }
]

class RAGMLJudge:
    """LLM-as-a-judge evaluator for RAG outputs."""

    def __init__(self):
        self.llm = pipeline._editor_llm or pipeline._llm

    def _call_judge(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Invoke the LLM judge and parse a structured JSON score/reason output."""
        if self.llm is None:
            return {"score": 0.5, "reason": "LLM not available"}

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            raw_content = getattr(response, "content", str(response)).strip()
            
            # Clean markdown code blocks if returned
            clean_json = raw_content.strip()
            if clean_json.startswith("```"):
                # strip out ```json ... ```
                clean_json = re.sub(r"^```(?:json)?\n", "", clean_json)
                clean_json = re.sub(r"\n```$", "", clean_json)
                clean_json = clean_json.strip()

            score = 0.5
            reason = "Failed to parse LLM response"
            try:
                parsed = json.loads(clean_json)
                score = float(parsed.get("score", 0.5))
                reason = str(parsed.get("reason", "No reason provided")).strip()
            except Exception as json_err:
                print(f"[Warning] LLM judge standard JSON parsing failed: {json_err}")
                print(f"[Warning] Raw response was:\n{raw_content}\n")
                
                # Regex fallback for robustness
                score_match = re.search(r'"score"\s*:\s*([0-9.]+)', clean_json)
                reason_match = re.search(r'"reason"\s*:\s*"(.*?)"\s*(?:,|\})', clean_json, re.DOTALL)
                if not reason_match:
                    reason_match = re.search(r'"reason"\s*:\s*"(.*)"', clean_json, re.DOTALL)
                
                if score_match:
                    try:
                        score = float(score_match.group(1))
                    except ValueError:
                        pass
                
                if reason_match:
                    r_val = reason_match.group(1).strip()
                    if r_val.endswith('}'):
                        r_val = r_val[:-1].strip()
                    if r_val.endswith('"'):
                        r_val = r_val[:-1].strip()
                    reason = r_val
                else:
                    reason = f"Regex parsing fallback (JSON err: {json_err}). Raw output: {raw_content[:200]}"
            
            return {"score": min(1.0, max(0.0, score)), "reason": reason}
        except Exception as exc:
            return {"score": 0.5, "reason": f"Evaluation error: {exc}"}

    def judge_faithfulness(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        """Evaluate if the answer is grounded in the retrieved context documents."""
        system_prompt = (
            "You are a strict RAG Quality Judge evaluating FAITHFULNESS (grounding).\n"
            "Your task is to check if the generated answer is entirely grounded in the retrieved context documents.\n"
            "Rules:\n"
            "1. If the generated answer states 'Internal data does not currently address this topic' or declines to answer, and the context is empty/unrelated, this is a PERFECT refusal. Give it a score of 1.0.\n"
            "2. If the answer makes statements NOT supported by the context, or contradicts the context, give it a low score (0.0 to 0.3).\n"
            "3. If the answer is fully supported by the context, give it a high score (0.9 to 1.0).\n"
            "Respond ONLY with a valid JSON object in this format:\n"
            "{\n"
            "  \"score\": <float between 0.0 and 1.0>,\n"
            "  \"reason\": \"<brief justification>\"\n"
            "}\n"
            "Do NOT include raw newlines or unescaped double quotes inside the \"reason\" field value."
        )
        user_prompt = (
            f"### User Query: {query}\n\n"
            f"### Retrieved Context:\n{context}\n\n"
            f"### Generated Answer:\n{answer}"
        )
        return self._call_judge(system_prompt, user_prompt)

    def judge_relevance(self, query: str, answer: str) -> Dict[str, Any]:
        """Evaluate if the generated answer directly addresses the user query."""
        system_prompt = (
            "You are a strict RAG Quality Judge evaluating ANSWER RELEVANCE.\n"
            "Your task is to score how well the generated answer directly answers the user's initial query.\n"
            "Rules:\n"
            "1. The answer must address the user's core intent. If the user asks about visa requirements, but the answer talks about something unrelated, the score is 0.0.\n"
            "2. If the answer is a correct refusal (e.g. 'I don't have this internal data') to an out-of-scope query, this is a relevant refusal. Give it a score of 1.0.\n"
            "3. The score should range between 0.0 (irrelevant) and 1.0 (perfectly relevant and helpful).\n"
            "Respond ONLY with a valid JSON object in this format:\n"
            "{\n"
            "  \"score\": <float between 0.0 and 1.0>,\n"
            "  \"reason\": \"<brief justification>\"\n"
            "}\n"
            "Do NOT include raw newlines or unescaped double quotes inside the \"reason\" field value."
        )
        user_prompt = (
            f"### User Query: {query}\n\n"
            f"### Generated Answer:\n{answer}"
        )
        return self._call_judge(system_prompt, user_prompt)


def calculate_rr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    for idx, rid in enumerate(retrieved_ids):
        if any(expected in rid.lower() for expected in expected_ids):
            return 1.0 / (idx + 1)
    return 0.0


def run_evaluation():
    print("=== Launching ML-as-a-Judge RAG Evaluation ===")
    judge = RAGMLJudge()
    
    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0
    total_faithfulness = 0.0
    total_relevance = 0.0
    
    eval_results = []

    for item in EVAL_DATASET:
        query = item["query"]
        print(f"\nProcessing query: '{query}'")
        
        start_time = time.time()
        res = pipeline.run(query)
        latency = time.time() - start_time
        
        retrieved = res.get("retrieved", [])
        retrieved_ids = [doc["doc_id"] for doc in retrieved]
        draft = res.get("generated", {}).get("draft", "")
        
        # 1. Classical Retrieval Metrics
        relevant_retrieved = 0
        for doc in retrieved:
            doc_id = doc["doc_id"]
            if any(expected in doc_id.lower() for expected in item["expected_doc_ids"]):
                relevant_retrieved += 1
            elif doc["score"] >= 2 or doc["semantic_score"] >= 0.2:
                relevant_retrieved += 1
        
        k = len(retrieved)
        precision_at_k = relevant_retrieved / k if k > 0 else (1.0 if item["is_out_of_domain"] else 0.0)
        
        expected_len = len(item["expected_doc_ids"])
        recall_at_k = relevant_retrieved / expected_len if expected_len > 0 else (1.0 if item["is_out_of_domain"] else 0.0)
        
        rr = calculate_rr(retrieved_ids, item["expected_doc_ids"]) if not item["is_out_of_domain"] else 1.0
        
        # Format context text for the LLM judge
        context_docs = [
            {"page_content": doc["snippet"], "metadata": {"doc_id": doc["doc_id"]}}
            for doc in retrieved
        ]
        # Using pipeline's standard formatting helper
        from langchain_core.documents import Document
        langchain_docs = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in context_docs
        ]
        formatted_context = pipeline._format_context(langchain_docs)

        # 2. ML Judges Evaluation
        print("Calling Faithfulness Judge...")
        faithfulness_res = judge.judge_faithfulness(query, formatted_context, draft)
        print(f"-> Faithfulness score: {faithfulness_res['score']} | Reason: {faithfulness_res['reason']}")

        print("Calling Answer Relevance Judge...")
        relevance_res = judge.judge_relevance(query, draft)
        print(f"-> Relevance score: {relevance_res['score']} | Reason: {relevance_res['reason']}")
        
        total_precision += precision_at_k
        total_recall += recall_at_k
        total_mrr += rr
        total_faithfulness += faithfulness_res["score"]
        total_relevance += relevance_res["score"]
        
        eval_results.append({
            "query": query,
            "latency_seconds": round(latency, 2),
            "retrieval": {
                "precision_at_k": round(precision_at_k, 2),
                "recall_at_k": round(recall_at_k, 2),
                "reciprocal_rank": round(rr, 2),
                "retrieved_ids": retrieved_ids
            },
            "ml_judge": {
                "faithfulness_score": faithfulness_res["score"],
                "faithfulness_reason": faithfulness_res["reason"],
                "relevance_score": relevance_res["score"],
                "relevance_reason": relevance_res["reason"]
            },
            "generation_mode": res["runtime"]["generation_mode"]
        })

    num_queries = len(EVAL_DATASET)
    m_precision = total_precision / num_queries
    m_recall = total_recall / num_queries
    m_mrr = total_mrr / num_queries
    m_faithfulness = total_faithfulness / num_queries
    m_relevance = total_relevance / num_queries

    summary = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": {
            "mean_precision_at_k": round(m_precision, 3),
            "mean_recall_at_k": round(m_recall, 3),
            "mean_mrr": round(m_mrr, 3),
            "mean_faithfulness_score": round(m_faithfulness, 3),
            "mean_relevance_score": round(m_relevance, 3)
        },
        "query_details": eval_results
    }

    os.makedirs("data", exist_ok=True)
    output_file = "data/rag_ml_evaluation.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*40)
    print("=== ML RAG EVALUATION COMPLETE ===")
    print(f"Results written to: {output_file}")
    print(f"Mean Precision@k: {m_precision:.2%}")
    print(f"Mean Recall@k: {m_recall:.2%}")
    print(f"Mean MRR: {m_mrr:.2%}")
    print(f"Mean Faithfulness (Grounding): {m_faithfulness:.2%}")
    print(f"Mean Answer Relevance: {m_relevance:.2%}")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()
