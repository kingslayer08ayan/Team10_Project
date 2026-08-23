"""
Evaluation metrics for the retrieval pipeline.

recall@k              - needs a small labeled eval set (query -> relevant
                        doc_ids you know are correct). Without ground
                        truth this can't be computed meaningfully, so it's
                        the one metric that requires you to supply
                        eval_queries yourself -- see EXAMPLE_EVAL_SET.
duplicate_rate         - fraction of a raw (pre-dedup) result set sharing
                        a doc_id. Should be ~0 post-dedup; useful to
                        confirm the dedup step in retriever.py is working
                        and to quantify how much it was doing before.
citation_correctness   - fraction of cited chunk_ids that actually exist
                        in the corpus. Catches hallucinated citations
                        (relevant once the Gap Finder/Critic agent starts
                        generating cited claims, not just raw retrieval).
answer_faithfulness    - offline heuristic (word overlap between an answer
                        and its cited source chunks) with an optional
                        LLM-judge upgrade via the existing llm_backend.
latency                - batch timing over a list of queries, reporting
                        mean/p50/p95.
"""
import time
import re
import numpy as np

from . import retriever as retriever_module
from .llm_backend import reason

# Fill this in with real (query, relevant_doc_ids) pairs from your own
# corpus before trusting recall@k -- there is no way to compute it
# without human-labeled ground truth.
EXAMPLE_EVAL_SET = [
     {"query": "which papers use federated learning?", "relevant_doc_ids": ["P04"]},
     {"query": "which papers use deep CNN?", "relevant_doc_ids": ["P01", "P17"]},
]


def recall_at_k(eval_queries, chunks, index, graph, communities, top_k=5, use_hyde=True):
    """
    eval_queries: list of {"query": str, "relevant_doc_ids": [str, ...]}
    Returns: {"mean_recall": float, "per_query": [{"query", "recall", "found", "missed"}]}
    """
    if not eval_queries:
        return {"mean_recall": None, "per_query": [],
                "note": "No labeled eval queries supplied -- recall@k needs "
                        "ground truth, see EXAMPLE_EVAL_SET."}

    per_query = []
    for item in eval_queries:
        result = retriever_module.retrieve(
            item["query"], chunks, index, graph, communities, top_k=top_k, use_hyde=use_hyde
        )
        retrieved_docs = {r["chunk"].doc_id for r in result["results"]}
        relevant = set(item["relevant_doc_ids"])
        found = retrieved_docs & relevant
        recall = len(found) / len(relevant) if relevant else None
        per_query.append({
            "query": item["query"],
            "recall": recall,
            "found": sorted(found),
            "missed": sorted(relevant - found),
        })

    valid = [p["recall"] for p in per_query if p["recall"] is not None]
    mean_recall = float(np.mean(valid)) if valid else None
    return {"mean_recall": mean_recall, "per_query": per_query}


def duplicate_rate(results):
    """
    results: list of retrieval result dicts (each with a "chunk" holding doc_id)
    Returns fraction of results sharing a doc_id with an earlier result
    in the same list. 0.0 = no duplicates.
    """
    if not results:
        return 0.0
    seen = set()
    dup_count = 0
    for r in results:
        doc_id = r["chunk"].doc_id
        if doc_id in seen:
            dup_count += 1
        seen.add(doc_id)
    return dup_count / len(results)


def citation_correctness(cited_chunk_ids, valid_chunk_ids):
    """
    cited_chunk_ids: chunk_ids referenced in a generated answer/report
    valid_chunk_ids: set of every real chunk_id in the corpus
    Returns fraction of citations that point to a real chunk (1.0 = no
    hallucinated citations).
    """
    if not cited_chunk_ids:
        return None
    valid = set(valid_chunk_ids)
    correct = sum(1 for cid in cited_chunk_ids if cid in valid)
    return correct / len(cited_chunk_ids)


def _tokenize(text):
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def answer_faithfulness_heuristic(answer_text, source_chunk_texts):
    """
    Offline heuristic: fraction of the answer's content words that also
    appear somewhere in the cited source chunks. Cheap, no LLM call, but
    only measures lexical overlap -- a paraphrased-but-faithful answer
    can score lower than it deserves. Use answer_faithfulness_llm for a
    more accurate (but paid/networked) judgment.
    """
    answer_words = _tokenize(answer_text)
    if not answer_words:
        return None
    source_words = set()
    for text in source_chunk_texts:
        source_words |= _tokenize(text)
    overlap = answer_words & source_words
    return len(overlap) / len(answer_words)


FAITHFULNESS_JUDGE_PROMPT = (
    "You are checking whether an ANSWER is faithful to its SOURCE passages "
    "-- i.e. every claim in the answer is actually supported by the sources, "
    "with no invented facts. Respond with only a number from 0.0 to 1.0 "
    "(1.0 = fully faithful, 0.0 = unsupported/contradicted), nothing else."
)


def answer_faithfulness_llm(answer_text, source_chunk_texts):
    """
    LLM-as-judge version. Requires ANTHROPIC_API_KEY or HF_TOKEN to be
    set; returns None if no backend is available (falls through silently
    rather than erroring, since this is an optional upgrade).
    """
    sources_joined = "\n---\n".join(source_chunk_texts)
    prompt = f"SOURCES:\n{sources_joined}\n\nANSWER:\n{answer_text}"
    result, source = reason(prompt, system=FAITHFULNESS_JUDGE_PROMPT, max_tokens=10,
                            offline_fn=lambda: None)
    if result is None:
        return None
    try:
        return max(0.0, min(1.0, float(result.strip())))
    except ValueError:
        return None


def benchmark_latency(queries, chunks, index, graph, communities, top_k=5, use_hyde=True):
    """
    Runs retrieve() over a list of query strings and times each call.
    Returns {"mean_ms", "p50_ms", "p95_ms", "per_query_ms": [...]}
    """
    timings = []
    for q in queries:
        t0 = time.perf_counter()
        retriever_module.retrieve(q, chunks, index, graph, communities,
                                  top_k=top_k, use_hyde=use_hyde)
        timings.append((time.perf_counter() - t0) * 1000)

    if not timings:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None, "per_query_ms": []}

    arr = np.array(timings)
    return {
        "mean_ms": round(float(np.mean(arr)), 1),
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "per_query_ms": [round(t, 1) for t in timings],
    }
