"""Evidence-backed gap and limitation analysis over retrieved chunks."""
import json
import re

from .llm_backend import reason

ANALYSIS_SECTIONS = {"Introduction", "Methodology", "Observations", "Discussion", "Conclusion"}
MAX_EVIDENCE_CHUNKS = 10
GAP_EVIDENCE_CUES = (
    "limitation", "limited", "weakness", "shortcoming", "challenge", "problem",
    "gap", "lack", "remain", "remains", "unresolved", "underexplored", "under-explored", "not addressed",
    "not evaluated", "without evaluating", "fails", "failure", "cannot", "difficult",
    "future work", "however", "although", "despite", "inconsistent",
)
STRONG_GAP_CUES = (
    "future work", "not evaluated", "not addressed", "without evaluating",
    "underexplored", "under-explored", "unresolved", "remains unclear", "lack of evidence",
    "has yet to be", "has not been",
)
VALID_CLAIM_TYPES = {"Gap", "Limitation", "Contradiction"}

GAP_ANALYSIS_SYSTEM_PROMPT = (
    "You are a research gap analyst. Extract only evidence-backed candidate claims "
    "from the supplied chunks and knowledge-graph context. Do not invent facts or "
    "treat a paper's Related Work as proof. A method description alone is not a gap; "
    "return no candidate unless the evidence states a limitation, unresolved problem, "
    "negative result, contradiction, missing evaluation, or future work. Return strict JSON with a top-level "
    "array. Each item must contain: claim, type (Gap|Limitation|Contradiction), "
    "supporting_chunk_ids, supporting_papers, relevant_entities, kg_relationships, "
    "evidence_type, and reasoning. A single paper's stated cost, speed, memory, "
    "or accuracy drawback is a Limitation, not a Gap. Reserve Gap for an explicit "
    "unresolved or missing research question, preferably supported across papers. "
    "Do not add strength words such as prominent, substantial, or significant unless "
    "the supplied evidence explicitly supports them. Keep claims concise and cite "
    "only supplied IDs."
)


def _chunk_kg_context(chunk, graph):
    entities = []
    relationships = []
    if graph is None:
        return entities, relationships
    for name, attrs in graph.nodes(data=True):
        if chunk.chunk_id not in attrs.get("chunk_ids", set()):
            continue
        entities.append(name)
        for neighbor in list(graph.neighbors(name))[:6]:
            edge = graph[name][neighbor]
            relation_types = ", ".join(sorted(edge.get("relation_types", {"co-occurs_with"})))
            relationships.append(f"{name} -[{relation_types}]-> {neighbor}")
    return sorted(set(entities)), sorted(set(relationships))


def build_evidence_context(retrieval_result, graph, limit=MAX_EVIDENCE_CHUNKS):
    context = []
    results = retrieval_result.get("evidence_results", retrieval_result.get("results", []))
    for result in results[:limit]:
        chunk = result["chunk"]
        if chunk.canonical_section not in ANALYSIS_SECTIONS:
            continue
        entities, relationships = _chunk_kg_context(chunk, graph)
        context.append({
            "chunk_id": chunk.chunk_id,
            "paper": chunk.doc_id,
            "section": chunk.canonical_section,
            "evidence_type": chunk.canonical_section,
            "text": chunk.text,
            "entities": entities,
            "kg_relationships": relationships,
            "scores": {
                "evidence": result.get("evidence_score", result.get("final_score", 0.0)),
                "initial": result.get("initial_score", result.get("final_score", 0.0)),
            },
        })
    return context


def _claim_type(text, section):
    lowered = text.lower()
    if any(
        word in lowered for word in ("limitation", "weakness", "unresolved", "future work")
    ):
        return "Limitation"
    if any(word in lowered for word in ("contradict", "inconsistent", "however", "but ")):
        return "Contradiction"
    return "Gap"


def _has_gap_evidence(text, section):
    lowered = text.lower()
    return any(cue in lowered for cue in GAP_EVIDENCE_CUES)


def _has_strong_gap_evidence(text):
    lowered = text.lower()
    return any(cue in lowered for cue in STRONG_GAP_CUES)


def _offline_candidates(context):
    candidates = []
    for item in context:
        if not _has_gap_evidence(item["text"], item["section"]):
            continue
        sentences = re.split(r"(?<=[.!?])\s+", item["text"].strip())
        sentence = next((s for s in sentences if len(s.split()) >= 6), item["text"][:240])
        candidates.append({
            "claim": sentence[:300],
            "type": _claim_type(item["text"], item["section"]),
            "supporting_chunk_ids": [item["chunk_id"]],
            "supporting_papers": [item["paper"]],
            "relevant_entities": item["entities"],
            "kg_relationships": item["kg_relationships"],
            "evidence_type": item["evidence_type"],
            "reasoning": "Extracted from a retrieved evidence chunk without an LLM.",
        })
    return candidates


def _parse_candidates(text):
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        parsed = parsed.get("candidates", parsed.get("claims", []))
    return parsed if isinstance(parsed, list) else []


def _filter_unsupported_candidates(candidates, context):
    """Reject method-only claims that lack explicit unresolved evidence."""
    by_id = {item["chunk_id"]: item for item in context}
    filtered = []
    for candidate in candidates:
        if candidate.get("type") not in VALID_CLAIM_TYPES:
            continue
        evidence = [by_id[cid] for cid in candidate.get("supporting_chunk_ids", [])
                    if cid in by_id]
        if not evidence:
            continue
        if not any(_has_gap_evidence(item["text"], item["section"])
                   for item in evidence):
            continue
        candidate["supporting_papers"] = list(dict.fromkeys(item["paper"] for item in evidence))
        claim = candidate.get("claim", "")
        if candidate.get("type") == "Gap":
            evidence_text = " ".join(item["text"] for item in evidence)
            # A gap must name an explicit unresolved or missing area. Generic
            # difficulty claims are limitations or unsupported assertions.
            if not (_has_strong_gap_evidence(claim)
                    or _has_strong_gap_evidence(evidence_text)):
                continue
        candidate["supporting_chunk_ids"] = [item["chunk_id"] for item in evidence]
        filtered.append(candidate)
    return filtered


def _token_set(text):
    return set(re.findall(r"[a-z0-9]{4,}", text.lower()))


def _similar_claim(left, right):
    a, b = _token_set(left), _token_set(right)
    return bool(a and b) and len(a & b) / len(a | b) >= 0.45


def aggregate_candidates(candidates):
    bundles = []
    for candidate in candidates:
        candidate["supporting_chunk_ids"] = list(dict.fromkeys(candidate.get("supporting_chunk_ids", [])))
        candidate["supporting_papers"] = list(dict.fromkeys(candidate.get("supporting_papers", [])))
        candidate["relevant_entities"] = list(dict.fromkeys(candidate.get("relevant_entities", [])))
        candidate["kg_relationships"] = list(dict.fromkeys(candidate.get("kg_relationships", [])))
        match = next((bundle for bundle in bundles
                      if bundle["type"] == candidate.get("type")
                      and _similar_claim(bundle["claim"], candidate.get("claim", ""))), None)
        if match is None:
            bundles.append({
                "claim": candidate.get("claim", "").strip(),
                "type": candidate.get("type", "Gap"),
                "supporting_chunk_ids": candidate["supporting_chunk_ids"],
                "supporting_papers": candidate["supporting_papers"],
                "relevant_entities": candidate["relevant_entities"],
                "kg_relationships": candidate["kg_relationships"],
                "evidence_types": [candidate.get("evidence_type", "")],
                "reasoning": candidate.get("reasoning", ""),
            })
        else:
            for field in ("supporting_chunk_ids", "supporting_papers", "relevant_entities", "kg_relationships"):
                match[field] = list(dict.fromkeys(match[field] + candidate[field]))
            match["evidence_types"] = list(dict.fromkeys(
                match["evidence_types"] + [candidate.get("evidence_type", "")]
            ))
    for index, bundle in enumerate(bundles, 1):
        bundle["gap_id"] = f"G{index}"
        bundle["supporting_paper_count"] = len(bundle["supporting_papers"])
        bundle["supporting_chunk_count"] = len(bundle["supporting_chunk_ids"])
        bundle["independent_evidence"] = bundle["supporting_paper_count"] > 1
        explicit = sum("explicit" in value.lower() or value in {"Discussion", "Observations"}
                       for value in bundle["evidence_types"])
        contradiction_penalty = 0.15 if bundle["type"] == "Contradiction" else 0.0
        bundle["confidence"] = round(min(1.0, 0.45
            + 0.12 * min(bundle["supporting_paper_count"], 3)
            + 0.05 * min(bundle["supporting_chunk_count"], 4)
            + 0.05 * min(explicit, 2) - contradiction_penalty), 2)
        bundle["importance"] = round(min(1.0, 0.45
            + 0.08 * min(bundle["supporting_paper_count"], 4)
            + (0.12 if bundle["type"] in {"Gap", "Limitation"} else 0.05)
            + (0.10 if any(word in bundle["claim"].lower()
                           for word in ("scalab", "generaliz", "bias", "robust", "evaluat")) else 0.0)), 2)
    return bundles


def build_landscape(candidates):
    """Create the structured evidence table used by later review generation."""
    rows = []
    for candidate in candidates:
        for paper in candidate["supporting_papers"]:
            rows.append({
                "paper": paper,
                "claim_type": candidate["type"],
                "claim": candidate["claim"],
                "evidence_type": ", ".join(candidate["evidence_types"]),
                "evidence": candidate["supporting_chunk_ids"],
                "entities": candidate["relevant_entities"],
                "confidence": candidate["confidence"],
                "importance": candidate["importance"],
            })
    return rows


def rerank_candidates(candidates, query):
    """Prioritize extracted claims using evidence and corpus-level signals."""
    query_terms = {
        term for term in re.findall(r"[a-z0-9]{3,}", query.lower())
        if term not in {"the", "and", "for", "with", "from", "into", "about"}
    }
    for candidate in candidates:
        claim_terms = set(re.findall(r"[a-z0-9]{3,}", candidate["claim"].lower()))
        query_relevance = (
            len(query_terms & claim_terms) / len(query_terms) if query_terms else 0.0
        )
        paper_support = min(candidate["supporting_paper_count"] / 3.0, 1.0)
        chunk_support = min(candidate["supporting_chunk_count"] / 4.0, 1.0)
        validation_status = candidate.get("validation_status", "")
        corroboration = 1.0 if validation_status == "corroborated_within_corpus" else 0.0
        candidate["priority_score"] = round(min(1.0, (
            0.25 * candidate.get("confidence", 0.0)
            + 0.20 * candidate.get("importance", 0.0)
            + 0.20 * paper_support
            + 0.10 * chunk_support
            + 0.15 * query_relevance
            + 0.10 * corroboration
        )), 3)
        candidate["priority_reasons"] = []
        if paper_support >= 0.67:
            candidate["priority_reasons"].append("supported across multiple papers")
        if corroboration:
            candidate["priority_reasons"].append("corroborated within corpus")
        if query_relevance >= 0.5:
            candidate["priority_reasons"].append("closely matches research question")
        if not candidate["priority_reasons"]:
            candidate["priority_reasons"].append("supported by selected evidence")
    return sorted(candidates, key=lambda candidate: candidate["priority_score"], reverse=True)


def validate_candidates(candidates, chunks, index, graph, communities):
    """Run targeted corpus checks; absence is scoped to this corpus only."""
    from . import retriever
    validated = []
    for candidate in candidates:
        target_query = candidate["claim"]
        result = retriever.retrieve(target_query, chunks, index, graph, communities,
                                    top_k=5, use_hyde=False)
        found = [item["chunk"].chunk_id for item in result["results"]
                 if item["chunk"].doc_id not in candidate["supporting_papers"]]
        item = dict(candidate)
        item["validation_query"] = target_query
        item["validation_chunk_ids"] = found
        if found:
            item["validation_status"] = "corroborated_within_corpus"
            if candidate["type"] == "Limitation":
                item["validation_note"] = "Related evidence was found within the corpus; this supports the reported limitation."
            else:
                item["validation_note"] = "Related evidence was found within the corpus; this weakens the claim that the issue is a gap."
        else:
            item["validation_status"] = "no_evidence_within_current_corpus"
            if candidate["type"] == "Limitation":
                item["validation_note"] = "No independent supporting evidence was found within the current corpus."
            else:
                item["validation_note"] = "No related evidence was found within the current corpus; this is not proof that the gap exists beyond this corpus."
        validated.append(item)
    return validated


def analyze(query, retrieval_result, graph):
    context = build_evidence_context(retrieval_result, graph)
    if not context:
        return {"source": "offline", "evidence": [], "candidates": [], "note": "No analytical evidence chunks retrieved."}
    prompt = (
        f"Research question: {query}\n\n"
        "Evidence context (cite only these IDs):\n"
        f"{json.dumps(context, ensure_ascii=True)}"
    )
    raw, source = reason(
        prompt, system=GAP_ANALYSIS_SYSTEM_PROMPT, max_tokens=1400,
        offline_fn=lambda: json.dumps(_offline_candidates(context)),
    )
    try:
        candidates = _parse_candidates(raw)
    except (TypeError, json.JSONDecodeError):
        candidates = _offline_candidates(context)
        source = "offline"
    candidates = _filter_unsupported_candidates(candidates, context)
    return {"source": source, "evidence": context,
            "candidates": aggregate_candidates(candidates)}