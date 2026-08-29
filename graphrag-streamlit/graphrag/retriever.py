"""
Stage 6: Retrieval.

Two signals combined per chunk:

1. VECTOR SCORE (from ChromaDB / fastembed)
   Chroma returns cosine distance; embeddings.py converts to similarity
   so scores sit in [0, 1] with higher = more relevant.

2. HOP-WEIGHTED GRAPH BOOST
   Instead of a flat community-membership check, we now do a real BFS
   traversal from each query-matched entity node, up to 2 hops out.
   Each entity found along the traversal gets a score contribution:

       contribution = (edge_weight / max_edge_weight) * hop_decay[hop]

   where hop_decay = {0: 1.0, 1: 0.5, 2: 0.2}

   A chunk's boost = max contribution across all entities it mentions.
   We take max (not sum) so a chunk mentioning 5 weak 2-hop entities
   doesn't outrank one mentioning the single strongest 1-hop entity.

   The final boost is scaled to BASE_BOOST_FRACTION of the top vector
   score in this result set, not a fixed constant, so the boost stays
   proportionally meaningful regardless of how compressed the embedding
   similarity distribution is (dense embeddings cluster higher than
   TF-IDF, so a fixed 0.15 additive would distort rankings).

FINAL SCORE = vector_score + graph_boost + section_prior

3. SECTION PRIOR (query-dependent)
   If the query contains words suggesting intent ("limitation", "gap",
   "method", "dataset", ...), chunks from the matching canonical section
   get a small additive nudge. A query about "limitations" should surface
   Discussion-section chunks over Methodology chunks even at equal vector
   score -- this is a cheap, interpretable prior, not a learned model.
"""

import re
import numpy as np
from collections import deque
from . import hyde as hyde_module

HOP_DECAY = {0: 1.0, 1: 0.5, 2: 0.2}
MAX_HOPS = 2
BASE_BOOST_FRACTION = 0.15  # graph boost <= this fraction of the top vector score
SECTION_PRIOR_FRACTION = 0.08  # section prior <= this fraction of the top vector score

# Keyword -> canonical section it suggests. Checked as substrings against
# the lowercased query; first match per section family wins.
SECTION_PRIOR_KEYWORDS = {
    "Discussion": ["limitation", "gap", "weakness", "open problem", "future work",
                  "shortcoming", "drawback", "unresolved"],
    "Methodology": ["method", "approach", "architecture", "algorithm", "how does",
                    "how do", "technique", "framework"],
    "Observations": ["result", "performance", "accuracy", "benchmark", "evaluation",
                     "how well", "outperform", "metric"],
    "Related Work": ["prior work", "existing", "previous", "compared to", "literature"],
    "Introduction": ["motivation", "why", "problem statement", "overview"],
    "Conclusion": ["summary", "overall", "in conclusion", "takeaway"],
}

# Evidence reranking is deliberately a bounded second pass over the best
# initial candidates, so it improves evidence selection without rescoring the
# entire corpus with another model call.
EVIDENCE_CANDIDATE_COUNT = 20
INITIAL_WEIGHTS = (0.70, 0.20, 0.10)  # vector, graph, section prior
EVIDENCE_WEIGHTS = (0.45, 0.25, 0.20, 0.10)  # initial, semantic, type, claim
EVIDENCE_SECTIONS = {
    "Observations", "Discussion", "Conclusion", "Methodology", "Introduction",
}
EVIDENCE_TYPE_KEYWORDS = {
    "Observations": ("result", "results", "evaluat", "benchmark", "outperform",
                      "improv", "performance", "comparison", "compared", "negative"),
    "Discussion": ("limitation", "limited", "weakness", "shortcoming", "future work",
                    "remain", "cannot", "fail", "unresolved", "challenge"),
    "Conclusion": ("limitation", "future work", "remain", "conclude", "challenge"),
    "Methodology": ("method", "approach", "model", "algorithm", "propose", "framework"),
    "Introduction": ("problem", "lack", "challenge", "difficult", "motivat", "gap"),
}
STOPWORDS = {
    "about", "which", "what", "where", "when", "that", "this", "with", "from",
    "paper", "papers", "does", "their", "there", "have", "into", "under", "are",
}


# ------------------------------------------------------------------
# Graph traversal
# ------------------------------------------------------------------

def _bfs_entity_scores(seed_entities, graph, max_hops=MAX_HOPS):
    """
    BFS from each seed entity up to max_hops.
    Returns dict: entity_name -> best_contribution_score
    where contribution = (edge_weight / max_edge_weight) * hop_decay[hop]
    """
    if graph is None or graph.number_of_edges() == 0:
        return {}

    # Normalise edge weights across the whole graph once
    all_weights = [d.get("weight", 1) for _, _, d in graph.edges(data=True)]
    max_w = max(all_weights) if all_weights else 1

    entity_scores = {}

    for seed in seed_entities:
        if seed not in graph:
            continue
        visited = {seed: 0}   # node -> hop distance
        queue = deque([(seed, 0, 1.0)])   # (node, hop, inherited_weight_fraction)

        while queue:
            node, hop, parent_w_frac = queue.popleft()
            contribution = parent_w_frac * HOP_DECAY[hop]
            entity_scores[node] = max(entity_scores.get(node, 0.0), contribution)

            if hop < max_hops:
                for neighbor in graph.neighbors(node):
                    if neighbor in visited and visited[neighbor] <= hop + 1:
                        continue
                    visited[neighbor] = hop + 1
                    edge_w = graph[node][neighbor].get("weight", 1)
                    w_frac = edge_w / max_w
                    queue.append((neighbor, hop + 1, w_frac))

    return entity_scores


def _match_query_entities(query, graph):
    """
    Find graph nodes whose name appears in the query text.
    Requires at least 4 chars to avoid single-letter false matches.
    """
    if graph is None:
        return []
    q_lower = query.lower()
    return [name for name in graph.nodes
            if len(name) >= 4 and name.lower() in q_lower]


def _matched_prior_sections(query):
    """Returns the set of canonical sections whose keywords appear in the query."""
    q_lower = query.lower()
    matched = set()
    for section, keywords in SECTION_PRIOR_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            matched.add(section)
    return matched


def _chunk_section_prior(chunk, matched_sections, top_vector_score):
    """Flat additive nudge if this chunk's canonical section matches
    what the query's phrasing suggests it's looking for."""
    if not matched_sections:
        return 0.0
    if getattr(chunk, "canonical_section", None) in matched_sections:
        return SECTION_PRIOR_FRACTION * top_vector_score
    return 0.0


def _chunk_graph_boost(chunk, entity_scores, graph, top_vector_score,
                        chunk_entities=None):
    """
    Returns the boost for one chunk: max contribution across all entities
    it mentions, scaled to BASE_BOOST_FRACTION of the top vector score.
    """
    if not entity_scores or graph is None:
        return 0.0

    best = max((entity_scores.get(name, 0.0) for name in
                (chunk_entities or {}).get(chunk.chunk_id, ())), default=0.0)
    if best == 0.0 and chunk_entities is None:
        for name, attrs in graph.nodes(data=True):
            if chunk.chunk_id in attrs.get("chunk_ids", set()):
                best = max(best, entity_scores.get(name, 0.0))

    return best * BASE_BOOST_FRACTION * top_vector_score


def _claim_relevance(query, text):
    query_terms = {term for term in re.findall(r"[a-z0-9]{3,}", query.lower())
                   if term not in STOPWORDS}
    if not query_terms:
        return 0.0
    text_terms = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
    return len(query_terms & text_terms) / len(query_terms)


def _evidence_type_score(chunk, text):
    section = getattr(chunk, "canonical_section", "")
    keywords = EVIDENCE_TYPE_KEYWORDS.get(section, ())
    if section not in EVIDENCE_SECTIONS or not keywords:
        return 0.0
    text_lower = text.lower()
    matches = sum(keyword in text_lower for keyword in keywords)
    return min(1.0, matches / 3.0)


def _rerank_evidence(candidates, query, top_vector_score):
    """Rerank only initial candidates using research-evidence signals."""
    for result in candidates:
        initial = (
            INITIAL_WEIGHTS[0] * result["vector_score"] / top_vector_score
            + INITIAL_WEIGHTS[1] * result["graph_boost"] / top_vector_score
            + INITIAL_WEIGHTS[2] * result["section_prior"] / top_vector_score
        )
        chunk = result["chunk"]
        semantic = result["vector_score"] / top_vector_score
        evidence_type = _evidence_type_score(chunk, chunk.text)
        claim_relevance = _claim_relevance(query, chunk.text)
        evidence_score = (
            EVIDENCE_WEIGHTS[0] * initial
            + EVIDENCE_WEIGHTS[1] * semantic
            + EVIDENCE_WEIGHTS[2] * evidence_type
            + EVIDENCE_WEIGHTS[3] * claim_relevance
        )
        result["initial_score"] = round(initial, 4)
        result["semantic_score"] = round(semantic, 4)
        result["evidence_type_score"] = round(evidence_type, 4)
        result["claim_relevance"] = round(claim_relevance, 4)
        result["evidence_score"] = round(evidence_score, 4)
    candidates.sort(key=lambda result: result["evidence_score"], reverse=True)
    return candidates


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def retrieve(query, chunks, index, graph, communities, top_k=5, use_hyde=True):
    """
    Returns dict:
      hyde_text        str
      hyde_source      'llm' | 'offline' | 'n/a'
      matched_entities list of entity names matched in query
      hop_entity_scores dict entity -> traversal score (for debugging)
      results          list of dicts:
                         chunk, vector_score, graph_boost, final_score
    """
    if use_hyde:
        hyde_text, hyde_source = hyde_module.generate(query, graph)
        search_text = hyde_text
    else:
        hyde_text, hyde_source = query, "n/a (HyDE disabled)"
        search_text = query

    # --- vector scores from Chroma (or TF-IDF fallback) ---
    query_vec = index.embed_query(search_text)
    vector_scores = index.similarity_scores(
        query_vec, top_n=min(len(chunks), max(top_k * 8, EVIDENCE_CANDIDATE_COUNT * 2))
    )

    top_vector_score = float(np.max(vector_scores)) if len(vector_scores) else 1.0
    if top_vector_score == 0:
        top_vector_score = 1.0  # avoid divide-by-zero on empty corpus

    # --- graph traversal for boost ---
    matched_entities = _match_query_entities(query, graph)
    entity_scores = _bfs_entity_scores(matched_entities, graph)
    chunk_entities = {}
    for name, attrs in graph.nodes(data=True):
        for chunk_id in attrs.get("chunk_ids", set()):
            chunk_entities.setdefault(chunk_id, []).append(name)

    # --- section priors from query phrasing ---
    matched_sections = _matched_prior_sections(query)

    # --- combine and rank ---
    scored = []
    for chunk, vscore in zip(chunks, vector_scores):
        graph_boost = _chunk_graph_boost(
            chunk, entity_scores, graph, top_vector_score, chunk_entities
        )
        section_prior = _chunk_section_prior(chunk, matched_sections, top_vector_score)
        final = float(vscore) + graph_boost + section_prior
        scored.append({
            "chunk": chunk,
            "vector_score": round(float(vscore), 4),
            "graph_boost": round(graph_boost, 4),
            "section_prior": round(section_prior, 4),
            "final_score": round(final, 4),
        })

    scored.sort(key=lambda r: r["final_score"], reverse=True)

    # Evidence reranking happens after initial retrieval and before paper-level
    # deduplication, so each paper still competes using its best evidence chunk.
    evidence_candidates = scored[:min(EVIDENCE_CANDIDATE_COUNT, len(scored))]
    _rerank_evidence(evidence_candidates, query, top_vector_score)
    scored = evidence_candidates + scored[len(evidence_candidates):]

    # --- paper-level deduplication ---
    # Chroma ranks chunks independently, so the same PDF can fill all
    # top-k slots with different pages. Keep only the highest-scoring
    # chunk per doc_id so every result slot shows a different paper.
    # We oversample (top_k * 4) before dedup so we have enough candidates
    # to fill top_k slots even after collapsing duplicates.
    seen_docs = {}
    for r in scored:
        doc_id = r["chunk"].doc_id
        if doc_id not in seen_docs:
            seen_docs[doc_id] = r   # first occurrence = highest evidence score

    deduped = list(seen_docs.values())[:top_k]
    top = [r for r in deduped if r["final_score"] > 0]

    return {
        "hyde_text": hyde_text,
        "hyde_source": hyde_source,
        "matched_entities": matched_entities,
        "hop_entity_scores": entity_scores,
        "matched_sections": matched_sections,
        "evidence_results": evidence_candidates,
        "results": top,
    }
