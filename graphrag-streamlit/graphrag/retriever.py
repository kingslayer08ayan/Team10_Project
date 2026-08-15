"""
Stage 6: Retrieval.
Combines two signals per chunk:
  1. Vector similarity between the chunk and the HyDE hypothetical answer
     (catches semantically relevant chunks even with low keyword overlap
     to the raw query).
  2. A graph boost: chunks that mention entities belonging to the same
     community as entities matched by the query get a small score bump.
     This is the "Graph" in GraphRAG -- it lets a chunk surface because
     it's structurally connected to the topic, not just textually similar.
Returns the top-k chunks with full source/page citations plus a score
breakdown so results are auditable, not a black box.
"""
from . import hyde as hyde_module


def _query_matched_communities(query, graph, communities):
    if graph is None:
        return set()
    q_words = set(query.lower().split())
    matched = set()
    for name in graph.nodes:
        if any(w in name.lower() for w in q_words if len(w) > 3):
            matched.add(communities.get(name, -1))
    matched.discard(-1)
    return matched


def _chunk_graph_boost(chunk, graph, communities, matched_communities, boost_weight=0.15):
    if not matched_communities or graph is None:
        return 0.0
    # A chunk's boost = whether any entity it mentions belongs to a matched community
    for name, attrs in graph.nodes(data=True):
        if chunk.chunk_id in attrs.get("chunk_ids", set()):
            if communities.get(name, -1) in matched_communities:
                return boost_weight
    return 0.0


def retrieve(query, chunks, index, graph, communities, top_k=5, use_hyde=True):
    """
    Returns: dict with keys 'hyde_text', 'hyde_source', 'results' (list of
    dicts: chunk, vector_score, graph_boost, final_score), ready to render.
    """
    if use_hyde:
        hyde_text, hyde_source = hyde_module.generate(query, graph)
        search_text = hyde_text
    else:
        hyde_text, hyde_source = query, "n/a (HyDE disabled)"
        search_text = query

    query_vec = index.embed_query(search_text)
    vector_scores = index.similarity_scores(query_vec)

    matched_communities = _query_matched_communities(query, graph, communities)

    scored = []
    for chunk, vscore in zip(chunks, vector_scores):
        boost = _chunk_graph_boost(chunk, graph, communities, matched_communities)
        final = float(vscore) + boost
        scored.append({
            "chunk": chunk,
            "vector_score": float(vscore),
            "graph_boost": boost,
            "final_score": final,
        })

    scored.sort(key=lambda r: r["final_score"], reverse=True)
    top = [r for r in scored[:top_k] if r["final_score"] > 0]

    return {
        "hyde_text": hyde_text,
        "hyde_source": hyde_source,
        "matched_communities": matched_communities,
        "results": top,
    }
