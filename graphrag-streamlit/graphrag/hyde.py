"""
Stage 5: HyDE (Hypothetical Document Embeddings).
Short queries embed poorly against dense chunk text -- "does anything
handle domain shift?" shares almost no vocabulary with the papers that
actually answer it. HyDE fixes this by first generating a hypothetical
*answer* to the query (even a wrong one is fine -- it just needs to be
written in the vocabulary a real answer would use), then embedding and
searching with that instead of the raw query.

The offline fallback can't generate free text, so it approximates the
same effect: it finds the graph entities whose names best match the
query, then builds a pseudo-answer by stitching the query together with
those entity names and their types. This is a weaker version of the same
trick (borrow the target vocabulary before you search), not full HyDE --
swap in the LLM path for the real thing.
"""
from .llm_backend import reason

HYDE_SYSTEM_PROMPT = (
    "You are helping a research-review retrieval system. Given a user's "
    "question about a corpus of AI/healthcare research papers, write a short "
    "(3-5 sentence) hypothetical passage that would plausibly ANSWER the "
    "question, in the style of a paper abstract. Do not hedge or say you "
    "don't know -- invent plausible technical detail; it is only used to "
    "improve retrieval vocabulary, not shown to the user as fact."
)


def _offline_hyde(query, graph, top_n_entities=6):
    if graph is None or graph.number_of_nodes() == 0:
        return query

    q_words = set(query.lower().split())
    scored = []
    for name, attrs in graph.nodes(data=True):
        overlap = sum(1 for w in q_words if w in name.lower())
        if overlap > 0 or attrs.get("mentions", 0) > 1:
            scored.append((overlap, attrs.get("mentions", 0), name, attrs.get("type", "CONCEPT")))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = scored[:top_n_entities]

    if not top:
        return query

    parts = [query, "Relevant concepts:"]
    for _, _, name, typ in top:
        parts.append(f"{name} ({typ})")
    return " ".join(parts)


def generate(query, graph=None):
    """Returns (hypothetical_text, source) where source is 'llm' or 'offline'."""
    text, source = reason(
        f"Question: {query}", system=HYDE_SYSTEM_PROMPT, max_tokens=200,
        offline_fn=lambda: _offline_hyde(query, graph),
    )
    return text, source
