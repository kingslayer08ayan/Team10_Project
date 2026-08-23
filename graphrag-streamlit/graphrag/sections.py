"""
Canonical section schema + classifier.

Maps the wide variety of section names papers actually use ("Methods",
"Approach", "Proposed Framework", ...) onto a fixed schema, so retrieval,
gap-finding, and the Critic agent can all filter/prioritize by section
meaning instead of by whatever string a given paper happened to use.

Three-tier cascade, cheapest first:
  1. Alias dict + fuzzy match   -- instant, zero cost, handles most cases
  2. Embedding classification   -- for headers Tier 1 didn't confidently match
  3. Position heuristic         -- last resort, uses where in the doc it sits
"""
import difflib

CANONICAL_SECTIONS = [
    "Abstract",
    "Introduction",
    "Related Work",
    "Methodology",
    "Observations",
    "Discussion",
    "Conclusion",
]

# Tier 1: known aliases -> canonical name. Keys are lowercased for matching.
SECTION_ALIASES = {
    "abstract": "Abstract",
    "summary": "Abstract",

    "introduction": "Introduction",
    "background": "Introduction",
    "motivation": "Introduction",
    "overview": "Introduction",
    "preliminaries": "Introduction",

    "related work": "Related Work",
    "prior work": "Related Work",
    "previous work": "Related Work",
    "literature review": "Related Work",
    "state of the art": "Related Work",

    "methodology": "Methodology",
    "methods": "Methodology",
    "method": "Methodology",
    "approach": "Methodology",
    "proposed method": "Methodology",
    "our method": "Methodology",
    "proposed framework": "Methodology",
    "framework": "Methodology",
    "architecture": "Methodology",
    "system design": "Methodology",
    "model": "Methodology",
    "technical approach": "Methodology",

    "results": "Observations",
    "experiments": "Observations",
    "experimental results": "Observations",
    "empirical evaluation": "Observations",
    "evaluation": "Observations",
    "performance analysis": "Observations",
    "quantitative results": "Observations",
    "benchmarks": "Observations",
    "ablation study": "Observations",
    "ablation": "Observations",

    "discussion": "Discussion",
    "limitations": "Discussion",
    "limitations and future work": "Discussion",
    "threats to validity": "Discussion",
    "broader impact": "Discussion",
    "caveats": "Discussion",

    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "summary and conclusion": "Conclusion",
    "future work": "Conclusion",
}

FUZZY_MATCH_CUTOFF = 0.78  # difflib similarity ratio threshold


def classify_tier1(header_text):
    """
    Exact/fuzzy alias match. Returns (canonical_name, confidence) or (None, 0.0).
    confidence is 1.0 for exact match, ~0.78-0.99 for fuzzy.
    """
    key = header_text.strip().lower().rstrip(":")
    if key in SECTION_ALIASES:
        return SECTION_ALIASES[key], 1.0

    matches = difflib.get_close_matches(key, SECTION_ALIASES.keys(),
                                        n=1, cutoff=FUZZY_MATCH_CUTOFF)
    if matches:
        ratio = difflib.SequenceMatcher(None, key, matches[0]).ratio()
        return SECTION_ALIASES[matches[0]], round(ratio, 2)

    return None, 0.0


def classify_tier2(header_text, context_text, embedder=None):
    """
    Embedding-based classification using (header + first ~200 chars of
    section content) against canonical section name embeddings.
    Returns (canonical_name, confidence) or (None, 0.0) if no embedder.
    """
    if embedder is None:
        return None, 0.0

    import numpy as np
    query_text = f"{header_text}. {context_text[:200]}"
    query_vec = np.array(list(embedder.embed([query_text]))[0])

    canonical_vecs = np.array(list(embedder.embed(CANONICAL_SECTIONS)))
    sims = canonical_vecs @ query_vec / (
        np.linalg.norm(canonical_vecs, axis=1) * np.linalg.norm(query_vec) + 1e-9
    )
    best_idx = int(np.argmax(sims))
    return CANONICAL_SECTIONS[best_idx], round(float(sims[best_idx]), 2)


def classify_tier3_position(position_fraction):
    """
    Fallback using position in the document (0.0 = start, 1.0 = end).
    No confidence score -- this is a guess, always tagged as such.
    """
    if position_fraction < 0.08:
        return "Introduction"
    elif position_fraction < 0.15:
        return "Introduction"
    elif position_fraction > 0.92:
        return "Conclusion"
    elif position_fraction > 0.55:
        return "Observations"
    else:
        return "Methodology"


def classify_section(header_text, context_text, position_fraction, embedder=None):
    """
    Runs the full cascade. Returns (canonical_name, original_header, confidence, tier).
    tier is 1, 2, or 3 -- lower is more trustworthy, and callers (like the
    Critic agent) can choose to only trust tier-1/2 sections for structural
    claims, treating tier-3 as a soft guess.
    """
    original = header_text.strip()

    name, conf = classify_tier1(header_text)
    if name:
        return name, original, conf, 1

    name, conf = classify_tier2(header_text, context_text, embedder)
    if name and conf >= 0.5:
        return name, original, conf, 2

    name = classify_tier3_position(position_fraction)
    return name, original, 0.3, 3  # flat low confidence, it's a positional guess
