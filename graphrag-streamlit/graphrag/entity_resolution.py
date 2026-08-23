"""
Entity resolution: merge synonymous entity names ("CNN" <-> "Convolutional
Neural Network") into one canonical node before the graph is built.

Two layers, cheapest first:

  1. Acronym-alias extraction (always on, free, zero ambiguity)
     Academic text almost always spells out an acronym on first use:
     "...a Convolutional Neural Network (CNN)...". A regex catches this
     exact pattern directly from the corpus text -- no embedding call
     needed, and no false positives since the paper defined the link
     itself.

  2. Embedding-based merge (only when an embedder is available)
     For everything the alias regex didn't catch (different phrasings
     that were never explicitly linked in text -- "ConvNet", "conv net",
     etc.), embed each unique entity name and union-find merge names
     whose cosine similarity clears MERGE_THRESHOLD.

     MERGE_THRESHOLD is a genuine judgment call, not a validated
     constant -- 0.82 is a reasonable starting point for bge-small on
     short technical phrases, but it needs calibration against your
     actual corpus (too low merges "CNN" with "RNN"; too high misses
     real synonyms). Log what got merged and eyeball it before trusting
     this at scale.

Without an embedder, only layer 1 runs -- entities stay unresolved
beyond what the corpus explicitly spelled out, which is an honest
limitation, not a silent failure.
"""
import re
import numpy as np

ACRONYM_ALIAS_RE = re.compile(
    r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,5})\s*\(([A-Z]{2,6})\)"
)

MERGE_THRESHOLD = 0.82


def _initials_match(phrase, acronym):
    """Loose check that the acronym plausibly stands for the phrase --
    e.g. 'Convolutional Neural Network' -> 'CNN'. Not exact-strict since
    papers sometimes skip minor words ('a', 'of') in the acronym."""
    words = [w for w in phrase.split() if w[0].isupper()]
    initials = "".join(w[0] for w in words)
    return acronym in initials or initials.endswith(acronym) or initials.startswith(acronym)


def extract_acronym_aliases(all_chunk_texts):
    """
    Scans corpus text for "Full Phrase (ACRONYM)" patterns.
    Returns dict: acronym -> full_phrase (the canonical form to merge to).
    """
    aliases = {}
    for text in all_chunk_texts:
        for match in ACRONYM_ALIAS_RE.finditer(text):
            phrase, acronym = match.group(1).strip(), match.group(2).strip()
            if _initials_match(phrase, acronym):
                # Keep the first (or longest-seen) definition if an
                # acronym is defined more than once in the corpus
                if acronym not in aliases or len(phrase) > len(aliases[acronym]):
                    aliases[acronym] = phrase
    return aliases


class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def embedding_merge(entity_names, embedder, threshold=MERGE_THRESHOLD):
    """
    entity_names: list of unique entity name strings (post acronym-alias step)
    embedder: a fastembed TextEmbedding instance, or None to skip this layer
    Returns dict: original_name -> canonical_name (may map to itself)
    """
    if embedder is None or len(entity_names) < 2:
        return {n: n for n in entity_names}

    names = list(entity_names)
    vecs = np.array(list(embedder.embed(names)))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    unit_vecs = vecs / norms
    sim_matrix = unit_vecs @ unit_vecs.T

    uf = _UnionFind(names)
    merge_log = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if sim_matrix[i, j] >= threshold:
                uf.union(names[i], names[j])
                merge_log.append((names[i], names[j], round(float(sim_matrix[i, j]), 3)))

    # Pick canonical label per cluster: prefer the longest name (usually
    # the spelled-out form rather than an abbreviation)
    clusters = {}
    for n in names:
        root = uf.find(n)
        clusters.setdefault(root, []).append(n)

    mapping = {}
    for members in clusters.values():
        canonical = max(members, key=len)
        for m in members:
            mapping[m] = canonical

    if merge_log:
        print(f"[entity_resolution] Embedding merge combined {len(merge_log)} pairs "
              f"(threshold={threshold}):")
        for a, b, sim in merge_log[:20]:
            print(f"    '{a}' <-> '{b}'  (sim={sim})")

    return mapping


def build_resolution_map(all_chunk_texts, entity_names, embedder=None):
    """
    Full pipeline: acronym aliases first, then embedding merge on top of
    whatever the acronym pass didn't already canonicalize.
    Returns dict: original_entity_name -> canonical_name
    """
    acronym_map = extract_acronym_aliases(all_chunk_texts)

    # Apply acronym canonicalization first
    after_acronyms = {}
    for name in entity_names:
        after_acronyms[name] = acronym_map.get(name, name)

    # Then embedding-merge the post-acronym unique names
    unique_post_acronym = sorted(set(after_acronyms.values()))
    embed_map = embedding_merge(unique_post_acronym, embedder)

    # Compose both steps into one final mapping
    final_map = {}
    for original, after_acr in after_acronyms.items():
        final_map[original] = embed_map.get(after_acr, after_acr)

    return final_map
