"""
Stage 3: Graph construction.

Two-pass build:
  Pass 1 -- run entity/relation extraction on every chunk, but hold the
            raw (name, type, relations) results rather than adding nodes
            immediately, so entity resolution (Stage 3a) can canonicalize
            names across the WHOLE corpus before any node exists. Doing
            resolution after graph construction would mean re-merging
            nodes and their accumulated edges after the fact; resolving
            first means the graph is built with correct nodes from the
            start.
  Pass 2 -- add nodes/edges using canonical names. Edges carry a real
            relation TYPE now (USES/EVALUATED_ON/OUTPERFORMS/...), not a
            flat co-occurs_with, wherever the extraction stage could
            infer one.

After the graph is built, communities are detected (still one flat level
via greedy modularity) and each community gets a short summary -- LLM if
a backend is configured, otherwise an offline summary built from its
highest-mention entities.
"""
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from . import entity_extraction, entity_resolution
from .llm_backend import reason

MAX_RELATIONS_PER_CHUNK = 250

COMMUNITY_SUMMARY_SYSTEM_PROMPT = (
    "You summarize one cluster of related entities from a research-paper "
    "knowledge graph in exactly one sentence. Say what theme or topic connects "
    "them, in plain language a researcher skimming a report would understand."
)


def _emit_progress(progress_callback, done, total, stage=None, message=None):
    if progress_callback is None:
        return
    try:
        progress_callback(done, total, stage=stage, message=message)
    except TypeError:
        progress_callback(done, total)


def build_graph(chunks, progress_callback=None, embedder=None):
    """
    chunks: list of ingest.Chunk
    embedder: optional fastembed instance, reused for entity-name merge
             (Stage 3a) -- pass None to skip embedding-based resolution
             and rely on acronym-alias resolution only.
    Returns: (networkx.Graph, dict node_name -> community_id, dict
             community_id -> summary_text)
    """
    # --- Pass 1: extract, but don't add to the graph yet ---
    raw_results = []
    for i, chunk in enumerate(chunks):
        result = entity_extraction.extract(chunk.text)
        raw_results.append((chunk, result))
        _emit_progress(
            progress_callback,
            i + 1,
            len(chunks) * 2,
            stage="entity extraction",
            message=f"Extracting entities: chunk {i + 1}/{len(chunks)}",
        )

    filtered_results = []
    for chunk, result in raw_results:
        entities = [
            ent for ent in result.get("entities", [])
            if ent.get("type") in {"METHOD", "DATASET", "METRIC", "PROBLEM"}
        ]
        kept_names = {ent["name"] for ent in entities}
        relations = [
            rel for rel in result.get("relations", [])
            if rel.get("source") in kept_names and rel.get("target") in kept_names
        ][:MAX_RELATIONS_PER_CHUNK]
        filtered_results.append((chunk, {"entities": entities, "relations": relations}))

    raw_results = filtered_results

    # --- Stage 3a: entity resolution across the whole corpus ---
    all_texts = [c.text for c, _ in raw_results]
    all_names = {ent["name"] for _, result in raw_results for ent in result.get("entities", [])}
    resolution_map = entity_resolution.build_resolution_map(all_texts, all_names, embedder=embedder)

    # --- Pass 2: build the graph with canonical names + typed edges ---
    G = nx.Graph()
    for i, (chunk, result) in enumerate(raw_results):
        for ent in result.get("entities", []):
            name = resolution_map.get(ent["name"], ent["name"])
            if G.has_node(name):
                G.nodes[name]["mentions"] += 1
                G.nodes[name]["chunk_ids"].add(chunk.chunk_id)
                G.nodes[name]["doc_ids"].add(chunk.doc_id)
                G.nodes[name]["aliases"].add(ent["name"])
            else:
                G.add_node(name, type=ent["type"], mentions=1,
                          chunk_ids={chunk.chunk_id}, doc_ids={chunk.doc_id},
                          aliases={ent["name"]})

        for rel in result.get("relations", []):
            src = resolution_map.get(rel["source"], rel["source"])
            tgt = resolution_map.get(rel["target"], rel["target"])
            if src == tgt or not G.has_node(src) or not G.has_node(tgt):
                continue
            relation_type = rel.get("relation", "co-occurs_with")
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += 1
                G[src][tgt]["chunk_ids"].add(chunk.chunk_id)
                # A node pair can be described multiple ways across the
                # corpus; keep every distinct relation type seen, not
                # just the first.
                G[src][tgt]["relation_types"].add(relation_type)
            else:
                G.add_edge(src, tgt, weight=1, relation_types={relation_type},
                          chunk_ids={chunk.chunk_id})

        _emit_progress(
            progress_callback,
            len(chunks) + i + 1,
            len(chunks) * 2,
            stage="graph assembly",
            message=f"Building graph: chunk {i + 1}/{len(chunks)}",
        )

    # --- Community detection (still one flat level) ---
    communities = {}
    if G.number_of_edges() > 0:
        comms = list(greedy_modularity_communities(G, weight="weight"))
        for cid, members in enumerate(comms):
            for m in members:
                communities[m] = cid
    for n in G.nodes:
        communities.setdefault(n, -1)

    community_summaries = _summarize_communities(G, communities)

    return G, communities, community_summaries


# ------------------------------------------------------------------
# Community summaries
# ------------------------------------------------------------------

def _offline_community_summary(members_with_attrs):
    top = sorted(members_with_attrs, key=lambda x: x[1]["mentions"], reverse=True)[:6]
    parts = [f"{name} ({attrs.get('type', 'CONCEPT')})" for name, attrs in top]
    return "Centers on: " + ", ".join(parts)


def _llm_community_summary(members_with_attrs):
    top = sorted(members_with_attrs, key=lambda x: x[1]["mentions"], reverse=True)[:10]
    listing = ", ".join(f"{name} ({attrs.get('type', 'CONCEPT')})" for name, attrs in top)
    prompt = f"Entities in this cluster: {listing}"
    text, source = reason(
        prompt, system=COMMUNITY_SUMMARY_SYSTEM_PROMPT, max_tokens=60,
        offline_fn=lambda: _offline_community_summary(members_with_attrs),
    )
    return (text or "").strip() or _offline_community_summary(members_with_attrs)


def _summarize_communities(G, communities):
    """Returns dict: community_id -> one-sentence summary string."""
    by_community = {}
    for name, cid in communities.items():
        if cid == -1:
            continue
        by_community.setdefault(cid, []).append((name, G.nodes[name]))

    summaries = {}
    for cid, members in by_community.items():
        if len(members) < 2:
            continue  # singleton communities aren't worth summarizing
        summaries[cid] = _llm_community_summary(members)

    return summaries


def graph_stats(G, communities):
    n_communities = len(set(c for c in communities.values() if c != -1))
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_communities": n_communities,
        "top_connected_entities": top_nodes,
    }