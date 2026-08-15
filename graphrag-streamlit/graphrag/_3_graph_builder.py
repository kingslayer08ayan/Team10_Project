"""
Stage 3: Graph construction.
Aggregates entity/relation triples extracted per chunk into one networkx
graph. Each node keeps every chunk_id that mentioned it (traceability);
each edge keeps a weight (co-mention count) and the chunk_ids that support
it. Runs greedy-modularity community detection so retrieval can later
pull in "everything in this paper's neighborhood," which is the part that
makes this GraphRAG rather than plain chunk-similarity RAG.
"""
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from . import _2_entity_extraction as entity_extraction


def build_graph(chunks, progress_callback=None):
    """
    chunks: list of ingest.Chunk
    Returns: (networkx.Graph, dict node_name -> community_id)
    """
    G = nx.Graph()

    for i, chunk in enumerate(chunks):
        result = entity_extraction.extract(chunk.text)

        for ent in result.get("entities", []):
            name = ent["name"]
            if G.has_node(name):
                G.nodes[name]["mentions"] += 1
                G.nodes[name]["chunk_ids"].add(chunk.chunk_id)
                G.nodes[name]["doc_ids"].add(chunk.doc_id)
            else:
                G.add_node(name, type=ent.get("type", "CONCEPT"), mentions=1,
                          chunk_ids={chunk.chunk_id}, doc_ids={chunk.doc_id})

        for rel in result.get("relations", []):
            src, tgt = rel["source"], rel["target"]
            if src == tgt or not G.has_node(src) or not G.has_node(tgt):
                continue
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += 1
                G[src][tgt]["chunk_ids"].add(chunk.chunk_id)
            else:
                G.add_edge(src, tgt, weight=1, relation=rel.get("relation", "related_to"),
                          chunk_ids={chunk.chunk_id})

        if progress_callback:
            progress_callback(i + 1, len(chunks))

    # Community detection over the largest connected structure; isolated
    # nodes each become their own singleton community.
    communities = {}
    if G.number_of_edges() > 0:
        comms = list(greedy_modularity_communities(G, weight="weight"))
        for cid, members in enumerate(comms):
            for m in members:
                communities[m] = cid
    for n in G.nodes:
        communities.setdefault(n, -1)

    return G, communities


def graph_stats(G, communities):
    n_communities = len(set(c for c in communities.values() if c != -1))
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_communities": n_communities,
        "top_connected_entities": top_nodes,
    }
