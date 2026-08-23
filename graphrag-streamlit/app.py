import os
import sys
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads .env in the working directory into os.environ, if present

sys.path.insert(0, os.path.dirname(__file__))
from graphrag import pipeline, retriever, llm_backend

st.set_page_config(page_title="GraphRAG Literature Explorer", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: backend selection + index build
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Backend")

backend_choice = st.sidebar.radio(
    "Reasoning backend for entity extraction & HyDE",
    ["Offline (fast, free, runs locally)", "Hugging Face (hosted, needs HF_TOKEN)",
     "Anthropic (hosted, needs ANTHROPIC_API_KEY)"],
    index=0,
    help="Offline mode never makes a network call and never loads a local model -- "
         "it's the lightest option and is on by default. The hosted options give "
         "richer entity/relation extraction but add a network round trip per chunk.",
)

if backend_choice.startswith("Hugging Face"):
    hf_token_input = st.sidebar.text_input("HF_TOKEN", type="password",
                                           value=os.environ.get("HF_TOKEN", ""),
                                           help="Auto-filled from .env if HF_TOKEN is set there.")
    if hf_token_input:
        os.environ["HF_TOKEN"] = hf_token_input
    os.environ.pop("ANTHROPIC_API_KEY", None)
    st.sidebar.caption(f"Model: `{os.environ.get('HF_MODEL', llm_backend.HF_MODEL_DEFAULT)}` "
                       f"(set HF_MODEL env var to change)")
    if st.sidebar.button("Test HF connection"):
        try:
            reply = llm_backend.call_llm("Say 'ok' and nothing else.", max_tokens=10)
            st.sidebar.success(f"HF router responded: {reply!r}")
        except Exception as e:
            st.sidebar.error(f"HF call failed: {e}")
elif backend_choice.startswith("Anthropic"):
    anthropic_key_input = st.sidebar.text_input("ANTHROPIC_API_KEY", type="password",
                                                value=os.environ.get("ANTHROPIC_API_KEY", ""),
                                                help="Auto-filled from .env if ANTHROPIC_API_KEY is set there.")
    if anthropic_key_input:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key_input
    os.environ.pop("HF_TOKEN", None)
    if st.sidebar.button("Test Anthropic connection"):
        try:
            reply = llm_backend.call_llm("Say 'ok' and nothing else.", max_tokens=10)
            st.sidebar.success(f"Anthropic responded: {reply!r}")
        except Exception as e:
            st.sidebar.error(f"Anthropic call failed: {e}")
else:
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("HF_TOKEN", None)

st.sidebar.divider()
st.sidebar.header("📚 Knowledge Graph")

database_dir = st.sidebar.text_input("PDF folder", value="database")

if "bundle" not in st.session_state:
    st.session_state.bundle = pipeline.load_cached()

build_clicked = st.sidebar.button("🔨 Build / Rebuild Knowledge Graph", type="primary")

if build_clicked:
    progress_bar = st.sidebar.progress(0, text="Starting...")

    def _progress(done, total):
        progress_bar.progress(done / total, text=f"Extracting entities: chunk {done}/{total}")

    t0 = time.time()
    with st.spinner("Ingesting PDFs and building graph..."):
        try:
            st.session_state.bundle = pipeline.build(database_dir, progress_callback=_progress)
            st.sidebar.success(f"Built in {time.time() - t0:.2f}s")
        except Exception as e:
            st.sidebar.error(f"Build failed: {e}")

bundle = st.session_state.bundle

if bundle:
    G = bundle["graph"]
    n_communities = len(set(bundle["communities"].values()))
    resolved_count = sum(1 for _, attrs in G.nodes(data=True) if len(attrs.get("aliases", {1})) > 1)
    st.sidebar.metric("PDFs indexed", len(bundle["pdf_files"]))
    st.sidebar.metric("Chunks", len(bundle["chunks"]))
    st.sidebar.metric("Graph nodes / edges", f"{G.number_of_nodes()} / {G.number_of_edges()}")
    st.sidebar.metric("Communities", n_communities)
    st.sidebar.caption(f"{resolved_count} nodes merged from multiple surface forms "
                       f"(entity resolution)")

    with st.sidebar.expander("Community summaries", expanded=False):
        summaries = bundle.get("community_summaries", {})
        if not summaries:
            st.caption("No multi-member communities yet.")
        for cid, summary in summaries.items():
            st.write(f"**Community {cid}:** {summary}")
else:
    st.sidebar.info("No index yet -- click **Build Knowledge Graph** to get started.")

# ---------------------------------------------------------------------------
# Main: query
# ---------------------------------------------------------------------------
st.title("🔎 GraphRAG Literature Explorer")
st.caption("PDFs → knowledge graph → HyDE-enhanced retrieval, entirely lightweight by default.")

if not bundle:
    st.warning("Build the knowledge graph from the sidebar first.")
    st.stop()

col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    query = st.text_input("Ask a question about the indexed papers",
                          placeholder="e.g. which papers struggle with generalization across institutions?")
with col2:
    top_k = st.number_input("Top-k", min_value=1, max_value=20, value=5)
with col3:
    use_hyde = st.checkbox("Use HyDE", value=True,
                           help="Generate a hypothetical answer first, then search with that "
                                "instead of the raw query -- improves recall for short questions.")

search_clicked = st.button("Search", type="primary")

if search_clicked and query.strip():
    t0 = time.time()
    result = retriever.retrieve(
        query, bundle["chunks"], bundle["index"], bundle["graph"], bundle["communities"],
        top_k=int(top_k), use_hyde=use_hyde,
    )
    elapsed = time.time() - t0

    with st.expander(f"HyDE hypothetical passage (source: {result['hyde_source']})", expanded=False):
        st.write(result["hyde_text"])

    st.caption(f"Retrieved {len(result['results'])} results in {elapsed:.3f}s "
              f"· backend: `{bundle['index'].backend}` "
              f"· {len(result['matched_entities'])} query entities matched in graph")

    if result["matched_entities"]:
        with st.expander("Graph traversal detail", expanded=False):
            st.write("**Entities matched in query:**", result["matched_entities"])
            top_entity_scores = sorted(
                result["hop_entity_scores"].items(), key=lambda x: x[1], reverse=True
            )[:15]
            if top_entity_scores:
                st.write("**Top traversal scores (entity → score):**")
                for ent, sc in top_entity_scores:
                    st.write(f"  `{ent}` → {sc:.3f}")

            st.write("**Relation types on edges from matched entities:**")
            G = bundle["graph"]
            for ent in result["matched_entities"]:
                if ent not in G:
                    continue
                for neighbor in list(G.neighbors(ent))[:8]:
                    rel_types = ", ".join(sorted(G[ent][neighbor].get("relation_types", {"co-occurs_with"})))
                    st.write(f"  `{ent}` —[{rel_types}]→ `{neighbor}` "
                            f"(weight {G[ent][neighbor].get('weight', 1)})")

    for r in result["results"]:
        chunk = r["chunk"]
        with st.container(border=True):
            st.markdown(
                f"**{chunk.source_file}** — *{chunk.canonical_section}* "
                f"({chunk.section_heading or 'untitled'}), p{chunk.page}  "
                f"·  score `{r['final_score']:.3f}` "
                f"(vector `{r['vector_score']:.3f}` "
                f"+ graph `{r['graph_boost']:.3f}` "
                f"+ section prior `{r['section_prior']:.3f}`)"
            )
            st.write(chunk.text)

            parent = bundle.get("sections_by_id", {}).get(chunk.parent_id)
            if parent and len(parent.text) > len(chunk.text):
                with st.expander("Show full section (parent context)", expanded=False):
                    st.write(parent.text)

elif search_clicked:
    st.warning("Enter a question first.")

# ---------------------------------------------------------------------------
# Evaluation panel
# ---------------------------------------------------------------------------
st.divider()
with st.expander("📊 Evaluation metrics", expanded=False):
    from graphrag import evaluate_retrieval as ev

    st.caption("recall@k needs a labeled eval set (query → known-relevant doc_ids) -- "
              "edit EXAMPLE_EVAL_SET in graphrag/evaluate_retrieval.py with real "
              "examples from your corpus before trusting this number.")

    bench_queries = st.text_area(
        "Latency benchmark queries (one per line)",
        value="which papers use CNN for skin lesion classification?\n"
              "what are the limitations around domain generalization?",
        height=80,
    )
    if st.button("Run latency + duplicate-rate benchmark"):
        queries = [q.strip() for q in bench_queries.split("\n") if q.strip()]
        with st.spinner("Running benchmark..."):
            timing = ev.benchmark_latency(
                queries, bundle["chunks"], bundle["index"],
                bundle["graph"], bundle["communities"], top_k=int(top_k),
            )
            # Duplicate rate measured on RAW (pre-dedup) scoring to show
            # how much the dedup step in retriever.py is actually doing.
            sample_result = retriever.retrieve(
                queries[0], bundle["chunks"], bundle["index"],
                bundle["graph"], bundle["communities"], top_k=int(top_k),
            )
            dup_rate = ev.duplicate_rate(sample_result["results"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean latency", f"{timing['mean_ms']} ms")
        c2.metric("p95 latency", f"{timing['p95_ms']} ms")
        c3.metric("Duplicate rate (post-dedup)", f"{dup_rate:.0%}")
        st.caption("Duplicate rate should read 0% -- that's the paper-level "
                  "dedup in retriever.py confirmed working.")

# ---------------------------------------------------------------------------
# Optional graph view (off by default -- rendering a big graph is the one
# thing here that can feel heavy in-browser, so it's opt-in).
# ---------------------------------------------------------------------------
st.divider()
if st.checkbox("Show knowledge graph visualization (top connected entities only)", value=False):
    from pyvis.network import Network
    import streamlit.components.v1 as components

    G = bundle["graph"]
    top_nodes = [n for n, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:40]]
    sub = G.subgraph(top_nodes)

    net = Network(height="500px", width="100%", bgcolor="#ffffff", notebook=False)
    for n, attrs in sub.nodes(data=True):
        net.add_node(n, label=n, title=f"{attrs.get('type','')} · mentions: {attrs.get('mentions',0)}")
    for u, v, attrs in sub.edges(data=True):
        net.add_edge(u, v, value=attrs.get("weight", 1))

    net.save_graph(".cache/graph_view.html")
    with open(".cache/graph_view.html") as f:
        components.html(f.read(), height=520)
