import os
import sys
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads .env in the working directory into os.environ, if present

sys.path.insert(0, os.path.dirname(__file__))
from graphrag import pipeline, retriever, llm_backend, gap_analysis, workflow

st.set_page_config(page_title="GraphRAG Literature Explorer", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --ink: #172126;
    --muted: #637078;
    --paper: #edf1f0;
    --surface: #fbfcfb;
    --line: #cbd5d2;
    --coral: #d65b48;
    --accent-soft: #f7e6e1;
}

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
[data-testid="stAppViewContainer"] {
    background-color: var(--paper);
    background-image:
        linear-gradient(rgba(23, 33, 38, .028) 1px, transparent 1px),
        linear-gradient(90deg, rgba(23, 33, 38, .028) 1px, transparent 1px);
    background-size: 32px 32px;
    color: var(--ink);
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background: linear-gradient(115deg, rgba(255,255,255,.32), transparent 42%);
    z-index: 0;
}
[data-testid="stAppViewContainer"] > .main { position: relative; z-index: 1; }
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"],
[data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
    color: var(--ink);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: #e9eeeb;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child { padding: 2rem 1.35rem; }
[data-testid="stSidebar"] h2 {
    color: var(--ink);
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-top: 1.3rem;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption { color: var(--muted); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--ink); }

section.main > div { max-width: 1280px; padding: 3.2rem 4rem 5rem; }
h1, h2, h3 { color: var(--ink); }
h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700;
    font-size: clamp(2.4rem, 4vw, 4.7rem);
    line-height: .98;
    letter-spacing: 0;
    max-width: 780px;
    margin-bottom: .65rem;
}
h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
h2 { font-size: 1.7rem; font-weight: 700; }
h3 { font-size: 1.05rem; letter-spacing: .01em; }

.console-kicker {
    color: var(--coral);
    font: 600 .7rem 'IBM Plex Mono', monospace;
    letter-spacing: .12em;
    text-transform: uppercase;
    margin-bottom: .9rem;
}
.console-deck {
    color: var(--muted);
    font-size: 1rem;
    max-width: 650px;
    margin-bottom: 2.3rem;
}
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-testid="stNumberInput"] input {
    background: var(--surface);
    border: 1px solid #cbd3d0;
    border-radius: 3px;
    color: var(--ink);
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder { color: #66736f; opacity: 1; }
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus { border-color: var(--coral); box-shadow: 0 0 0 1px var(--coral); }
button[kind="primary"] {
    background: var(--coral) !important;
    border: 1px solid var(--coral) !important;
    border-radius: 3px !important;
    font-weight: 700 !important;
    color: #fff !important;
}
button[kind="primary"] *, button[kind="primary"] p { color: #fff !important; }
button[kind="secondary"] {
    border-radius: 3px !important;
    background: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid #aab8b4 !important;
}
button[kind="secondary"] *, button[kind="secondary"] p { color: var(--ink) !important; }
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 4px;
    box-shadow: 0 5px 18px rgba(23, 33, 38, .045);
}
[data-testid="stMetric"] {
    background: rgba(255, 253, 250, .62);
    border-left: 3px solid var(--coral);
    padding: .55rem .7rem;
}
[data-testid="stMetricLabel"] { color: var(--muted); font-size: .72rem; }
[data-testid="stMetricValue"] { color: var(--ink); font-family: 'IBM Plex Mono', monospace; font-size: 1.25rem; }
[data-baseweb="tab-list"] { gap: 1.2rem; border-bottom: 1px solid var(--line); }
[data-baseweb="tab"] { color: var(--muted); font-weight: 600; padding: .75rem .1rem; }
[aria-selected="true"] { color: var(--coral) !important; }
[data-testid="stExpander"] { border-color: var(--line); border-radius: 3px; background: transparent; }
[data-testid="stExpander"] summary p { color: var(--ink); font-weight: 600; }
code { color: var(--coral); background: var(--accent-soft); border-radius: 2px; }
hr { border-color: var(--line); margin: 2.4rem 0; }
small, [data-testid="stCaptionContainer"] { color: var(--muted); }
section.main > div { animation: workspace-in .55s cubic-bezier(.22, 1, .36, 1); }
button:active { transform: translateY(1px); }
@keyframes workspace-in {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: backend selection + index build
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="console-kicker">GraphRAG / research console</div>', unsafe_allow_html=True)
st.sidebar.header("Backend")

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
st.sidebar.header("Knowledge graph")

database_dir = st.sidebar.text_input("PDF folder", value="database")

if "bundle" not in st.session_state:
    st.session_state.bundle = pipeline.load_cached()

build_clicked = st.sidebar.button("Build / Rebuild Knowledge Graph", type="primary")

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
st.markdown('<div class="console-kicker">Evidence intelligence · local corpus</div>', unsafe_allow_html=True)
st.title("GraphRAG Literature Explorer")
st.markdown('<div class="console-deck">Trace ideas across papers, surface defensible gaps, and turn scattered evidence into a research landscape.</div>', unsafe_allow_html=True)

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
    st.session_state.last_retrieval = result
    st.session_state.last_query = query
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
                f"·  evidence score `{r.get('evidence_score', r['final_score']):.3f}` "
                f"(initial `{r.get('initial_score', r['final_score']):.3f}`) "
                f"(vector `{r['vector_score']:.3f}` "
                f"+ graph `{r['graph_boost']:.3f}` "
                f"+ section prior `{r['section_prior']:.3f}` "
                f"+ evidence type `{r.get('evidence_type_score', 0.0):.3f}` "
                f"+ claim `{r.get('claim_relevance', 0.0):.3f}`)"
            )
            st.write(chunk.text)

            parent = bundle.get("sections_by_id", {}).get(chunk.parent_id)
            if parent and len(parent.text) > len(chunk.text):
                with st.expander("Show full section (parent context)", expanded=False):
                    st.write(parent.text)

elif search_clicked:
    st.warning("Enter a question first.")

if st.session_state.get("last_retrieval"):
    st.divider()
    gap_tab, review_tab = st.tabs(["Gap Analysis", "Literature Review"])
    with gap_tab:
        st.subheader("Gap and limitation analysis")
        st.caption("Analyzes the latest reranked evidence chunks and their graph context.")
        if st.button("Run gap analysis and literature review"):
            with st.spinner("Extracting and aggregating evidence-backed claims..."):
                workflow_state = workflow.run(
                    st.session_state.get("last_query", ""), bundle["chunks"],
                    bundle["index"], bundle["graph"], bundle["communities"],
                    top_k=int(top_k), use_hyde=use_hyde, generate_review=True,
                )
                st.session_state.gap_analysis = workflow_state["analysis"]
                st.session_state.landscape = workflow_state["landscape"]
                st.session_state.literature_review = workflow_state.get("review", "")
                st.session_state.literature_review_source = workflow_state.get("review_source", "offline")

        analysis = st.session_state.get("gap_analysis")
        if analysis:
            st.caption(f"Source: `{analysis['source']}` · "
                       f"{len(analysis['candidates'])} aggregated candidate claims")
            for candidate in analysis["candidates"]:
                with st.container(border=True):
                    st.markdown(
                        f"**{candidate['gap_id']} · {candidate['type']}** · "
                        f"priority `{candidate.get('priority_score', 0.0):.3f}`"
                    )
                    st.write(candidate["claim"])
                    st.caption(
                        f"Papers: {', '.join(candidate['supporting_papers']) or 'none'} · "
                        f"Evidence: {', '.join(candidate['evidence_types']) or 'none'} · "
                        f"Independent papers: {candidate['supporting_paper_count']} · "
                        f"Confidence: {candidate['confidence']:.2f} · "
                        f"Importance: {candidate['importance']:.2f}"
                    )
                    st.caption("Priority factors: " + "; ".join(
                        candidate.get("priority_reasons", ["not ranked"])
                    ))
                    st.caption(f"Validation: {candidate.get('validation_note', 'not run')}")
                    with st.expander("Evidence details", expanded=False):
                        st.write("**Supporting chunks:**", candidate["supporting_chunk_ids"])
                        st.write("**Entities:**", candidate["relevant_entities"])
                        st.write("**KG relationships:**", candidate["kg_relationships"])
                        st.write("**Reasoning:**", candidate["reasoning"])

        if st.session_state.get("landscape"):
            st.subheader("Research landscape matrix")
            st.dataframe(st.session_state.landscape, width="stretch")

    with review_tab:
        if st.session_state.get("literature_review"):
            st.caption(f"Source: `{st.session_state.get('literature_review_source', 'offline')}`")
            st.write(st.session_state.literature_review)
        else:
            st.info("Run the analysis from the Gap Analysis tab to generate a review.")

# ---------------------------------------------------------------------------
# Evaluation panel
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Evaluation metrics", expanded=False):
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
