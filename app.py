import os
import sys
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # reads .env in the working directory into os.environ, if present

sys.path.insert(0, os.path.dirname(__file__))
from graphrag import pipeline, retriever, llm_backend
import slr
st.set_page_config(page_title="GraphRAG Literature Explorer", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar: backend selection + index build
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Backend")
### MAKES A SIDEBAR TO SELECT MODEL
backend_choice = st.sidebar.radio(
    "Reasoning backend for entity extraction & HyDE",
    ["Offline (fast, free, runs locally)", "Hugging Face (hosted, needs HF_TOKEN)",
     "Anthropic (hosted, needs ANTHROPIC_API_KEY)"],
    index=0,
    help="Offline mode never makes a network call and never loads a local model -- "
         "it's the lightest option and is on by default. The hosted options give "
         "richer entity/relation extraction but add a network round trip per chunk.",
)#allows user to choose between a local extraction,Hugging face or Anthropic. Default is local model.

### meant to acess and check hugging face
if backend_choice.startswith("Hugging Face"):
    hf_token_input = st.sidebar.text_input("HF_TOKEN", type="password",
                                           value=os.environ.get("HF_TOKEN", ""),
                                           help="Auto-filled from .env if HF_TOKEN is set there.") # a text sidebar if user choose hugging face
    if hf_token_input:
        os.environ["HF_TOKEN"] = hf_token_input #upon entering, it will then set the environment variable
    os.environ.pop("ANTHROPIC_API_KEY", None) #to avoid conflict between the two llm env vars
    st.sidebar.caption(f"Model: `{os.environ.get('HF_MODEL', llm_backend.HF_MODEL_DEFAULT)}` "
                       f"(set HF_MODEL env var to change)") #to show selected model

    ## the part below create a button . its meant to check if hugging face works
    if st.sidebar.button("Test HF connection"):
        try:
            reply = llm_backend.call_llm("Say 'ok' and nothing else.", max_tokens=10)
            st.sidebar.success(f"HF router responded: {reply!r}")
        except Exception as e:
            st.sidebar.error(f"HF call failed: {e}")

        ##    same as the upper one done with hugging face but instead for anthropic
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
    ### if user choose local modle then we dont need the other two en variables
else:
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("HF_TOKEN", None)

#### meant to set up details on the side bar
st.sidebar.divider()
st.sidebar.header("📚 Knowledge Graph")

database_dir = st.sidebar.text_input("PDF folder", value="database")

if "bundle" not in st.session_state: #session state refers to the pythons cript being run
    st.session_state.bundle = pipeline.load_cached() #laod a prevoius graph isntead of rebuildign one

build_clicked = st.sidebar.button("🔨 Build / Rebuild Knowledge Graph", type="primary")
#if clicked
if build_clicked:
    ###related to implementign a progress bar
    progress_bar = st.sidebar.progress(0, text="Starting...")

    def _progress(done, total):
        progress_bar.progress(done / total, text=f"Extracting entities: chunk {done}/{total}")

    t0 = time.time() # is used to find buidl time
    with st.spinner("Ingesting PDFs and building graph..."): #creates a loading idnicator whiel work is doen in bkgrnd
        try:
            st.session_state.bundle = pipeline.build(database_dir, progress_callback=_progress) #refer this. seems to do lots of work
            st.sidebar.success(f"Built in {time.time() - t0:.2f}s")
        except Exception as e:
            st.sidebar.error(f"Build failed: {e}")

bundle = st.session_state.bundle #thus this will now store the knowldge graph if it has eben built ocne before

if bundle: #if the knowldge  graph exists then display details fo that graph
    G = bundle["graph"]
    n_communities = len(set(bundle["communities"].values()))
    st.sidebar.metric("PDFs indexed", len(bundle["pdf_files"]))
    st.sidebar.metric("Chunks", len(bundle["chunks"]))
    st.sidebar.metric("Graph nodes / edges", f"{G.number_of_nodes()} / {G.number_of_edges()}")
    st.sidebar.metric("Communities", n_communities)
else:
    st.sidebar.info("No index yet -- click **Build Knowledge Graph** to get started.")

# ---------------------------------------------------------------------------
# Main: query
# ---------------------------------------------------------------------------
st.title("🔎 GraphRAG Literature Explorer")
st.caption("PDFs → knowledge graph → HyDE-enhanced retrieval, entirely lightweight by default.")
#related to knowldge rgaph
if not bundle:
    st.warning("Build the knowledge graph from the sidebar first.")
    st.stop()
###the search area
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

#The processing part once search is clicked.
if search_clicked and query.strip():
    t0 = time.time() #to find retrieval time
    result = retriever.retrieve(
        query, bundle["chunks"], bundle["index"], bundle["graph"], bundle["communities"],
        top_k=int(top_k), use_hyde=use_hyde,
    )# the retrieval engine
    elapsed = time.time() - t0
    st.session_state.search_result = result

    with st.expander(f"HyDE hypothetical passage (source: {result['hyde_source']})", expanded=False):
        st.write(result["hyde_text"])
#each paper will be shown the hypothetical apsseg amde with HyDE
    st.caption(f"Retrieved {len(result['results'])} results in {elapsed:.3f}s "
              f"· backend: `{bundle['index'].backend}` "
              f"· {len(result['matched_entities'])} query entities matched in graph")
# below section expands each apepr, and scores them based on certian metrics. its related to ranking
    if result["matched_entities"]: 
        with st.expander("Graph traversal detail", expanded=False):
            st.write("**Entities matched in query:**", result["matched_entities"])
            top_entity_scores = sorted(
                result["hop_entity_scores"].items(), key=lambda x: x[1], reverse=True
            )[:15]
            if top_entity_scores:
                st.write("**Top traversal scores (entity → score):**")
                for ent, sc in top_entity_scores: #this displays the scores
                    st.write(f"  `{ent}` → {sc:.3f}")

    for r in result["results"]: #dispalys the actual results ie the resarch apeprs
        chunk = r["chunk"]
        with st.container(border=True):
            st.markdown(f"**{chunk.source_file}** — page {chunk.page}  "
                       f"·  score `{r['final_score']:.3f}` "
                       f"(vector `{r['vector_score']:.3f}` + graph boost `{r['graph_boost']:.3f}`)")
            st.write(chunk.text)

elif search_clicked:
    st.warning("Enter a question first.")

# ---------------------------------------------------------------------------
# Literature Review
# ---------------------------------------------------------------------------

if "search_result" in st.session_state:
    result = st.session_state.search_result

    st.divider()
    st.subheader("📚 Literature Review")

    generate_review_clicked = st.button(
        "Generate Literature Review",
        type="primary"
    )

    if generate_review_clicked:
        with st.spinner(
            "Analyzing papers, comparing findings, and writing review..."
        ):
            try:
                review = slr.generate_review(
                    result["results"],
                    bundle["chunks"]
                )

                st.write(review)

            except Exception as e:
                st.error(
                    f"Literature review generation failed: {e}"
                )

# ---------------------------------------------------------------------------
# Optional graph view (off by default -- rendering a big graph is the one
# thing here that can feel heavy in-browser, so it's opt-in).
# ---------------------------------------------------------------------------

### all realted to dispalyign knowldge graph after search. 
st.divider()
if st.checkbox("Show knowledge graph visualization (top connected entities only)", value=False):
    from pyvis.network import Network # a sepcial moduel to create graphs
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