"""LangGraph orchestration for retrieval, gap analysis, and synthesis."""
from . import gap_analysis, literature_review, retriever


def run(query, chunks, index, graph, communities, top_k=5, use_hyde=True,
        generate_review=False):
    state = {
        "query": query,
        "chunks": chunks,
        "index": index,
        "graph": graph,
        "communities": communities,
        "top_k": top_k,
        "use_hyde": use_hyde,
        "generate_review": generate_review,
    }

    def retrieve_node(current):
        current["retrieval"] = retriever.retrieve(
            current["query"], current["chunks"], current["index"],
            current["graph"], current["communities"],
            top_k=current["top_k"], use_hyde=current["use_hyde"],
        )
        return current

    def analyze_node(current):
        current["analysis"] = gap_analysis.analyze(
            current["query"], current["retrieval"], current["graph"]
        )
        return current

    def validate_node(current):
        analysis = current["analysis"]
        analysis["candidates"] = gap_analysis.validate_candidates(
            analysis["candidates"], current["chunks"], current["index"],
            current["graph"], current["communities"],
        )
        analysis["candidates"] = gap_analysis.rerank_candidates(
            analysis["candidates"], current["query"]
        )
        return current

    def landscape_node(current):
        current["landscape"] = gap_analysis.build_landscape(
            current["analysis"]["candidates"]
        )
        return current

    def review_node(current):
        current["review"], current["review_source"] = literature_review.generate(
            current["query"], current["landscape"],
            current["analysis"]["candidates"], current["analysis"].get("evidence", []),
        )
        return current

    try:
        from langgraph.graph import END, StateGraph
        graph_builder = StateGraph(dict)
        graph_builder.add_node("retrieve", retrieve_node)
        graph_builder.add_node("analyze", analyze_node)
        graph_builder.add_node("validate", validate_node)
        graph_builder.add_node("landscape", landscape_node)
        graph_builder.add_node("review", review_node)
        graph_builder.set_entry_point("retrieve")
        graph_builder.add_edge("retrieve", "analyze")
        graph_builder.add_edge("analyze", "validate")
        graph_builder.add_edge("validate", "landscape")
        graph_builder.add_conditional_edges(
            "landscape", lambda current: "review" if current["generate_review"] else END,
            {"review": "review", END: END},
        )
        graph_builder.add_edge("review", END)
        return graph_builder.compile().invoke(state)
    except ImportError:
        retrieve_node(state)
        analyze_node(state)
        validate_node(state)
        landscape_node(state)
        if generate_review:
            review_node(state)
        return state