from graphrag import pipeline,retriever
import slr


print("===== LOADING GRAPH =====")

bundle = pipeline.load_cached()

if not bundle:
    raise RuntimeError(
        "No cached GraphRAG index found. "
        "Build the knowledge graph first."
    )


query = "AI methods for skin lesion classification"


print("\n===== RETRIEVING PAPERS =====")

result = retriever.retrieve(
    query,
    bundle["chunks"],
    bundle["index"],
    bundle["graph"],
    bundle["communities"],
    top_k=5,
    use_hyde=True,
)


print("\n===== RETRIEVED PAPERS =====")

for i, r in enumerate(result["results"], start=1):
    print(
        f"{i}. {r['chunk'].source_file} "
        f"(score: {r['final_score']:.3f})"
    )


print("\n===== RUNNING MULTI-AGENT SLR =====")

review = slr.generate_review(
    results=result["results"],
    all_chunks=bundle["chunks"],
)


print("\n===== FINAL LITERATURE REVIEW =====\n")

print(review)