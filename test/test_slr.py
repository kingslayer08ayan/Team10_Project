from graphrag import pipeline, retriever
import slr


# Load the existing GraphRAG index
bundle = pipeline.load_cached()

if not bundle:
    raise RuntimeError(
        "No cached GraphRAG index found. "
        "Run app.py and build the knowledge graph first."
    )


# Change this to whatever you want to test
query = "skin lesion classification"


# Ask the existing GraphRAG retriever for the top 5 papers
result = retriever.retrieve(
    query,
    bundle["chunks"],
    bundle["index"],
    bundle["graph"],
    bundle["communities"],
    top_k=5,
    use_hyde=True,
)


print("\n===== RETRIEVED PAPERS =====\n")

for i, r in enumerate(result["results"], start=1):
    chunk = r["chunk"]

    print(
        f"{i}. {chunk.source_file} "
        f"(score: {r['final_score']:.3f})"
    )


print("\n===== GENERATING LITERATURE REVIEW =====\n")

review = slr.generate_review(
    result["results"],
    bundle["chunks"],
)


print(review)