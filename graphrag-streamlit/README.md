# GraphRAG Literature Explorer (lightweight edition)

A GraphRAG-style retrieval pipeline over your own PDFs, built to actually run
on a laptop: PDFs → chunks → entity/relation graph → HyDE-enhanced retrieval,
all through a Streamlit UI.

## Why this won't hang your laptop

The real Microsoft `graphrag` package is heavy because it makes an LLM call
per chunk for extraction **and then more LLM calls per community for
hierarchical summarization** -- multiple full passes over the corpus. This
build only does one pass, and by default that pass is **fully offline**:

- **Entity/relation extraction**: keyword + regex heuristics, no model.
- **Embeddings**: TF-IDF (`scikit-learn`), no downloaded model, no GPU.
- **HyDE**: offline mode approximates it by pulling in matching graph-entity
  vocabulary instead of generating free text.
- **Community detection**: `networkx`'s greedy modularity, runs in-process.

On the 17-PDF sample set this builds the whole graph in ~0.1s and answers a
query in ~2ms.

FastEmbed is opt-in because local model initialization can consume substantial
CPU, memory, disk, and network resources. Set `GRAPHRAG_USE_FASTEMBED=1` before
launching only when you want local embeddings; the default uses TF-IDF.

Hosted LLM calls are limited to 20 per process by default so an online build
cannot accidentally issue thousands of requests. Override with
`GRAPHRAG_LLM_CALL_LIMIT`, or set it to `0` only for a deliberately small test
corpus.

## Optional: better extraction quality via your HF key

If you want richer entity extraction and real HyDE generation, flip the
sidebar to **Hugging Face** and paste your token (or `export HF_TOKEN=...`
before launching). This calls HF's hosted inference router
(`https://router.huggingface.co/v1/chat/completions`) -- **no model is
downloaded or run on your machine**, it's a plain HTTPS request per chunk.
That's the deliberate design: better quality should cost you a network
round trip, never local compute.

```bash
export HF_TOKEN="hf_..."
# optional -- pick a different hosted model:
export HF_MODEL="Qwen/Qwen2.5-7B-Instruct:fastest"
```

Anthropic works the same way if you set `ANTHROPIC_API_KEY` instead.

## Setup

```bash
pip install -r requirements.txt
python3 generate_sample_pdfs.py   # only needed once, to populate database/ with demo PDFs
streamlit run app.py
```

For a bounded online test, select Hugging Face in the sidebar and launch with:

```powershell
$env:GRAPHRAG_LLM_CALL_LIMIT = "20"
$env:GRAPHRAG_USE_FASTEMBED = "0"
python -m streamlit run app.py
```

Replace the files in `database/` with your own PDFs any time -- the
pipeline only assumes "some PDFs live in `database/`".

## Project layout

```
app.py                     Streamlit UI
generate_sample_pdfs.py    one-off script that makes demo PDFs
database/                  put your PDFs here
graphrag/
  ingest.py                PDF loading + chunking
  entity_extraction.py     per-chunk entity/relation extraction
  graph_builder.py         aggregates into a networkx graph + communities
  embeddings.py            TF-IDF chunk index (swap this file for real embeddings later)
  hyde.py                  hypothetical-document query expansion
  retriever.py             combines vector similarity + graph boost -> top-k
  pipeline.py              wires the above together + disk caching
  llm_backend.py           pluggable Anthropic / HF / offline backend
.cache/                    build cache (graph_index.pkl) + generated graph HTML view
```

## What to swap first if you outgrow this

The one genuinely lightweight-but-limited piece is TF-IDF embeddings in
`embeddings.py` -- it has no notion of synonyms. If retrieval quality
becomes the bottleneck rather than speed, that's the file to replace with
real embeddings (e.g. an HF-hosted embedding model via feature-extraction,
or a local `sentence-transformers` model if you have the RAM for it). The
rest of the pipeline only depends on `vectorize()`/`similarity_scores()`,
so the swap is localized to that one file.
