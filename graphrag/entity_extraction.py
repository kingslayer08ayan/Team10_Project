"""
Stage 2: Entity & relation extraction.
For every chunk, extracts (entity, type) pairs and (source, target,
relation) triples. The offline fallback uses a domain hint list (methods,
datasets, metrics) plus generic capitalized-phrase / acronym detection,
and treats entities that co-occur in the same chunk as related. Swap in
the LLM path (already wired) for richer, less domain-specific extraction.
"""
import re
import json
from .llm_backend import reason

METHOD_HINTS = ["CNN", "Vision Transformer", "ViT", "Federated Learning", "FedAvg",
                "GAN", "Random Forest", "self-supervised", "SimCLR", "contrastive",
                "Grad-CAM", "attention", "ensemble", "ResNet", "DenseNet", "EfficientNet",
                "MobileNetV3", "TabNet", "gradient boosted trees", "multi-instance learning",
                "domain adaptation", "Monte Carlo dropout", "deep ensembles",
                "graph neural network", "gradient boosting"]
DATASET_HINTS = ["ISIC", "HAM10000", "ChestX-ray14", "CBIS-DDSM", "EyePACS", "PANDA",
                 "TCGA-GBM", "TCGA-LGG", "BraTS", "CAMELYON16", "MIMIC-IV", "DrugBank",
                 "ChEMBL", "EyeQ", "COVID-CT", "RSNA"]
METRIC_HINTS = ["AUC", "AUROC", "F1", "sensitivity", "specificity", "precision", "recall",
               "quadratic weighted kappa", "quadratic kappa", "accuracy", "calibration error"]

CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*){0,3})\b")
ACRONYM_RE = re.compile(r"\b([A-Z]{2,6}(?:-\d+)?)\b")

STOPWORD_CAPS = {"The", "This", "A", "An", "It", "We", "In", "For", "On", "Our", "These"}

SYSTEM_PROMPT = (
    "You extract a knowledge graph fragment from one chunk of a research paper. "
    'Return strict JSON: {"entities": [{"name": "...", "type": "METHOD|DATASET|METRIC|PROBLEM|CONCEPT"}], '
    '"relations": [{"source": "...", "target": "...", "relation": "short_verb_phrase"}]}. '
    "Only include entities actually named in the text. Keep entity names short (<=4 words)."
)


def _classify(term):
    if term in METHOD_HINTS:
        return "METHOD"
    if term in DATASET_HINTS:
        return "DATASET"
    if term in METRIC_HINTS:
        return "METRIC"
    return "CONCEPT"


def _offline_extract(chunk_text):
    found = set()
    for hint_list in (METHOD_HINTS, DATASET_HINTS, METRIC_HINTS):
        for h in hint_list:
            if h.lower() in chunk_text.lower():
                found.add(h)

    for m in CAP_PHRASE_RE.finditer(chunk_text):
        phrase = m.group(1).strip()
        first_word = phrase.split()[0]
        if first_word in STOPWORD_CAPS or len(phrase) < 3:
            continue
        if len(phrase.split()) >= 2 or ACRONYM_RE.fullmatch(phrase):
            found.add(phrase)

    entities = [{"name": e, "type": _classify(e)} for e in found]

    # Co-occurrence relations: every pair of entities found in the same chunk
    names = [e["name"] for e in entities]
    relations = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            relations.append({"source": names[i], "target": names[j], "relation": "co-occurs_with"})

    return {"entities": entities, "relations": relations}


def extract(chunk_text):
    result, source = reason(
        chunk_text, system=SYSTEM_PROMPT, max_tokens=600,
        offline_fn=lambda: json.dumps(_offline_extract(chunk_text)),
    )
    if source == "offline":
        return json.loads(result)
    try:
        return json.loads(result)
    except Exception:
        return _offline_extract(chunk_text)
