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
MAX_OFFLINE_RELATIONS = 200

CAP_PHRASE_RE = re.compile(r"\b([A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*){0,3})\b")
ACRONYM_RE = re.compile(r"\b([A-Z]{2,6}(?:-\d+)?)\b")

STOPWORD_CAPS = {"The", "This", "A", "An", "It", "We", "In", "For", "On", "Our", "These"}

SYSTEM_PROMPT = (
    "You extract a knowledge graph fragment from one chunk of a research paper. "
    'Return strict JSON: {"entities": [{"name": "...", "type": "METHOD|DATASET|METRIC|PROBLEM|CONCEPT"}], '
    '"relations": [{"source": "...", "target": "...", "relation": "USES|EVALUATED_ON|OUTPERFORMS|'
    'COMPARED_TO|COMBINED_WITH|ADDRESSES_LIMITATION|co-occurs_with"}]}. '
    "Only include entities actually named in the text. Keep entity names short (<=4 words). "
    "Pick the most specific relation type the text actually supports; use co-occurs_with only "
    "when no clearer relationship is stated."
)

# Cue phrases -> relation type, checked in this priority order (first match
# wins) against the chunk's own text. This is a coarse heuristic: the whole
# chunk gets one inferred type applied to all entity pairs found in it,
# rather than per-pair relation extraction -- cheap and offline-safe, but
# imprecise on chunks that describe multiple distinct relationships at
# once. The LLM path (SYSTEM_PROMPT above) does real per-pair typing when
# a backend is configured.
RELATION_CUES = [
    ("OUTPERFORMS", ["outperform", "beats", "exceeds", "surpasses", "better than"]),
    ("ADDRESSES_LIMITATION", ["address the limitation", "overcome", "mitigat", "to overcome"]),
    ("EVALUATED_ON", ["evaluated on", "tested on", "benchmarked on", "trained on", "validated on"]),
    ("COMPARED_TO", ["compared to", "compared with", "versus", " vs.", "baseline"]),
    ("COMBINED_WITH", ["combined with", "integrat", "fusion of", "coupled with", "combines"]),
    ("USES", ["uses", "used ", "based on", "built on", "employs", "leverages", "adopts"]),
]


def _infer_relation_type(chunk_text):
    text_lower = chunk_text.lower()
    for relation_type, cues in RELATION_CUES:
        if any(cue in text_lower for cue in cues):
            return relation_type
    return "co-occurs_with"


def _classify(term):
    if term in METHOD_HINTS:
        return "METHOD"
    if term in DATASET_HINTS:
        return "DATASET"
    if term in METRIC_HINTS:
        return "METRIC"
    return None


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

    entities = [
        {"name": e, "type": entity_type}
        for e in found
        if (entity_type := _classify(e)) is not None
    ]

    # Relations: every pair of entities found in the same chunk, tagged
    # with the strongest relation type the chunk's own text suggests
    # (see RELATION_CUES above) rather than a flat co-occurs_with.
    relation_type = _infer_relation_type(chunk_text)
    typed_names = {e["name"]: e["type"] for e in entities}
    names = list(typed_names)
    relations = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            # Generic capitalized phrases are often author names or example
            # text. Do not create a dense concept-to-concept graph.
            if typed_names[names[i]] == "CONCEPT" and typed_names[names[j]] == "CONCEPT":
                continue
            relations.append({"source": names[i], "target": names[j], "relation": relation_type})
            if len(relations) >= MAX_OFFLINE_RELATIONS:
                return {"entities": entities, "relations": relations}

    return {"entities": entities, "relations": relations}


def extract(chunk_text):
    result, source = reason(
        chunk_text, system=SYSTEM_PROMPT, max_tokens=600,
        offline_fn=lambda: json.dumps(_offline_extract(chunk_text)),
    )
    if source == "offline":
        return json.loads(result)
    try:
        parsed = json.loads(result)
        parsed["entities"] = [
            entity for entity in parsed.get("entities", [])
            if entity.get("type") in {"METHOD", "DATASET", "METRIC", "PROBLEM"}
        ]
        kept_names = {entity["name"] for entity in parsed["entities"]}
        parsed["relations"] = [
            relation for relation in parsed.get("relations", [])
            if relation.get("source") in kept_names
            and relation.get("target") in kept_names
        ]
        return parsed
    except Exception:
        return _offline_extract(chunk_text)
