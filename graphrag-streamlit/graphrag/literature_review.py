"""Literature-review synthesis over validated claims and evidence."""
import os

from .llm_backend import llm_available, reason

REVIEW_MODEL_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"
REVIEW_SYSTEM_PROMPT = (
    "You write a concise evidence-grounded literature synthesis. Use only the "
    "supplied landscape, validated gaps, evidence chunks, and KG relationships. "
    "Organize by research evolution, methodological clusters, consistent findings, "
    "contradictions, limitations, and validated gaps. Cite paper IDs in brackets. "
    "Do not claim that an issue is universally unsolved; say when no evidence was "
    "found within the current corpus."
)
REVIEW_CRITIQUE_PROMPT = (
    "Review the draft below against the supplied evidence. Identify unsupported "
    "claims, missing paper citations, overstatements, and weaknesses in the "
    "comparison of methods, findings, limitations, and gaps. Return concise, "
    "actionable corrections only."
)
REVIEW_REFINE_PROMPT = (
    "Rewrite the draft using the critique. Preserve accurate details, remove "
    "unsupported claims, improve comparison across papers, and cite paper IDs "
    "in brackets. Return only the final literature review."
)
REVIEW_MERGE_PROMPT = (
    "Merge the recursive evidence summaries below into one literature review. "
    "Resolve duplicated points, preserve disagreements, and cite only the supplied "
    "paper IDs in brackets. Do not add facts absent from the summaries."
)
REVIEW_LEAF_SIZE = 3
REVIEW_MAX_DEPTH = 3


def _offline_review(query, landscape, gaps):
    lines = [f"Research question: {query}", "", "Evidence-grounded synthesis"]
    papers = sorted({row["paper"] for row in landscape})
    if papers:
        lines.append(f"The current corpus contains evidence from {len(papers)} papers: {', '.join(papers)}.")
    for label, types in (("Methods and findings", {"Gap"}), ("Limitations and gaps", {"Limitation", "Contradiction"})):
        rows = [row for row in landscape if row["claim_type"] in types]
        if rows:
            lines.append(f"\n{label}:")
            for row in rows[:8]:
                lines.append(f"- {row['claim']} [{row['paper']}]")
    if gaps:
        lines.append("\nValidated gap candidates:")
        for gap in gaps[:8]:
            status = gap.get("validation_status", "not validated")
            lines.append(f"- {gap['gap_id']} ({status}): {gap['claim']}")
    return "\n".join(lines)


def _recursive_synthesis(query, landscape, gaps, evidence, model, depth=0):
    """Recursively reduce evidence groups before the final review pass."""
    if len(evidence) <= REVIEW_LEAF_SIZE or depth >= REVIEW_MAX_DEPTH:
        payload = {
            "query": query,
            "landscape": landscape,
            "validated_gaps": gaps,
            "key_evidence": evidence,
        }
        return reason(
            str(payload), system=REVIEW_SYSTEM_PROMPT, max_tokens=1000,
            model=model, offline_fn=lambda: "",
        )

    midpoint = len(evidence) // 2
    left, left_source = _recursive_synthesis(
        query, landscape, gaps, evidence[:midpoint], model, depth + 1
    )
    right, right_source = _recursive_synthesis(
        query, landscape, gaps, evidence[midpoint:], model, depth + 1
    )
    if left_source != "llm" or right_source != "llm":
        return "", "offline"

    merged, source = reason(
        f"Research question: {query}\n\nRecursive summary A:\n{left}\n\n"
        f"Recursive summary B:\n{right}",
        system=REVIEW_MERGE_PROMPT, max_tokens=1100, model=model,
        offline_fn=lambda: "",
    )
    return merged, source


def generate(query, landscape, gaps, evidence):
    model = os.environ.get("HF_REVIEW_MODEL", os.environ.get("HF_MODEL", REVIEW_MODEL_DEFAULT))
    offline_review = lambda: _offline_review(query, landscape, gaps)
    if not llm_available():
        return offline_review().strip(), "offline"

    draft, source = _recursive_synthesis(
        query, landscape, gaps, evidence, model,
    )

    # Use two additional bounded calls when an LLM is available: critique the
    # draft, then revise it while keeping the original evidence in context.
    if source != "llm":
        return (draft or offline_review()).strip(), source

    critique, critique_source = reason(
        f"Evidence payload:\n{evidence}\n\nDraft:\n{draft}",
        system=REVIEW_CRITIQUE_PROMPT, max_tokens=700, model=model,
        offline_fn=lambda: "No additional critique available.",
    )
    if critique_source != "llm":
        return (draft or offline_review()).strip(), source

    refined, refine_source = reason(
        f"Evidence payload:\n{evidence}\n\nDraft:\n{draft}\n\nCritique:\n{critique}",
        system=REVIEW_REFINE_PROMPT, max_tokens=1400, model=model,
        offline_fn=lambda: draft or offline_review(),
    )
    if refine_source != "llm":
        return (draft or offline_review()).strip(), source
    return (refined or draft or offline_review()).strip(), "llm"