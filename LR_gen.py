"""
Literature Review Generator.

Takes the research question and the selected research papers,
extracts structured information from each paper, and then uses
an LLM to generate a synthesized literature review.
"""

import os
import json

from dotenv import load_dotenv
from openai import OpenAI

import prompts


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-oss-20b"

if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Add it to your .env file."
    )


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


# ---------------------------------------------------------------------------
# Paper Extraction
# ---------------------------------------------------------------------------

def extract_paper_information(paper_text, paper_id):
    """
    Extract structured information from one research paper.

    Args:
        paper_text:
            Full text/content of the research paper.

        paper_id:
            Identifier assigned to the paper.

    Returns:
        dict:
            Structured information extracted from the paper.
    """

    if not paper_text:
        raise ValueError(
            f"No paper text was provided for paper {paper_id}."
        )

    user_prompt = prompts.build_paper_extraction_prompt(
        paper_text=paper_text,
        paper_id=paper_id,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": prompts.PAPER_EXTRACTION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    extracted_text = response.choices[0].message.content

    if not extracted_text:
        raise ValueError(
            f"Paper extraction returned an empty response for {paper_id}."
        )

    try:
        extracted_data = json.loads(extracted_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Paper extraction returned invalid JSON for {paper_id}."
        ) from e

    return extracted_data


# ---------------------------------------------------------------------------
# Literature Review Generator
# ---------------------------------------------------------------------------

def generate_literature_review(research_question, papers):
    """
    Generate a literature review from the selected research papers.

    The process has two stages:

    1. Extract structured information from each paper.
    2. Generate a synthesized literature review from the extracted
       information.

    Args:
        research_question:
            The research question provided by the user.

        papers:
            The selected papers/evidence provided by GraphRAG.

    Returns:
        str:
            Generated literature review.
    """

    if not research_question or not research_question.strip():
        raise ValueError(
            "No research question was provided."
        )

    if not papers:
        raise ValueError(
            "No papers were provided for the literature review."
        )

    # -----------------------------------------------------------------------
    # Stage 1: Extract structured information from every paper
    # -----------------------------------------------------------------------

    extracted_papers = []

    for index, paper in enumerate(papers, start=1):

        # Each paper is expected to contain its text/content.
        # If GraphRAG does not provide a paper ID, create one.
        if isinstance(paper, dict):
            paper_id = paper.get("paper_id") or paper.get("id") or f"P{index:02d}"
            paper_text = (
                paper.get("paper_text")
                or paper.get("text")
                or paper.get("content")
            )
        else:
            paper_id = f"P{index:02d}"
            paper_text = str(paper)

        if not paper_text:
            raise ValueError(
                f"No text/content found for paper {paper_id}."
            )

        extracted_data = extract_paper_information(
            paper_text=paper_text,
            paper_id=paper_id,
        )

        extracted_papers.append(extracted_data)

    # Convert the structured paper information into text
    # for the literature review prompt.
    extracted_papers_text = json.dumps(
        extracted_papers,
        indent=2,
        ensure_ascii=False,
    )

    # -----------------------------------------------------------------------
    # Stage 2: Generate the literature review
    # -----------------------------------------------------------------------

    user_prompt = prompts.build_literature_review_prompt(
        research_question=research_question,
        papers=extracted_papers_text,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": prompts.LITERATURE_REVIEW_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    literature_review = response.choices[0].message.content

    if not literature_review:
        raise ValueError(
            "Literature review generator returned an empty response."
        )

    return literature_review