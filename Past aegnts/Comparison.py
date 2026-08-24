"""
Comparison Agent.

Takes structured analyses of multiple research papers and identifies
meaningful relationships between them.
"""

import json
import os

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
# Agent
# ---------------------------------------------------------------------------

def compare(analyses):
    """
    Compare multiple paper analyses.

    Args:
        analyses:
            List of dictionaries produced by Paper_Analysis.analyze().

    Returns:
        dict:
            Structured comparison of the papers.
    """

    if not analyses:
        raise ValueError(
            "No paper analyses were provided."
        )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": prompts.COMPARISON_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompts.build_comparison_prompt(
                    analyses
                ),
            },
        ],
    )

    content = response.choices[0].message.content

    try:
        comparison = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(
            f"Comparison agent returned invalid JSON:\n{content}"
        )

    return comparison