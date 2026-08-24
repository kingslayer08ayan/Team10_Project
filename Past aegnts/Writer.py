"""
Writer Agent.

Takes individual paper analyses and their comparison, then writes
the final literature review.
"""

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

def write(analyses, comparison):
    """
    Write the final literature review.

    Args:
        analyses:
            Structured analyses produced by Paper_Analysis.

        comparison:
            Structured comparison produced by Comparison.

    Returns:
        str:
            Final literature review.
    """

    if not analyses:
        raise ValueError(
            "No paper analyses were provided."
        )

    if not comparison:
        raise ValueError(
            "No comparison was provided."
        )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": prompts.WRITER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompts.build_writer_prompt(
                    analyses,
                    comparison,
                ),
            },
        ],
    )

    return response.choices[0].message.content