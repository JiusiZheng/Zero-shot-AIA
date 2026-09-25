"""The user-facing classification prompt template and its builder."""

from __future__ import annotations

USER_PROMPT_TEMPLATE = """Your task is to infer the speaker's [ATTRIBUTE] from the audio material.

Allowed output space (strict): [CATEGORIES]

Constraints:

- Output must be exactly one of the allowed outputs.
- Just give the most simplified output. Do not output sentences, explanations, or symbols.
- If the output is not one of the allowed outputs, it is considered INVALID.

Now output the classification result:"""


def build_user_prompt(attribute: str, categories: tuple[str, ...]) -> str:
    return USER_PROMPT_TEMPLATE.replace("[ATTRIBUTE]", attribute).replace(
        "[CATEGORIES]", ", ".join(categories)
    )
