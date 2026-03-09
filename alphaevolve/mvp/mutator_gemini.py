from __future__ import annotations

import os
import re
from pathlib import Path

from google import genai

from mvp.types import Candidate, DiffProposal, FeedbackBundle


BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\n.*?\n=======\n.*?\n>>>>>>> REPLACE",
    re.DOTALL,
)


def _load_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    dotenv_path = Path(__file__).resolve().parents[1] / ".env"
    if not dotenv_path.exists():
        return None

    for line in dotenv_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith("GEMINI_API_KEY="):
            continue
        value = stripped.split("=", 1)[1].strip().strip("'\"")
        if value:
            return value
    return None


def _extract_search_replace_only(text: str) -> str:
    blocks = [m.group(0) for m in BLOCK_RE.finditer(text)]
    if blocks:
        return "\n\n".join(blocks).strip()
    return text.strip()


class GeminiMutator:
    """Gemini-powered mutator using local SDK patterns from gemini_examples."""

    def __init__(self, model_name: str = "gemini-3-flash-lite") -> None:
        api_key = _load_api_key()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Use --mode mock or export the API key.")
        os.environ["GEMINI_API_KEY"] = api_key
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    async def mutate(
        self,
        parent: Candidate,
        inspirations: list[Candidate],
        feedback: FeedbackBundle,
        prompt: str,
    ) -> DiffProposal:
        aio_client = getattr(self.client, "aio", None)
        if aio_client is not None:
            response = await aio_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
        else:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

        raw_text = (response.text or "").strip()
        if not raw_text:
            raw_text = "".join(
                getattr(part, "text", "")
                for candidate in (response.candidates or [])
                for part in getattr(getattr(candidate, "content", None), "parts", [])
            ).strip()

        if not raw_text:
            raise RuntimeError("Gemini returned an empty mutation response.")

        raw_text = _extract_search_replace_only(raw_text)
        if "<<<<<<< SEARCH" not in raw_text or ">>>>>>> REPLACE" not in raw_text:
            raise RuntimeError("Gemini response did not include valid SEARCH/REPLACE blocks.")

        return DiffProposal(
            raw_diff=raw_text,
            model=self.model_name,
            metadata={
                "parent_id": parent.id,
                "inspiration_ids": [c.id for c in inspirations],
                "feedback_failure_count": len(feedback.weak_failure_reasons),
                "prompt_chars": len(prompt),
            },
        )
