from __future__ import annotations

import os
import re
from pathlib import Path

from openai import AsyncOpenAI

from mvp.types import Candidate, DiffProposal, FeedbackBundle


BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\n.*?\n=======\n.*?\n>>>>>>> REPLACE",
    re.DOTALL,
)


def _load_api_key() -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    dotenv_path = Path(__file__).resolve().parents[1] / ".env"
    if not dotenv_path.exists():
        return None

    for line in dotenv_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith("OPENAI_API_KEY="):
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


def _response_text(resp) -> str:
    # Preferred path in new OpenAI SDK.
    output_text = getattr(resp, "output_text", None)
    if output_text:
        return str(output_text)

    # Fallback parse for structured response payloads.
    pieces: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content_item in getattr(item, "content", []) or []:
            text_obj = getattr(content_item, "text", None)
            if isinstance(text_obj, str):
                pieces.append(text_obj)
            else:
                value = getattr(text_obj, "value", None)
                if isinstance(value, str):
                    pieces.append(value)
    return "\n".join(pieces).strip()


def _candidate_fallback_models(model_name: str) -> list[str]:
    fallbacks: list[str] = []

    # Explicit known fallbacks for common naming mismatches.
    if model_name == "gpt-5.2-mini":
        fallbacks.extend(["gpt-5-mini", "gpt-4.1-mini"])
    elif model_name == "gpt-5.2":
        fallbacks.extend(["gpt-5", "gpt-5-mini"])

    # Generic fallback: remove .2 segment.
    if ".2" in model_name:
        fallbacks.append(model_name.replace(".2", ""))

    # Deduplicate while preserving order and remove self references.
    out: list[str] = []
    seen: set[str] = set()
    for m in fallbacks:
        if m == model_name or m in seen:
            continue
        seen.add(m)
        out.append(m)
    return out


class OpenAIMutator:
    """OpenAI Responses API mutator with fast+slow model scheduling."""

    def __init__(
        self,
        model_name: str = "gpt-5.2",
        fast_model_name: str = "gpt-5.2-mini",
        slow_every: int = 4,
        request_timeout_s: float = 45.0,
        max_retries: int = 1,
        max_output_tokens: int = 1000,
    ) -> None:
        api_key = _load_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Use --mode mock or export the API key.")
        os.environ["OPENAI_API_KEY"] = api_key
        self.slow_model_name = model_name
        self.fast_model_name = fast_model_name
        self.slow_every = max(1, int(slow_every))
        self.request_timeout_s = float(request_timeout_s)
        self.max_retries = max(0, int(max_retries))
        self.max_output_tokens = max(64, int(max_output_tokens))
        # Disable SDK internal retries; handle retries explicitly in this mutator.
        self.client = AsyncOpenAI(
            api_key=api_key,
            timeout=self.request_timeout_s,
            max_retries=0,
        )
        self._call_count = 0

    def _pick_model(self) -> tuple[str, str]:
        """Use fast model most of the time, and slow base model periodically."""
        self._call_count += 1
        use_slow = (self._call_count % self.slow_every) == 0
        if use_slow:
            return self.slow_model_name, "slow"
        if self.fast_model_name and self.fast_model_name != self.slow_model_name:
            return self.fast_model_name, "fast"
        return self.slow_model_name, "slow"

    async def mutate(
        self,
        parent: Candidate,
        inspirations: list[Candidate],
        feedback: FeedbackBundle,
        prompt: str,
    ) -> DiffProposal:
        selected_model, tier = self._pick_model()
        attempted_models: list[str] = []
        response = None
        actual_model = selected_model
        fallback_used = False
        last_error: str | None = None

        for model_try in [selected_model] + _candidate_fallback_models(selected_model):
            attempted_models.append(model_try)
            for attempt in range(self.max_retries + 1):
                retry_note = ""
                if attempt > 0:
                    retry_note = (
                        "\n\nRETRY INSTRUCTION:\n"
                        "Your previous output was empty or invalid.\n"
                        "Return ONLY valid SEARCH/REPLACE blocks.\n"
                        "Do not include markdown fences or explanations.\n"
                    )
                try:
                    output_token_budget = int(self.max_output_tokens * (1 + attempt))
                    response = await self.client.responses.create(
                        model=model_try,
                        input=prompt + retry_note,
                        timeout=self.request_timeout_s,
                        max_output_tokens=output_token_budget,
                        reasoning={"effort": "low"},
                        text={"verbosity": "low"},
                    )
                    incomplete_reason = None
                    incomplete = getattr(response, "incomplete_details", None)
                    if incomplete is not None:
                        if isinstance(incomplete, dict):
                            incomplete_reason = str(incomplete.get("reason") or "")
                        else:
                            incomplete_reason = str(getattr(incomplete, "reason", "") or "")
                    if incomplete_reason == "max_output_tokens":
                        last_error = "OpenAI response incomplete due max_output_tokens."
                        if attempt < self.max_retries:
                            continue
                        break

                    raw_text = _response_text(response).strip()
                    if not raw_text:
                        last_error = "OpenAI returned an empty mutation response."
                        if attempt < self.max_retries:
                            continue
                        break

                    raw_text = _extract_search_replace_only(raw_text)
                    if "<<<<<<< SEARCH" not in raw_text or ">>>>>>> REPLACE" not in raw_text:
                        last_error = "OpenAI response did not include valid SEARCH/REPLACE blocks."
                        if attempt < self.max_retries:
                            continue
                        break

                    actual_model = model_try
                    fallback_used = model_try != selected_model
                    return DiffProposal(
                        raw_diff=raw_text,
                        model=actual_model,
                        metadata={
                            "provider": "openai",
                            "model_tier": tier,
                            "requested_model": selected_model,
                            "selected_model": actual_model,
                            "fallback_used": fallback_used,
                            "attempted_models": attempted_models,
                            "slow_model_name": self.slow_model_name,
                            "fast_model_name": self.fast_model_name,
                            "slow_every": self.slow_every,
                            "request_timeout_s": self.request_timeout_s,
                            "max_retries": self.max_retries,
                            "max_output_tokens": output_token_budget,
                            "call_count": self._call_count,
                            "attempt_index": attempt,
                            "parent_id": parent.id,
                            "inspiration_ids": [c.id for c in inspirations],
                            "feedback_failure_count": len(feedback.weak_failure_reasons),
                            "prompt_chars": len(prompt),
                        },
                    )
                except Exception as exc:
                    msg = str(exc)
                    if "model_not_found" in msg or "does not exist" in msg:
                        last_error = msg
                        break
                    is_timeout = "timed out" in msg.lower() or "timeout" in msg.lower()
                    is_rate_limit = "rate_limit" in msg.lower() or "429" in msg
                    is_connection = "connection error" in msg.lower() or "api connection" in msg.lower()
                    if (is_timeout or is_rate_limit or is_connection) and attempt < self.max_retries:
                        last_error = msg
                        continue
                    raise

        hint = (
            f"OpenAI model lookup/response failed. Tried models: {attempted_models}."
            " Please update --model/--fast-model to available model ids."
        )
        if last_error:
            hint = f"{hint} Last error: {last_error}"
        raise RuntimeError(hint)
