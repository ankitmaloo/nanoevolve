"""LLM-driven mutation engine with multi-provider support.

Calls an LLM to generate code diffs based on doctrine-stage prompts.
Supports Anthropic, OpenAI, and Gemini as backends.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from enigma.config import ProviderConfig
from enigma.types import DiffProposal


class MutationError(Exception):
    pass


class LLMMutator:
    """Generate code mutations via LLM API calls."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self.config.provider == "anthropic":
            import anthropic
            api_key = os.environ.get(self.config.api_key_env or "ANTHROPIC_API_KEY")
            self._client = anthropic.Anthropic(api_key=api_key)
        elif self.config.provider == "openai":
            import openai
            api_key = os.environ.get(self.config.api_key_env or "OPENAI_API_KEY")
            kwargs: dict[str, Any] = {"api_key": api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = openai.OpenAI(**kwargs)
        elif self.config.provider == "gemini":
            import google.generativeai as genai
            api_key = os.environ.get(self.config.api_key_env or "GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.config.model_name)
        else:
            raise MutationError(f"Unknown provider: {self.config.provider}")

        return self._client

    def call_llm(self, prompt: str, *, stage: str = "unknown") -> str:
        """Call the LLM and return the raw text response."""
        client = self._get_client()

        if self.config.provider == "anthropic":
            response = client.messages.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        elif self.config.provider == "openai":
            response = client.chat.completions.create(
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

        elif self.config.provider == "gemini":
            response = client.generate_content(
                prompt,
                generation_config={
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                },
            )
            return response.text

        raise MutationError(f"Unsupported provider: {self.config.provider}")

    def call_llm_json(self, prompt: str, *, stage: str = "unknown") -> Any:
        """Call the LLM and parse the response as JSON."""
        raw = self.call_llm(prompt, stage=stage)
        return parse_json_response(raw)

    def generate_diff(self, prompt: str, *, stage: str = "mutation") -> DiffProposal:
        """Generate a code diff from the LLM."""
        raw = self.call_llm(prompt, stage=stage)

        # Extract SEARCH/REPLACE blocks
        blocks = extract_search_replace_blocks(raw)
        if not blocks:
            raise MutationError(
                "LLM response contained no valid SEARCH/REPLACE blocks. "
                f"Response preview: {raw[:200]}"
            )

        return DiffProposal(
            raw_diff=raw,
            model=self.config.model_name,
            stage=stage,
            metadata={"num_blocks": len(blocks)},
        )


def extract_search_replace_blocks(text: str) -> list[tuple[str, str]]:
    """Extract (search, replace) pairs from SEARCH/REPLACE block format."""
    pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def apply_search_replace_blocks(
    source: str,
    diff_text: str,
) -> tuple[str, dict[str, Any]]:
    """Apply SEARCH/REPLACE blocks to source code.

    Returns (modified_source, stats_dict).
    """
    blocks = extract_search_replace_blocks(diff_text)
    if not blocks:
        raise MutationError("No SEARCH/REPLACE blocks found in diff.")

    result = source
    applied = 0
    failed = 0

    for search, replace in blocks:
        if search in result:
            result = result.replace(search, replace, 1)
            applied += 1
        else:
            # Try with stripped whitespace matching
            search_stripped = "\n".join(l.rstrip() for l in search.splitlines())
            result_stripped_lines = result.splitlines()
            found = False

            for i in range(len(result_stripped_lines)):
                candidate_lines = result_stripped_lines[i:i + len(search.splitlines())]
                candidate = "\n".join(l.rstrip() for l in candidate_lines)
                if candidate == search_stripped:
                    # Found with whitespace normalization
                    original_chunk = "\n".join(result_stripped_lines[i:i + len(search.splitlines())])
                    result = result.replace(original_chunk, replace, 1)
                    applied += 1
                    found = True
                    break

            if not found:
                failed += 1

    if applied == 0:
        raise MutationError(
            f"None of {len(blocks)} SEARCH/REPLACE blocks matched. "
            f"First search block preview: {blocks[0][0][:100]}..."
        )

    return result, {"applied": applied, "failed": failed, "total": len(blocks)}


def parse_json_response(text: str) -> Any:
    """Parse JSON from an LLM response that may contain markdown fences."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line if it's ``
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array or object
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start >= 0:
            # Find matching end
            depth = 0
            for i in range(start, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    raise MutationError(f"Could not parse JSON from LLM response: {text[:200]}")
