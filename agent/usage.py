"""Token usage with prompt caching, and its cost in USD."""

from typing import Any

from agent.config import (
    PRICE_CACHE_READ_PER_MTOK,
    PRICE_CACHE_WRITE_PER_MTOK,
    PRICE_INPUT_PER_MTOK,
    PRICE_OUTPUT_PER_MTOK,
)


def usage_counts(usage: Any) -> tuple[int, int, int, int]:
    """(input, cache write, cache read, output) tokens of one API response.

    `input_tokens` is only the uncached part after the last cache breakpoint;
    the whole prompt is input + cache write + cache read. The cache fields may
    be missing or None when nothing was cached.
    """
    return (
        usage.input_tokens or 0,
        getattr(usage, "cache_creation_input_tokens", 0) or 0,
        getattr(usage, "cache_read_input_tokens", 0) or 0,
        usage.output_tokens or 0,
    )


def cost_usd(input_tokens: int, cache_write_tokens: int, cache_read_tokens: int, output_tokens: int) -> float:
    """Price of the tokens; each kind has its own rate (see config.py)."""
    return (
        input_tokens * PRICE_INPUT_PER_MTOK
        + cache_write_tokens * PRICE_CACHE_WRITE_PER_MTOK
        + cache_read_tokens * PRICE_CACHE_READ_PER_MTOK
        + output_tokens * PRICE_OUTPUT_PER_MTOK
    ) / 1_000_000
