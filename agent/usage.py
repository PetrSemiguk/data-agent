"""Token usage with prompt caching, and its cost in USD."""

import logging
from typing import Any

from agent.config import MODEL_PRICES

logger = logging.getLogger(__name__)


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


def request_cost(
    model: str, input_tokens: int, cache_write_tokens: int, cache_read_tokens: int, output_tokens: int
) -> float | None:
    """Price of ONE API request, or None if config.py has no prices for the model.

    Priced per request because some models (Claude Haiku 5.5) charge more when
    the request's whole prompt - input + cache write + cache read - is long.
    """
    tiers = MODEL_PRICES.get(model)
    if tiers is None:
        logger.warning("No prices for model %r in config.py; its cost is unknown", model)
        return None
    prompt = input_tokens + cache_write_tokens + cache_read_tokens
    prices = next(p for limit, p in tiers if limit is None or prompt <= limit)
    return (
        input_tokens * prices["input"]
        + cache_write_tokens * prices["cache_write"]
        + cache_read_tokens * prices["cache_read"]
        + output_tokens * prices["output"]
    ) / 1_000_000


def add_costs(*costs: float | None) -> float | None:
    """Sum of costs; unknown (None) if any part is unknown."""
    return None if any(c is None for c in costs) else sum(c for c in costs if c is not None)


def format_cost(cost: float | None, digits: int = 4) -> str:
    return "unknown" if cost is None else f"${cost:.{digits}f}"
