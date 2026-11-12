"""Hermes' own cost counter, priced for AdaptiqAI's models.

Hermes prices a turn on a custom endpoint from its `/v1/models` rows, but it reads the first nested object with a
price-like key: a row's `adaptiq_capabilities` comes before its `pricing` and holds `input`, the list of what the
model reads, so the read fails (`agent/model_metadata.py` `_extract_pricing`) and no AdaptiqAI turn has a cost.
Hermes asks a route's provider profile for a turn's cost first (`ProviderProfile.get_usage_cost`), and a plugin may
register a profile for its own named route. The provider entry on AdaptiqAI's URL gets one that is Hermes' generic
custom profile in every other way, hidden from the pickers, and prices each request from the row's `pricing`, as
DeepSeek Harness does: USD per token for new input (`prompt`), output with its thinking (`completion`), what is read
from memory (`input_cache_read`) and what is saved to it (`input_cache_write`).
"""
from __future__ import annotations

import dataclasses
import logging
from decimal import Decimal, InvalidOperation

from . import adaptiq_core as core

logger = logging.getLogger(__name__)

PRICES = ("prompt", "completion", "input_cache_read", "input_cache_write")


def prices(row) -> dict[str, Decimal] | None:
    """A `/v1/models` row's USD-per-token prices, or None unless all four are there (core's pricesFromRow)."""
    pricing = row.get("pricing") if isinstance(row, dict) else None
    if not isinstance(pricing, dict):
        return None
    found = {}
    for key in PRICES:
        value = pricing.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return None
        try:
            price = Decimal(str(value).strip())
        except InvalidOperation:
            return None
        if not price.is_finite() or price < 0:
            return None
        found[key] = price
    return found


def request_cost(row, usage) -> Decimal | None:
    """What one request cost in USD at the row's prices, from Hermes' canonical usage (new input apart from what
    was read from and saved to memory); None without prices."""
    rates = prices(row)
    if rates is None:
        return None

    def count(name: str) -> int:
        value = getattr(usage, name, 0)
        return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0

    return (count("input_tokens") * rates["prompt"] + count("output_tokens") * rates["completion"]
            + count("cache_read_tokens") * rates["input_cache_read"]
            + count("cache_write_tokens") * rates["input_cache_write"])


def register(entry_name: str | None) -> bool:
    """Price the turns of the provider entry `entry_name` (the one on AdaptiqAI's URL) in Hermes' own counter.
    A profile another plugin registered for that name is left as it is. True when the entry is priced."""
    if not entry_name:
        return False
    try:
        from agent.usage_pricing import CostResult, format_cost_label
        from providers import get_provider_profile, register_provider
    except ImportError:  # outside Hermes, or a Hermes without provider profiles
        return False
    try:
        generic = get_provider_profile("custom")
        current = get_provider_profile(entry_name)
    except Exception as error:  # noqa: BLE001 - a registry that cannot be read prices nothing
        logger.debug("AdaptiqAI: no provider profiles to price with: %s", error)
        return False
    if generic is None:
        return False
    if getattr(current, "adaptiq_priced", False):
        return True
    if current is not None and current is not generic:
        return False

    class AdaptiqProfile(type(generic)):
        adaptiq_priced = True

        def get_usage_cost(self, model, usage):
            amount = request_cost(core.model_row(model), usage)
            if amount is None:
                return super().get_usage_cost(model, usage)
            return CostResult(amount_usd=amount, status="estimated", source="provider_models_api",
                              label=format_cost_label(amount), pricing_version="adaptiqai-models")

    fields = {field.name: getattr(generic, field.name) for field in dataclasses.fields(generic) if field.init}
    fields.update(name=entry_name, aliases=(f"custom:{entry_name}",), hidden=True)
    try:
        register_provider(AdaptiqProfile(**fields))
    except Exception as error:  # noqa: BLE001 - Hermes keeps its own (unpriced) path
        logger.warning("AdaptiqAI: could not price %s's turns: %s", entry_name, error)
        return False
    return True
