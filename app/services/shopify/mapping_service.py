"""Shopify options <-> Rivollo configuration dimensions (ADR-017).

The mapping lives on the Shopify product (``tbl_shopify_products.dimension_mapping``),
keyed by the dimensions' stable codes:

    {"capacity": {"option_name": "Capacity", "values": {"3_seater": "3 Seater"}},
     "layout":   {"option_name": "Layout",   "values": {"corner": "Corner"}}}

It is matched EXACTLY against the synced option names and values: no fuzzy
matching, so a publish never depends on a guess. The configurator calls
``validate`` inside its PUT transaction; the public Shopify payload calls
``variant_models`` to tell the viewer which model each Shopify variant shows.

Takes plain data (codes and dicts), not configurator objects, so this module
and the configurator service do not import each other.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.database.shopify_repo import shopify_repository as repo
from app.models.shopify import ShopifyProduct
from app.services.configurator.configuration_errors import invalid


async def linked_product(
    db: AsyncSession, product_id: uuid.UUID, user_id: uuid.UUID
) -> Optional[ShopifyProduct]:
    """The Shopify product linked to this Rivollo product, or None (also with the flag off)."""
    if not settings.ENABLE_SHOPIFY_INTEGRATION:
        return None
    return await repo.get_linked_product(db, product_id, user_id)


def _shopify_options(product: ShopifyProduct) -> dict[str, list[str]]:
    return {o["name"]: list(o.get("values") or []) for o in product.options or []}


def _variant_options(variant) -> dict[str, str]:
    return {o["name"]: o["value"] for o in variant.options or []}


def selection_for(mapping: dict[str, Any], options: dict[str, str]) -> Optional[dict[str, str]]:
    """A Shopify variant's options -> {dimension code: value code}; None if any is unmapped."""
    selection: dict[str, str] = {}
    for code, entry in mapping.items():
        shopify_value = options.get(entry["option_name"])
        reverse = {v: k for k, v in (entry.get("values") or {}).items()}
        if shopify_value is None or shopify_value not in reverse:
            return None
        selection[code] = reverse[shopify_value]
    return selection


def validate(
    mapping: dict[str, Any],
    *,
    dimensions: dict[str, dict[str, str]],
    used_values: dict[str, set[str]],
    default_selection: dict[str, str],
    product: ShopifyProduct,
) -> None:
    """Raise a structured 400 unless ``mapping`` is complete, exact and unambiguous.

    ``dimensions``: {dimension code: {value code: label}} of the new configuration.
    ``used_values``: {dimension code: value codes some model uses}.
    """
    options = _shopify_options(product)
    seen_options: dict[str, str] = {}

    for code in mapping:
        if code not in dimensions:
            raise invalid(
                "SHOPIFY_UNKNOWN_DIMENSION",
                f"The Shopify mapping names dimension '{code}', which this configuration does not have.",
                [f"shopify_mapping.{code}"],
            )
    for code in dimensions:
        if code not in mapping:
            raise invalid(
                "SHOPIFY_DIMENSION_UNMAPPED",
                f"Dimension '{code}' is not mapped to a Shopify option.",
                [f"shopify_mapping.{code}"],
            )

    for code, entry in mapping.items():
        option_name = entry["option_name"]
        path = f"shopify_mapping.{code}"
        if option_name not in options:
            raise invalid(
                "SHOPIFY_OPTION_NOT_FOUND",
                f"The synced Shopify product has no option '{option_name}'.",
                [f"{path}.option_name"],
            )
        if option_name in seen_options:
            raise invalid(
                "SHOPIFY_MAPPING_AMBIGUOUS",
                f"Dimensions '{seen_options[option_name]}' and '{code}' are both mapped to "
                f"Shopify option '{option_name}'.",
                [f"shopify_mapping.{seen_options[option_name]}.option_name", f"{path}.option_name"],
            )
        seen_options[option_name] = code

        shopify_values = set(options[option_name])
        values: dict[str, str] = entry.get("values") or {}
        claimed: dict[str, str] = {}
        for value_code, shopify_value in values.items():
            if value_code not in dimensions[code]:
                raise invalid(
                    "SHOPIFY_UNKNOWN_VALUE",
                    f"'{value_code}' is not a value of dimension '{code}'.",
                    [f"{path}.values.{value_code}"],
                )
            if shopify_value not in shopify_values:
                raise invalid(
                    "SHOPIFY_VALUE_NOT_FOUND",
                    f"Shopify option '{option_name}' has no value '{shopify_value}'.",
                    [f"{path}.values.{value_code}"],
                )
            if shopify_value in claimed:
                raise invalid(
                    "SHOPIFY_MAPPING_AMBIGUOUS",
                    f"Values '{claimed[shopify_value]}' and '{value_code}' are both mapped to "
                    f"'{option_name}={shopify_value}'.",
                    [f"{path}.values.{claimed[shopify_value]}", f"{path}.values.{value_code}"],
                )
            claimed[shopify_value] = value_code
        for value_code in sorted(used_values.get(code, set())):
            if value_code not in values:
                raise invalid(
                    "SHOPIFY_VALUE_UNMAPPED",
                    f"'{dimensions[code][value_code]}' is used by a model but not mapped to a "
                    f"value of Shopify option '{option_name}'.",
                    [f"{path}.values.{value_code}"],
                )

    # Rule 8: the default combination must be a real Shopify variant.
    for variant in product.shopify_variants:
        if selection_for(mapping, _variant_options(variant)) == default_selection:
            return
    raise invalid(
        "SHOPIFY_DEFAULT_NOT_FOUND",
        "No Shopify variant matches the default combination "
        + ", ".join(f"{k}={v}" for k, v in sorted(default_selection.items()))
        + ".",
        ["variants"],
    )


def variant_models(
    mapping: Optional[dict[str, Any]],
    shopify_variants,
    selections_by_model: dict[str, dict[str, str]],
) -> dict[int, Optional[str]]:
    """{shopify_variant_id: model id ("original" or a variant id) or None}.

    ``selections_by_model`` holds only live, configured models. A Shopify
    variant whose options map to no model resolves to None: the viewer then
    shows its price but no shape change. Never a substitute model.
    """
    by_selection = {
        tuple(sorted(sel.items())): model_id for model_id, sel in selections_by_model.items()
    }
    result: dict[int, Optional[str]] = {}
    for variant in shopify_variants:
        model_id = None
        if mapping:
            selection = selection_for(mapping, _variant_options(variant))
            if selection is not None:
                model_id = by_selection.get(tuple(sorted(selection.items())))
        result[variant.shopify_variant_id] = model_id
    return result
