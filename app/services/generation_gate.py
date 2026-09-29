"""Plan and AI-credit gate for paid 3D generation.

The same three checks ``POST /createProductFal`` makes inline
(app/api/routes/products.py): resolve the requested model, refuse a paid model
on a Free plan (naming the free alternative), and refuse when the seller does
not have the model's credit cost left. Same rules, same messages, so a seller
sees one behaviour whichever entry point they use.

Used by the newer generation paths (layout-from-photo, the Shopify
integration). ``/createProductFal`` keeps its inline copy for now, untouched;
switching it to this module is a behaviour-preserving follow-up that needs its
own regression test.

Charging is separate from authorising on purpose: a caller authorises, writes
its own row, and only then charges, so a failed write never costs credits.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations.fal.registry import FalModelSpec, get_model_spec, list_model_specs
from app.services.licensing_service import LicensingService

PAID_PLANS = ("pro", "enterprise")


async def authorize_generation(
    db: AsyncSession, user_id: uuid.UUID, model_key: Optional[str]
) -> FalModelSpec:
    """The model to run, if this seller may run it now. Raises 400 / 403.

    Does not charge. Call :func:`charge_generation` once the work is recorded.
    """
    try:
        spec = await get_model_spec(db, model_key)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    if not spec.free_plan_eligible:
        plan_code = await LicensingService.get_user_plan_code(db, user_id)
        if plan_code not in PAID_PLANS:
            free_alternative = next(
                (s for s in await list_model_specs(db) if s.free_plan_eligible), None
            )
            suggestion = (
                f" or select {free_alternative.label}, which is free on every plan."
                if free_alternative
                else "."
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"{spec.label} requires a Pro or Enterprise plan. "
                    f"Please subscribe to continue{suggestion}"
                ),
            )

    allowed, _ = await LicensingService.check_quota(
        db, user_id, "ai_credits", increment=spec.credit_cost
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Not enough AI credits. {spec.credit_cost} credits "
                "are required for this generation."
            ),
        )
    return spec


async def charge_generation(db: AsyncSession, user_id: uuid.UUID, credit_cost: int) -> None:
    """Deduct the credits. Commits (LicensingService.increment_usage does)."""
    await LicensingService.increment_usage(db, user_id, "ai_credits", increment=credit_cost)
