"""Payment routes — promo code validation.

All subscription management is handled by /razorpay-subscriptions.
"""

from fastapi import APIRouter, Query, status

from app.api.deps import CurrentUser, DB
from app.database.promo_repo import PromoRepository
from app.schemas.payments import ValidatePromoResponse
from app.utils.envelopes import api_success

router = APIRouter(prefix="/payments", tags=["payments"])


@router.get(
    "/promo/validate",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Validate a promo code",
    description="""
Validate a promo code against the selected plan.

### How it works
1. User enters only the `code`. The frontend automatically sends `planCode` from the current plan context.
2. Only call this endpoint when the user has entered a promo code — skip the call if the field is empty.
3. The backend checks the code is active, within its validity window, not exceeded its usage limit, and applicable to the given plan.
4. On success, `razorpayOfferId` is returned — pass this in the Razorpay checkout payload.

### Error codes
| Code | Meaning |
|------|---------|
| `400` | Promo code is invalid, expired, or not applicable to the given plan |
| `401` | Not authenticated |
""",
    responses={
        200: {"description": "Promo code is valid"},
        400: {"description": "Invalid, expired, or inapplicable promo code"},
        401: {"description": "Authentication required"},
    },
)
async def validate_promo_code(
    db: DB,
    current_user: CurrentUser,
    code: str = Query(..., description="Promo code to validate."),
    plan_code: str = Query(..., description="Plan code — sent automatically by the frontend from the selected plan context."),
) -> dict:
    """Validate a promo code against the given plan. Frontend sends plan_code automatically; user only inputs the code."""
    promo = await PromoRepository.get_by_code(db, code)

    if not promo:
        return api_success(
            ValidatePromoResponse(
                valid=False,
                code=code,
                message="Promo code is invalid or expired.",
            ).model_dump(by_alias=True, exclude_none=True)
        )

    if promo.plan_code and promo.plan_code != plan_code:
        return api_success(
            ValidatePromoResponse(
                valid=False,
                code=code,
                message="Promo code is not valid for this plan.",
            ).model_dump(by_alias=True, exclude_none=True)
        )

    response = ValidatePromoResponse(
        valid=True,
        code=promo.code,
        discount_type=promo.discount_type,
        discount_value=promo.discount_value,
        description=promo.description,
        razorpay_offer_id=promo.razorpay_offer_id,
        message="Promo code is valid.",
    )
    return api_success(response.model_dump(by_alias=True))
