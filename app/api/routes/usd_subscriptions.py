"""USD subscription routes — creation and promo validation for customers
outside India.

Verification and cancellation are not duplicated here: POST
/razorpay-subscriptions/verify and /cancel are currency-agnostic (a signature
check, a status flip, a licence sync) and serve USD subscriptions unchanged.
Only creation differs by currency, so only creation lives here.
"""

from fastapi import APIRouter, Query, Request, status

from app.api.deps import CurrentUser, DB
from app.core.geo import resolve_checkout_country
from app.schemas.usd_subscriptions import (
    CreateUsdSubscriptionRequest,
    CreateUsdSubscriptionResponse,
    ValidateUsdPromoResponse,
    VerifyUsdSubscriptionRequest,
    VerifyUsdSubscriptionResponse,
)
from app.services.usd_subscription_service import (
    create_usd_subscription,
    validate_usd_promo_code,
    verify_usd_subscription,
)
from app.utils.envelopes import api_success

router = APIRouter(prefix="/usd-subscriptions", tags=["usd-subscriptions"])


@router.post(
    "/create",
    response_model=dict,
    status_code=status.HTTP_201_CREATED,
    summary="Create a USD recurring subscription",
    description="""
Create a USD Razorpay subscription — the **first step** of the non-India
subscription flow. The INR flow is unchanged and still lives at
`POST /razorpay-subscriptions/create`.

### What the client sends
Only `planCode`, `billingInterval` and an optional `promoCode`. A request
carrying an amount, a currency or a plan ID is rejected: the price is resolved
server-side from the tier and interval.

### Monthly vs annual
- **Monthly** — the subscription starts one calendar month from now and an
  upfront amount is charged at authentication. With the advertised promo that
  upfront amount is half the list price; without one it is the full list price.
  Either way the customer gets a full period for what they pay today.
- **Annual** — starts immediately at the list price. It carries no promo
  mechanism: the two-months-free discount is permanent and already inside the
  annual price. A promo code submitted against annual is rejected.

### Next steps
Pass `subscriptionId` and `keyId` to the Razorpay Checkout widget, then send the
callback values to `POST /razorpay-subscriptions/verify`.
""",
)
async def create_usd_sub(
    body: CreateUsdSubscriptionRequest,
    request: Request,
    current_user: CurrentUser,
    db: DB,
) -> dict:
    """Create a USD Razorpay subscription for the authenticated user."""
    result = await create_usd_subscription(
        db,
        user_id=current_user.id,
        plan_code=body.plan_code,
        billing_interval=body.billing_interval,
        promo_code=body.promo_code,
        # Cloudflare's header only. A forwarded country is never accepted on the
        # path that decides who may be charged in USD.
        checkout_country=resolve_checkout_country(request),
    )

    response = CreateUsdSubscriptionResponse(
        subscription_id=result["subscriptionId"],
        plan_code=result["planCode"],
        key_id=result["keyId"],
        status=result["status"],
        short_url=result.get("shortUrl"),
        currency=result["currency"],
        billing_interval=result["billingInterval"],
        full_amount=result["fullAmount"],
        upfront_amount=result.get("upfrontAmount"),
        promo_code=result.get("promoCode"),
        first_charge_at=result.get("firstChargeAt"),
    )
    return api_success(response.model_dump(by_alias=True))


@router.post(
    "/verify",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Verify a completed USD subscription payment",
    description="""
Verify a completed USD checkout — the **second step** of the flow. Send the three
values Razorpay's checkout handler returns.

Separate from `POST /razorpay-subscriptions/verify` because that one resolves
entitlements through the INR price list, which for a USD subscription would apply
INR credit limits and could fail *after* the card was charged.

Usage quotas are not reset here — that is the webhook's job when a new billing
period actually begins.
""",
)
async def verify_usd_sub(
    body: VerifyUsdSubscriptionRequest,
    current_user: CurrentUser,
    db: DB,
) -> dict:
    """Verify a USD subscription payment for the authenticated user."""
    result = await verify_usd_subscription(
        db,
        user_id=current_user.id,
        razorpay_payment_id=body.razorpay_payment_id,
        razorpay_subscription_id=body.razorpay_subscription_id,
        razorpay_signature=body.razorpay_signature,
    )
    response = VerifyUsdSubscriptionResponse(
        verified=result["verified"],
        message=result["message"],
        plan=result.get("plan"),
        subscription_id=result.get("subscriptionId"),
        period_end=result.get("periodEnd"),
    )
    return api_success(response.model_dump(by_alias=True, exclude_none=True))


@router.get(
    "/promo/validate",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Validate a USD promo code before checkout",
    description="""
Checks a promo code and returns the price it would produce, so the code-entry UI
can show the customer the real number before they commit.

Uses the same resolution as checkout, so a code that validates here is a code
that applies there. Rejections come back as a 400 with a customer-facing reason
rather than a silent fall-through to full price.
""",
)
async def validate_usd_promo(
    current_user: CurrentUser,
    db: DB,
    code: str = Query(..., max_length=64, description="Promo code to validate."),
    plan_code: str = Query("pro", alias="planCode", description="Tier the code applies to."),
    billing_interval: str = Query(
        "monthly", alias="billingInterval", description="'monthly' or 'yearly'."
    ),
) -> dict:
    """Validate a USD promo code for the authenticated user."""
    result = await validate_usd_promo_code(
        db,
        user_id=current_user.id,
        plan_code=plan_code,
        billing_interval=billing_interval,
        code=code,
    )
    response = ValidateUsdPromoResponse(
        valid=result["valid"],
        code=result["code"],
        currency=result["currency"],
        full_amount=result["fullAmount"],
        upfront_amount=result["upfrontAmount"],
        description=result.get("description"),
    )
    return api_success(response.model_dump(by_alias=True, exclude_none=True))
