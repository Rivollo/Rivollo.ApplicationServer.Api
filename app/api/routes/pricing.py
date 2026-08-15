"""Public pricing route.

Architecture:
    - Route is a thin HTTP layer: resolve deps -> call service -> envelope.
    - All resolution logic lives in pricing_service.py.
    - No authentication required, but an optional bearer token is honoured so a
      signed-in customer sees the currency their subscription is locked to.
"""

from fastapi import APIRouter, Request, Response, status

from app.api.deps import DB, OptionalUser
from app.services.pricing_service import get_pricing
from app.utils.envelopes import api_success

router = APIRouter(tags=["pricing"])


@router.get(
    "/pricing",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Currency-aware pricing for the current visitor",
    description="""
Returns everything a pricing page needs to render, resolved for the caller's
country. Consumed by both the marketing site and the application, so the two
cannot show different prices.

### Currency resolution
1. If the caller is authenticated and already has a subscription, its currency
   wins — currency is locked at first subscription and never changes.
2. `CF-IPCountry: IN` → INR.
3. Any other country code → USD.
4. Header absent, or `XX` / `T1` → USD.

### What this does not return
Razorpay plan IDs. Checkout takes a tier and a billing interval and resolves the
plan server-side, so a client can never name the plan it is charged for.

### Caching
Responses vary by caller location and identity. Never cache this at a shared
layer — a cached response serves one region's pricing to another.
""",
)
async def read_pricing(
    request: Request,
    response: Response,
    db: DB,
    current_user: OptionalUser = None,
) -> dict:
    """Return currency-aware pricing for the requesting visitor."""
    # Geo- and identity-varying: a shared cache holding this would leak one
    # region's prices to another.
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Vary"] = "CF-IPCountry, X-Rvl-Country, Authorization"

    pricing = await get_pricing(db, request=request, current_user=current_user)
    return api_success(pricing.model_dump(by_alias=True, exclude_none=True))
