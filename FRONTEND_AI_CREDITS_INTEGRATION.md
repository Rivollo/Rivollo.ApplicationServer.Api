# Frontend Integration Notes: Move From Product Limit To AI Credits

## Summary

The backend no longer enforces product creation using `products.limit`.

Product creation is now gated by AI credits:

- `10` AI credits are consumed per product creation
- free users get `50` included AI credits
- paid monthly users get interval-specific AI credits from `tbl_plan_prices.ai_credit_limit`
- paid yearly users get interval-specific AI credits from `tbl_plan_prices.ai_credit_limit`

## What Changed In Backend

### 1. Product quota is no longer the creation gate

Previously frontend could rely on:

- `quotas.products.limit`
- `quotas.products.used`

to decide whether the user can create a product.

That is no longer correct.

Now product creation is allowed or blocked based on:

- `quotas.aiCredits.included`
- `quotas.aiCredits.purchased`
- `quotas.aiCredits.used`

Effective remaining credits:

```text
remainingCredits = included + purchased - used
```

Required credits per product creation:

```text
productCreationCost = 10
```

### 2. `/subscriptions/plans` now exposes credits per billing interval

Each item in `pricing` now includes:

- `interval`
- `priceINR`
- `aiCredits`
- `available`

Example:

```json
{
  "code": "pro",
  "name": "Pro",
  "pricing": [
    {
      "interval": "monthly",
      "priceINR": 1999,
      "aiCredits": 200,
      "available": true
    },
    {
      "interval": "yearly",
      "priceINR": 23999,
      "aiCredits": 2000,
      "available": true
    }
  ]
}
```

## Frontend Changes Needed

### A. Stop using `products.limit` as the product creation gate

Do not block "Create Product" based on:

- `quotas.products.limit`
- `quotas.products.used`

This field may still appear in the response for compatibility, but it should no longer drive the UI logic for creation eligibility.

### B. Use AI credits instead

For enabling or disabling product creation:

```text
remainingCredits = aiCredits.included + aiCredits.purchased - aiCredits.used
canCreateProduct = remainingCredits >= 10
```

### C. Show AI credit balance in UI

Recommended primary display:

- Included credits
- Purchased credits
- Used credits
- Remaining credits

Recommended formula:

```text
remainingCredits = included + purchased - used
```

Recommended labels:

- `AI Credits`
- `Remaining Credits`
- `10 credits per product`

### D. Update upgrade/paywall messaging

Old messaging likely says:

- "You have reached your product limit"

Replace with:

- "You do not have enough AI credits"
- "Each product creation uses 10 AI credits"
- "Upgrade your plan or buy more credits"

### E. Update plan comparison UI

If the plans page previously showed product limits as the main differentiator, switch to interval-specific AI credits from:

- `GET /subscriptions/plans`
- `pricing[].aiCredits`

Recommended display:

- Monthly: `200 AI credits`
- Yearly: `2000 AI credits`

Do not infer these from shared plan features.

## Expected API Usage

### 1. Current subscription

Use:

```text
GET /subscriptions/me
```

Relevant response shape:

```json
{
  "plan": "free",
  "quotas": {
    "aiCredits": {
      "included": 50,
      "purchased": 0,
      "used": 10
    },
    "products": {
      "used": 1,
      "limit": 10
    }
  }
}
```

Frontend interpretation:

- `products.used` can still be shown as informational
- `products.limit` should not be used as the create gate
- `aiCredits` is the source of truth for create eligibility

### 2. Plans list

Use:

```text
GET /subscriptions/plans
```

Relevant response shape:

```json
{
  "code": "pro",
  "pricing": [
    {
      "interval": "monthly",
      "priceINR": 1999,
      "aiCredits": 200,
      "available": true
    },
    {
      "interval": "yearly",
      "priceINR": 23999,
      "aiCredits": 2000,
      "available": true
    }
  ]
}
```

## Recommended UI Behavior

### Product creation CTA

- enabled when `remainingCredits >= 10`
- disabled when `remainingCredits < 10`

### Warning state

When credits are low:

- show remaining credits clearly
- explain that 1 product creation costs 10 credits

### Empty/blocked state

When credits are exhausted:

- disable generation CTA
- show upgrade CTA
- later, optionally show top-up CTA

## Notes For Future Add-On Credits

The backend response already includes:

- `included`
- `purchased`
- `used`

So frontend should already compute:

```text
remainingCredits = included + purchased - used
```

even if `purchased` is currently `0`.

That will make future credit top-up integration much easier.

## Suggested Frontend Refactor

### Replace old logic

Old logic:

```text
canCreate = products.used < products.limit
```

New logic:

```text
remainingCredits = aiCredits.included + aiCredits.purchased - aiCredits.used
canCreate = remainingCredits >= 10
```

### Keep product count only for display

Examples:

- total products created
- products in account
- catalog size

Do not use product count to gate the create action.

## QA Checklist

- Free user sees `50` included AI credits
- After one product creation, AI credits used increases by `10`
- Product creation button disables when remaining credits are below `10`
- Plans page shows monthly and yearly `aiCredits` from `pricing`
- Product count can still be displayed without affecting create eligibility
