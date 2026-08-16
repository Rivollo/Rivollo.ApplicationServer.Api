# USD rollout — outstanding work

Everything the USD billing path and the new policy pages still need before they
can go live. Grouped by what blocks what, not by repo.

Code is on the branch `feature/usd-billing` in all three repos.

---

## 1. The policies now promise things the product does not do

This is the most important section on the page. The terms are a binding
document and they went further than the code. Each item below is a gap between
what a customer can now read and what actually happens.

### Model visibility and licensing

The terms state that free-plan output is licensed CC BY 4.0 and public, that
paid-plan output is private, and that output generated on a paid plan stays
private after a downgrade. None of this exists yet. `is_public` today lives
only on `tbl_galleries` as a user-set flag — products and generated models
carry no visibility field and no licence field at all.

- [ ] Add a licence field to the product/model row, written **at generation
      time** from the plan that was active at that moment.
- [ ] Add `licensed_at` alongside it.
- [ ] Derive visibility from that stored per-model licence, never from the
      user's *current* plan.
- [ ] Enforce it: free-plan models publicly reachable, paid-plan models not.
- [ ] Display attribution on public models — creator credit, a licence notice,
      and a link to the CC BY 4.0 deed. Attribution is what makes the licence
      valid; without it we are publishing unlicensed work.
- [ ] Credit the creator wherever Rivollo itself features a free-plan model —
      gallery, website, marketing, docs.

> **Why the stored field rather than a plan lookup.** CC BY 4.0 is
> irrevocable, and the terms say so. A model generated on the free plan stays
> CC BY 4.0 after the creator upgrades, and a model generated on a paid plan
> stays private after they downgrade. Reading visibility from the current plan
> gets both of those backwards, and the second one — silently publishing work
> someone paid to keep private — is the version that ends up on social media.

### Checkout consent

- [ ] Capture the consent to immediate performance at checkout, on both the
      INR and USD paths. The terms and the refund policy both say we do this,
      and it is what waives the EU/UK 14-day withdrawal right. Without it,
      every EU consumer has 14 days to withdraw regardless of our own window.
- [ ] Store the consent with the subscription record so it can be produced as
      chargeback evidence.
- [ ] Show the renewal terms at checkout — the amount, the date, and that it
      renews until cancelled. Already true of the USD promo copy; confirm the
      INR path says it too.

### Privacy commitments

- [ ] A process for responding to data-subject requests within 30 days.
- [ ] A breach-notification process.
- [ ] Deletion of account content within 90 days of closure.
- [ ] Log retention capped at 12 months.

---

## 2. Razorpay

- [ ] Submit for international activation. The four policy pages it requires
      now exist and are linked in the footer.
- [ ] Enable USD on Subscriptions.
- [ ] Create two USD plans: Pro monthly `2000`, Pro annual `20000` (cents).
- [ ] Confirm all six webhook events are ticked, **especially
      `subscription.authenticated`** — a no-op for INR, but the event that
      grants access for USD monthly. If it is off, a customer pays and gets
      nothing.
- [ ] Do not create Offers for USD; they are INR-locked and fail silently.

---

## 3. Database

The order below is not interchangeable. USD price rows share a table with INR
ones, and the old image's price lookup has no currency filter — so inserting a
USD row before the new image is serving makes every rupee checkout for that plan
fail outright.

`sql/add_usd_pricing.sql` is **schema only** — it adds columns and widens one
constraint, and writes no data at all. Every row below is yours to insert from
your own client, so nothing changes the data without you running it.

- [ ] **1.** Run `sql/add_usd_pricing.sql`. Safe against the running old image.
- [ ] **2.** Deploy the API (merge to `main`).
- [ ] **3.** Confirm the new image is live: `GET /pricing` with
      `cf-ipcountry: US` returns `"currency": "USD"`. The old image has no
      `/pricing` route at all.
- [ ] **4.** Insert the rows in the next section.
- [ ] Repeat all four on production, in the same order.
- [ ] New columns for the model licence work in section 1.

### The rows that must exist

Written as a specification rather than a script. Values in `price_inr` are
**whole units of the row's currency** — so `20` means twenty dollars.

**`tbl_plan_prices`** — four new rows.

| plan | interval | currency | price_inr | ai_credit_limit | total_count | razorpay_plan_id |
|---|---|---|---|---|---|---|
| pro | monthly | USD | 20 | 2000 | 1200 | `plan_TQ2m22UBRutnZu` |
| pro | yearly | USD | 200 | 24000 | 100 | `plan_TQ8SZe3nf6a0d3` |
| free | monthly | USD | 0 | 100 | 0 | NULL |
| free | yearly | USD | 0 | 100 | 0 | NULL |

Notes on the numbers, because several of them are load-bearing:

- **`price_inr = 200` for annual is ten times monthly, not twelve.** The two
  months free live permanently in the list price. Do not set 240 and add a
  two-month promo: that discount expires, and a foreign card gets a ~20%
  increase a year later with nobody in the loop. (This is what INR currently
  does through the `FIRST2MONTHS` offer.)
- **`ai_credit_limit` decides what a USD customer receives.** Copy it from the
  matching INR row so entitlements are identical — 2000 monthly, 24000 yearly.
- **`total_count`** is billing cycles before the subscription ends. 1200 months
  is a century; for annual keep it near 100, never 1200, which Razorpay can
  reject outright at checkout.
- **`razorpay_plan_id` NULL is what marks a tier as not purchasable.** That is
  why the free rows are safe: checkout rejects a NULL gateway plan with a 400
  before reaching Razorpay.
- Plan IDs are **per Razorpay mode.** A Test-mode ID only resolves against
  Test-mode keys, and they look identical.

Optionally, the same two free rows in `currency = 'INR'` (price 0,
`razorpay_plan_id` NULL). Without them `/pricing` omits the Free tier for
Indian visitors while showing it for everyone else. The only other visible
change is that `/subscriptions/plans` starts reporting Free at zero and
unpurchasable rather than reporting no pricing for it.

**`tbl_promo_codes`** — one new row, the advertised first-month discount.

| column | value | why |
|---|---|---|
| code | `USDINTRO50` | confirmed not to collide |
| discount_type | `percentage` | matches the existing CHECK constraint |
| discount_value | `50` | |
| billing_interval | `monthly` | annual is never eligible |
| plan_code | `pro` | |
| currency | `USD` | without it the INR lookup could match this code |
| is_public | `true` | advertised on the pricing page and auto-applied |
| max_usage | NULL | uncapped |
| valid_from / valid_to | now / +5 years | |
| razorpay_offer_id | NULL | USD discounts are computed server-side |

**`tbl_subscriptions`** — one backfill, whenever convenient:
`billing_country = 'IN'` where it is NULL. All 117 existing rows predate USD.
Nothing breaks while they are NULL; the column is only read for reporting,
never for currency resolution.

### Verifying

```sql
SELECT p.code, u.billing_interval, u.currency, u.price_inr,
       u.ai_credit_limit, u.total_count, u.razorpay_plan_id
FROM   tbl_plan_prices u
JOIN   tbl_mstr_plans p ON p.id = u.plan_id
ORDER  BY p.code, u.currency, u.billing_interval;
```

`sql/rollback_usd_pricing.sql` reverses the schema change; it refuses to run
while live USD subscriptions exist.

USD shares `tbl_plan_prices` and `tbl_promo_codes` with INR, separated by the
`currency` column, so both tables hold two rows per plan and interval. Every
read must filter on currency — `tests/test_currency_isolation.py` fails if any
of the lookups stops doing so.

---

## 4. Infrastructure

- [ ] Set `API_BASE_URL` on the marketing container app. Runtime env var, not a
      build arg, and the deploy workflow sets no runtime variables — so this is
      manual until someone adds it to the workflow.
- [ ] Consider adding it to `dev-deploy.yml` so a container rebuild cannot lose
      it.
- [ ] Lock the Azure origin to Cloudflare IP ranges. This is correctness, not
      hardening: a request that bypasses Cloudflare has no `CF-IPCountry`, and
      no country resolves to USD — so an Indian customer reaching the origin
      directly could check out in dollars.
- [ ] Bypass Cloudflare cache on `/pricing` (marketing) and `GET /pricing`
      (API). Both vary by country; one cached copy served across regions shows
      rupees to Americans.
- [ ] Check no broad "Cache Everything" rule already catches `/pricing`.

---

## 5. Merge and deploy

- [ ] Push the three branches and open PRs.
- [ ] Merge the API first. The frontends fall back to INR when `/pricing` is
      unavailable, so landing them first quotes rupees to US visitors — worse
      than a visible error.
- [ ] Dev deploys on merge to `main`. Production is `workflow_dispatch` in all
      three and will not fire on its own.

Optionally, cherry-pick the policy commits onto their own branch and merge them
first — Razorpay's international review is the long pole, and it only needs the
pages to be live.

---

## 6. Legal review

None of the policy text has been reviewed by a lawyer. It is standard and
defensible, but it is not advice.

- [ ] Have all four pages reviewed.
- [ ] Decide whether to restore a named arbitration seat. The clause now reads
      "the seat of arbitration shall be in India", which is enforceable but
      looser than naming a city — that was the cost of removing Pune.
- [ ] Confirm the 8-year billing-record retention with the CA.
- [ ] Decide whether Razorpay's review needs a full registered address. It
      currently displays as "India"; one line in `config/company.ts` changes it
      everywhere.
- [ ] Confirm the CC BY 4.0 arrangement is what you want commercially before
      the pages go live. It is irrevocable for every model generated under it,
      so it is much easier to adopt than to reverse.

---

## 7. Phase 2 — deferred by agreement

- [ ] USD invoices carrying the LUT declaration. Blocked on wording from the
      CA.
- [ ] Subscription confirmation email stating the full price and the date of
      the first full-price charge.
- [ ] The reminder email 7 days before that charge. Needs a scheduler.
- [ ] A reconciliation sweep over `processed = false` rows in
      `tbl_webhook_events`. `webhook_inbox.unprocessed_events()` is the query;
      nothing consumes it yet, so recovery is manual. An Azure Container Apps
      Job on cron would close this and provide the scheduler the two emails
      above need.

---

## 8. Known unknowns

- [ ] Whether Razorpay's checkout widget collects a billing address for
      international cards, which AVS and 3DS may want. One sandbox transaction
      answers it.
- [ ] Confirm the widget offers cards only for USD — UPI and eMandate are
      INR-only.
- [ ] `requirements.txt` is missing `azure-servicebus`. Unrelated to this work
      and harmless in deployment (containers build from `uv.lock`), but a local
      pip install from that file cannot import the app.

---

## Not done, deliberately

The INR path was not refactored, cleaned up, or shared with the USD path. An
Indian customer's request executes the same code before and after this work.
Two bugs were fixed in it, both with your explicit go-ahead: the replayable
verify endpoint that reset usage counters, and the webhook inbox that dropped
failed events while returning 200.
