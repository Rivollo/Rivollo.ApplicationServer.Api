"""Money formatting for pricing display.

Every amount crossing the pricing API is in **minor units** — paise for INR,
cents for USD. Mixing whole rupees (as tbl_plan_prices stores them) with cents
(as tbl_plan_prices_usd stores them) in one response would be a rounding bug
waiting to happen, so the pricing service converts both to minor units and the
frontends never have to know which is which.
"""

CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$"}

# Minor units per major unit. Both currencies happen to use 100, but naming it
# keeps the conversion honest if a zero-decimal currency is ever added.
MINOR_UNITS_PER_MAJOR = {"INR": 100, "USD": 100}


def to_minor_units(major_amount: int, currency: str) -> int:
    """Convert a whole-currency-unit amount to minor units."""
    return major_amount * MINOR_UNITS_PER_MAJOR.get(currency, 100)


def format_money(minor_amount: int, currency: str) -> str:
    """Render a minor-unit amount for display.

    USD always shows two decimals ("$20.00"). INR follows the existing product
    convention of whole rupees with lakh/crore grouping ("₹1,999"), and only
    shows paise when they are non-zero.
    """
    symbol = CURRENCY_SYMBOLS.get(currency, "")
    divisor = MINOR_UNITS_PER_MAJOR.get(currency, 100)
    major, minor = divmod(minor_amount, divisor)

    if currency == "INR":
        body = _group_indian(major)
        return f"{symbol}{body}" if minor == 0 else f"{symbol}{body}.{minor:02d}"

    return f"{symbol}{major:,}.{minor:02d}"


def _group_indian(amount: int) -> str:
    """Group digits the Indian way: 1,00,000 rather than 100,000."""
    digits = str(abs(amount))
    if len(digits) <= 3:
        grouped = digits
    else:
        head, tail = digits[:-3], digits[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        grouped = ",".join(parts + [tail])
    return f"-{grouped}" if amount < 0 else grouped
