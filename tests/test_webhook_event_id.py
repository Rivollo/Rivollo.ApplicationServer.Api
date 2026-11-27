"""Every renewal must be its own webhook event.

Razorpay sends an event's unique id only in the X-Razorpay-Event-Id header; the
JSON body has none. The handler used to read the id from the body, found
nothing, and fell back to "{event}_{subscription}" — the same key for every
monthly subscription.charged. The second charge onwards was then skipped as an
already-processed duplicate: the period was not extended and the credits were
not refilled, for a customer who had paid.
"""

from app.services.subscription_webhook_service import resolve_event_id


def _charged(payment_id=None, created_at=None):
    payload = {"event": "subscription.charged", "payload": {}}
    if payment_id:
        payload["payload"]["payment"] = {"entity": {"id": payment_id}}
    if created_at:
        payload["created_at"] = created_at
    return payload


def _resolve(payload, header=""):
    return resolve_event_id(
        header_event_id=header,
        payload=payload,
        event="subscription.charged",
        rz_subscription_id="sub_1",
    )


def test_header_id_is_the_key():
    assert _resolve(_charged("pay_1"), header="evt_abc") == "evt_abc"


def test_two_renewals_without_a_header_get_different_keys():
    first = _resolve(_charged("pay_1", 1700000000))
    second = _resolve(_charged("pay_2", 1702592000))
    assert first != second


def test_a_redelivery_of_the_same_event_keeps_its_key():
    assert _resolve(_charged("pay_1", 1700000000)) == _resolve(_charged("pay_1", 1700000000))


def test_created_at_separates_events_that_carry_no_payment():
    assert _resolve(_charged(created_at=1)) != _resolve(_charged(created_at=2))


def test_the_route_passes_the_header_on():
    import inspect

    from app.api.routes import razorpay_subscriptions

    source = inspect.getsource(razorpay_subscriptions.subscription_webhook)
    assert "x-razorpay-event-id" in source
    assert "event_id_header=event_id_header" in source
