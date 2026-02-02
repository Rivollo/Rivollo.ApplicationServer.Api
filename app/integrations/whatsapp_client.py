import httpx
from app.core.config import settings


class WhatsAppClient:
    """Low-level client for Meta WhatsApp Cloud API."""

    @staticmethod
    async def send_template_message(payload: dict) -> dict:
        url = (
            f"https://graph.facebook.com/"
            f"{settings.WHATSAPP_API_VERSION}/"
            f"{settings.WHATSAPP_PHONE_NUMBER_ID}/messages"
        )

        headers = {
            "Authorization": f"Bearer {settings.WHATSAPP_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=payload, headers=headers)

        return {
            "status_code": response.status_code,
            "text": response.text,
            "json": response.json() if response.headers.get("content-type", "").startswith("application/json") else None,
        }
