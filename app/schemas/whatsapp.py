from pydantic import BaseModel, Field  
from enum import Enum
from pydantic import field_validator
import re

class TemplateVarient(str, Enum):
    basic = "basic"
    professional = "professional"
    button = "button"


class SendWhatsAppLinkRequest(BaseModel):
    phone_number: str = Field(..., description="The recipient's phone number in international format.")

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v: str) -> str:
        """
        Validate phone number format.
        Must start with '+' followed by 1 to 15 digits.
        """
        if not re.match(r"^\+[1-9]\d{1,14}$", v):
            raise ValueError("Phone number must start with '+' followed by 1 to 15 digits")
        return v

    product_name: str = Field(..., description="The name of the product.")
    company_name: str = Field(..., description="The name of the company sending the link.")
    product_link: str = Field(..., description="The product link to be sent via WhatsApp.")
    template_varient: TemplateVarient = Field(
        TemplateVarient.basic,
        description="The varient of the WhatsApp template to be used."
    )

class SendWhatsAppLinkResponse(BaseModel):
    success: bool = Field(..., description="Indicates if the message was sent successfully.")
    message_id: str | None = Field(None, description="The ID of the sent message, if successful.")
    error_message: str | None = Field(None, description="Error message in case of failure.")
    