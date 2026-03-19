import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

	APP_NAME: str = Field(default="Rivollo API")
	DEBUG: bool = Field(default=False)
	API_PREFIX: str = Field(default="")

	# Database
	DATABASE_URL: str = Field(default="")

	# Auth / JWT
	JWT_SECRET: str = Field(default="dev-change-me")
	JWT_ALGORITHM: str = Field(default="HS256")
	ACCESS_TOKEN_EXPIRES_MINUTES: int = Field(default=60)

	# Password reset OTP
	PASSWORD_RESET_OTP_EXPIRES_MINUTES: int = Field(default=10)
	PASSWORD_RESET_TOKEN_EXPIRES_MINUTES: int = Field(default=15)

	# Signup email verification OTP
	SIGNUP_OTP_EXPIRES_MINUTES: int = Field(default=10)
	SIGNUP_TOKEN_EXPIRES_MINUTES: int = Field(default=15)

	# SendGrid email
	SENDGRID_API_KEY: str = Field(default="")
	SENDGRID_FROM_EMAIL: str = Field(default="noreply@rivollomail.com")
	SENDGRID_FROM_NAME: str = Field(default="Rivollo")
	SENDGRID_URL: str = Field(default="https://api.sendgrid.com/v3/mail/send")
	SUPPORT_EMAIL: str = Field(default="")

	# Frontend base URL (used in email links)
	FRONTEND_URL: str = Field(default="http://localhost:3000")

	# Storage / CDN placeholders (wire Azure later)
	CDN_BASE_URL: str = Field(default="https://cdn.example")
	STORAGE_CONTAINER_UPLOADS: str = Field(default="uploads")
	STORAGE_CONTAINER_MEDIA: str = Field(default="")  # product/background images; falls back to STORAGE_CONTAINER_UPLOADS
	AZURE_STORAGE_ACCOUNT: str = Field(default="")
	AZURE_STORAGE_KEY: str = Field(default="")
	AZURE_STORAGE_CONN_STRING: str = Field(default="")

	# External model service endpoint
	MODEL_SERVICE_URL: str = Field(default="mock://local")

	# 3D model generation API base URL (env key: 3D_MODEL_API_BASE_URL)
	MODEL_3D_API_BASE_URL: str = Field(
		default="",
		validation_alias=AliasChoices("MODEL_3D_API_BASE_URL", "3D_MODEL_API_BASE_URL"),
	)

	# Azure Monitor / Application Insights
	AZURE_MONITOR_CONN_STR: str = Field(default="")
	ENABLE_APP_INSIGHTS: bool = Field(default=True)
	SAMPLING_RATIO: float = Field(default=1.0)

	# Public API basic auth
	PUBLIC_API_USERNAME: str = Field(default="public")
	PUBLIC_API_PASSWORD: str = Field(default="public-secret")
	# Google OAuth
	GOOGLE_CLIENT_ID: str = Field(default="") 

	SERVICEBUS_CONNECTION_STRING: str = Field(default="")
	SERVICEBUS_QUEUE_NAME: str = Field(default="")

	# WhatsApp Business API			
	WHATSAPP_ACCESS_TOKEN: str = Field(default="")
	WHATSAPP_PHONE_NUMBER_ID: str = Field(default="")
	WHATSAPP_TEMPLATE_NAME: str = Field(default="")
	WHATSAPP_TEMPLATE_LANGUAGE: str = Field(default="en_US")
	WHATSAPP_API_VERSION: str = Field(default="v18.0")

	# Razorpay Payment Gateway
	RAZORPAY_BASE_URL: str = Field(default="https://api.razorpay.com/v1")
	RAZORPAY_KEY_ID: str = Field(default="")
	RAZORPAY_KEY_SECRET: str = Field(default="")
	# Webhook secret — must match the value set in Razorpay Dashboard → Settings → Webhooks
	RAZORPAY_WEBHOOK_SECRET: str = Field(default="")
	




settings = Settings()
