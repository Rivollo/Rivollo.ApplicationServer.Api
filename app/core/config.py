from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, AliasChoices, model_validator


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

	APP_NAME: str = Field(default="Rivollo API")
	DEBUG: bool = Field(default=False)
	API_PREFIX: str = Field(default="")

	# Database
	DATABASE_URL: str = Field(default="")
	# Set to True on Azure (App Service / Container App) to use Managed Identity
	# instead of a hardcoded password in DATABASE_URL.
	# The DATABASE_URL should then have NO password, e.g.:
	#   postgresql+asyncpg://your-app-name@server.postgres.database.azure.com:5432/dbname?ssl=require
	USE_MANAGED_IDENTITY: bool = Field(default=False)
	MANAGED_IDENTITY_TOKEN_SCOPE: str = Field(default="https://ossrdbms-aad.database.windows.net/.default")
	# Required for User-Assigned Managed Identity.
	MANAGED_IDENTITY_CLIENT_ID: str = Field(default="")

	@model_validator(mode="after")
	def _require_client_id_for_managed_identity(self) -> "Settings":
		if self.USE_MANAGED_IDENTITY and not self.MANAGED_IDENTITY_CLIENT_ID:
			raise ValueError(
				"MANAGED_IDENTITY_CLIENT_ID must be set when USE_MANAGED_IDENTITY is true. "
				"Find it in Azure Portal → Managed Identities → your identity → Overview → Client ID."
			)
		return self

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

	# Storage / CDN — set CDN_BASE_URL to your Azure CDN / Front Door hostname (no trailing slash)
	CDN_BASE_URL: str = Field(default="")
	STORAGE_CONTAINER_UPLOADS: str = Field(default="uploads")
	STORAGE_CONTAINER_MEDIA: str = Field(default="")  # product/background images; falls back to STORAGE_CONTAINER_UPLOADS
	AZURE_STORAGE_ACCOUNT: str = Field(default="")
	AZURE_STORAGE_KEY: str = Field(default="")
	AZURE_STORAGE_CONN_STRING: str = Field(default="")

	# Azure Blob Storage base URL — used by CDN middleware to rewrite blob URLs in responses.
	# Set explicitly in .env, or leave blank to auto-derive from AZURE_STORAGE_ACCOUNT.
	AZURE_BLOB_BASE_URL: str = Field(default="")

	@model_validator(mode="after")
	def _derive_blob_base_url(self) -> "Settings":
		if self.AZURE_BLOB_BASE_URL:
			return self
		if self.AZURE_STORAGE_ACCOUNT:
			self.AZURE_BLOB_BASE_URL = f"https://{self.AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
			return self
		for part in self.AZURE_STORAGE_CONN_STRING.split(";"):
			if part.startswith("AccountName="):
				account = part[len("AccountName="):]
				if account:
					self.AZURE_BLOB_BASE_URL = f"https://{account}.blob.core.windows.net"
					return self
		return self

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

	# OpenAI — GPT-4o Vision for AI suggestions
	# In Azure App Service, override via env vars: OPENAI_API_KEY, OPENAI_MODEL, OPENAI_MAX_TOKENS
	OPENAI_API_KEY: str = Field(default="")
	OPENAI_CHAT_URL: str = Field(default="https://api.openai.com/v1/chat/completions")
	OPENAI_MODEL: str = Field(default="gpt-4o")
	OPENAI_USE_AZURE: bool = Field(default=False)
	OPENAI_MAX_TOKENS: int = Field(default=200)
	# Max AI calls per user per minute before HTTP 429 is returned
	OPENAI_RATE_LIMIT_PER_MINUTE: int = Field(default=5)
	# Max retry attempts on OpenAI 429/5xx before giving up
	OPENAI_MAX_RETRIES: int = Field(default=3)
	# Per-request HTTP timeout in seconds (applies to each attempt, not the total)
	OPENAI_TIMEOUT_SECONDS: int = Field(default=30)
	# Input length caps
	AI_USER_PROMPT_MAX_CHARS: int = Field(default=500)
	AI_USER_INPUT_NAME_MAX_CHARS: int = Field(default=100)
	AI_USER_INPUT_DESC_MAX_CHARS: int = Field(default=500)
	AI_LINK_URL_MAX_CHARS: int = Field(default=500)
	




settings = Settings()
