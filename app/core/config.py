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
	APP_TOKEN_EXPIRES_MINUTES: int = Field(default=1440)  # 24 hours
	APP_CLIENT_KEYS: str = Field(default="")  # comma-separated allowed client keys

	def get_allowed_client_keys(self) -> set[str]:
		return {k.strip().lower() for k in self.APP_CLIENT_KEYS.split(",") if k.strip()}

	# Password reset OTP
	PASSWORD_RESET_OTP_EXPIRES_MINUTES: int = Field(default=10)
	PASSWORD_RESET_TOKEN_EXPIRES_MINUTES: int = Field(default=15)

	# Signup email verification OTP
	SIGNUP_OTP_EXPIRES_MINUTES: int = Field(default=10)
	SIGNUP_TOKEN_EXPIRES_MINUTES: int = Field(default=15)

	# Resend email
	RESEND_API_KEY: str = Field(default="")
	RESEND_FROM_EMAIL: str = Field(default="noreply@rivollomail.com")
	RESEND_FROM_NAME: str = Field(default="Rivollo")
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
	# Supports a single URL or comma-separated list for multiple storage accounts.
	# Set explicitly in .env, or leave blank to auto-derive from AZURE_STORAGE_ACCOUNT.
	# Example (single):   https://account1.blob.core.windows.net
	# Example (multiple): https://account1.blob.core.windows.net,https://account2.blob.core.windows.net
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

	def all_blob_base_urls(self) -> list[str]:
		"""Return every blob base URL that should be rewritten to the CDN URL.

		AZURE_BLOB_BASE_URL supports a comma-separated list for multiple storage
		accounts.  Duplicates and empty strings are removed automatically.
		"""
		urls: list[str] = []
		for raw in self.AZURE_BLOB_BASE_URL.split(","):
			url = raw.strip().rstrip("/")
			if url and url not in urls:
				urls.append(url)
		return urls

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
	SAMPLING_RATIO: float = Field(default=0.1)
	ENABLE_LIVE_METRICS: bool = Field(default=False)

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

	# Firebase Cloud Messaging
	FIREBASE_JSON_PATH: str = Field(default="")
	FIREBASE_SERVICE_ACCOUNT_JSON_B64: str = Field(default="")
	FCM_DRY_RUN: bool = Field(default=False)

	# Razorpay Payment Gateway
	RAZORPAY_BASE_URL: str = Field(default="https://api.razorpay.com/v1")
	RAZORPAY_KEY_ID: str = Field(default="")
	RAZORPAY_KEY_SECRET: str = Field(default="")
	# Webhook secret — must match the value set in Razorpay Dashboard → Settings → Webhooks
	RAZORPAY_WEBHOOK_SECRET: str = Field(default="")

	# Notification thresholds
	QUOTA_NOTIFICATION_THRESHOLDS: str = Field(default="75,90,100")
	SUBSCRIPTION_EXPIRY_REMINDER_DAYS: str = Field(default="5,1,0")

	def quota_notification_thresholds(self) -> list[int]:
		return self._parse_int_list(self.QUOTA_NOTIFICATION_THRESHOLDS)

	def subscription_expiry_reminder_days(self) -> list[int]:
		return self._parse_int_list(self.SUBSCRIPTION_EXPIRY_REMINDER_DAYS)

	@staticmethod
	def _parse_int_list(raw: str) -> list[int]:
		values: list[int] = []
		for part in raw.split(","):
			part = part.strip()
			if not part:
				continue
			try:
				value = int(part)
			except ValueError:
				continue
			if value not in values:
				values.append(value)
		return values

	# WebSocket LISTEN/NOTIFY
	WS_NOTIFY_CHANNEL: str = Field(default="tbl_product_status")

	# 3D GPU cold-start duration in seconds (default 720 = 12 min).
	# Override via GPU_COLD_START_SECONDS env var per environment.
	GPU_COLD_START_SECONDS: int = Field(default=720)

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
	# Root log level for application loggers. Also controls what reaches the
	# console: without a stream handler at this level, INFO logs land only in
	# .server.log and a developer watching the terminal sees nothing.
	LOG_LEVEL: str = Field(default="INFO")
	# Per-request HTTP client logs (httpx). Useful when debugging an outbound
	# provider call — it prints the exact URL and status — but noisy otherwise.
	LOG_HTTP_REQUESTS: bool = Field(default=True)

	# fal.ai API key for server-side model calls such as SAM2 image segmentation.
	FAL_KEY: str = Field(default="")
	# Max outbound SAM2 calls to fal.ai per minute, across ALL users. fal.ai
	# rate-limits per API key, so this mirrors the upstream constraint; the
	# per-user limit below is a separate, fairness concern. Confirm the real
	# ceiling for your fal tier before raising it.
	FAL_SEGMENT_RATE_LIMIT_PER_MINUTE: int = Field(default=5)
	# Max /ai/image-segment calls per user per minute before HTTP 429 is returned.
	# Deliberately separate from OPENAI_RATE_LIMIT_PER_MINUTE: SAM2 runs are billed
	# by fal.ai and are far slower than a GPT-4o suggestion, so the two budgets are
	# tuned independently.
	SEGMENTATION_RATE_LIMIT_PER_MINUTE: int = Field(default=5)
	# How many segmentation attempts the UI offers per uploaded image.
	#
	# Served to the portal via GET /ai/segmentation-config so the number lives in
	# one place instead of being hardcoded in both the API and the frontend.
	#
	# ENFORCED BY THE FRONTEND ONLY — the API does not count attempts per image.
	# A caller hitting /ai/image-segment directly is bounded only by
	# SEGMENTATION_RATE_LIMIT_PER_MINUTE, and the count resets whenever the user
	# reloads the page. This is a UX guardrail against over-clicking, not a spend
	# control; bounding fal.ai cost would need server-side per-image state.
	SEGMENTATION_MAX_ATTEMPTS_PER_IMAGE: int = Field(default=5)

	# Tripo's OWN API (openapi.tripo3d.ai), separate from the Tripo model we call
	# through fal. Only this direct integration exposes `generate_parts`, which
	# produces a segmented mesh — fal's wrapper has no such parameter.
	TRIPO_API_KEY: str = Field(default="", description="Bearer token for Tripo's direct API.")
	TRIPO_API_BASE_URL: str = Field(default="https://openapi.tripo3d.ai/v3")
	# Geometry and texture use different model versions on purpose: Tripo
	# documents the v3.0 texture model as the recommended pairing for BOTH v3.0
	# and v3.1 geometry — there is no v3.1 texture model.
	TRIPO_GEOMETRY_MODEL: str = Field(default="v3.1-20260211")
	TRIPO_TEXTURE_MODEL: str = Field(default="v3.0-20250812")
	# Two chained long-running tasks, so this covers both stages plus queueing.
	TRIPO_MAX_WAIT_SECONDS: float = Field(default=1800.0)

	# USDZ Azure Container Apps Job — triggers the GLB -> USDZ converter job.
	# Use the SAME values as the SAM-3D service so it hits the SAME job.
	# Auth is via DefaultAzureCredential (Managed Identity on Azure; locally set
	# AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID). If any of these
	# four are empty the trigger logs a warning and no-ops.
	AZURE_SUBSCRIPTION_ID: str = Field(default="", description="Azure Subscription ID — used to trigger the USDZ converter job.")
	AZURE_RESOURCE_GROUP: str = Field(default="", description="Azure Resource Group that contains the USDZ converter job.")
	AZURE_JOB_NAME: str = Field(default="", description="Azure Container Apps Job name for GLB -> USDZ conversion.")
	AZURE_JOB_IMAGE: str = Field(default="", description="Container image used by the USDZ converter job.")

	# Draco mesh compression for fal-generated GLBs, via the glTF-Transform Node.js
	# toolchain (scripts/glb_compress). Runs after the GLB is downloaded from fal
	# and before it is uploaded to Azure. On failure the original GLB is uploaded
	# instead, so disabling this only stops compression from being attempted.
	ENABLE_DRACO_COMPRESSION: bool = Field(default=True, description="Compress generated GLBs with Draco before upload to Azure.")

	# Draco-compressed glTF PACKAGE (model.gltf + model.bin + textures), stored
	# alongside — never instead of — the Draco GLB at asset id 9. Produced from
	# the ORIGINAL generated GLB in the same glTF-Transform pass as the GLB, so
	# the two describe identical geometry and share material indices.
	#
	# Entirely non-fatal: if this fails the product still completes with its GLB
	# exactly as before, so turning it off only stops the extra artifact being
	# built. Requires sql/add_gltf_draco_asset_type.sql to have been applied.
	ENABLE_GLTF_DRACO_PACKAGE: bool = Field(default=True, description="Also store a Draco-compressed glTF package for generated meshes.")
	# tbl_asset id for the gltf_draco type. Must match the row inserted by
	# sql/add_gltf_draco_asset_type.sql (17 unless that script had to allocate
	# a different id in this environment).
	GLTF_DRACO_ASSET_ID: int = Field(default=17, description="tbl_asset id of the Draco glTF package type.")
	# Input length caps
	AI_USER_PROMPT_MAX_CHARS: int = Field(default=500)
	AI_USER_INPUT_NAME_MAX_CHARS: int = Field(default=100)
	AI_USER_INPUT_DESC_MAX_CHARS: int = Field(default=500)
	AI_LINK_URL_MAX_CHARS: int = Field(default=500)
	




settings = Settings()
