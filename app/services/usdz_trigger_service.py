"""
USDZ Trigger Service

Fires an Azure Container Apps Job to convert GLB -> USDZ.
Fire-and-forget: does not wait for job completion.

Authentication:
  - Azure (production): Managed Identity (no credentials needed)
  - Local / VM:         Service Principal via AZURE_CLIENT_ID,
                        AZURE_CLIENT_SECRET, AZURE_TENANT_ID
  DefaultAzureCredential handles both automatically.
"""

import asyncio
import logging
import time
import uuid
from functools import partial
from typing import Optional

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from azure.identity import DefaultAzureCredential
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False
    logger.warning(
        "azure-identity not installed — USDZ job triggering disabled. "
        "Add 'azure-identity' to requirements.txt."
    )

_ARM_API_VERSION = "2024-03-01"
_ARM_SCOPE = "https://management.azure.com/.default"


class USDZTriggerService:
    _MAX_RETRIES = 3
    _BASE_BACKOFF = 2  # seconds — doubles each attempt: 2s, 4s

    def _is_configured(self) -> bool:
        return bool(
            _SDK_AVAILABLE
            and settings.AZURE_SUBSCRIPTION_ID
            and settings.AZURE_RESOURCE_GROUP
            and settings.AZURE_JOB_NAME
            and settings.AZURE_JOB_IMAGE
        )

    def _trigger_sync(
        self,
        glb_blob_url: str,
        product_id: str,
        user_id: str,
        product_name: str,
        output_blob_name: str,
        job_id: str,
    ) -> None:
        last_error: Optional[Exception] = None

        url = (
            f"https://management.azure.com/subscriptions/{settings.AZURE_SUBSCRIPTION_ID}"
            f"/resourceGroups/{settings.AZURE_RESOURCE_GROUP}"
            f"/providers/Microsoft.App/jobs/{settings.AZURE_JOB_NAME}"
            f"/start?api-version={_ARM_API_VERSION}"
        )

        body = {
            "containers": [
                {
                    "name": settings.AZURE_JOB_NAME,
                    "image": settings.AZURE_JOB_IMAGE,
                    "args": [
                        f"--job-id={job_id}",
                        f"--glb-blob-url={glb_blob_url}",
                        f"--output-blob-name={output_blob_name}",
                        f"--product-id={product_id}",
                        f"--user-id={user_id}",
                        f"--product-name={product_name}",
                    ],
                    "env": [
                        {"name": "STORAGE_CONTAINER",          "secretRef": "storagecontainer"},
                        {"name": "AZURE_BLOB_BASE_URL",        "secretRef": "azureblobbaseurl"},
                        {"name": "AZURE_STORAGE_CONN_STRING",  "secretRef": "azurestoragconnectionstring"},
                        {"name": "BLENDER_BIN",                "secretRef": "blenderbin"},
                        {"name": "CDN_BASE_URL",               "secretRef": "cdnbaseurl"},
                        {"name": "DATABASE_URL",               "secretRef": "databaseurl"},
                        {"name": "BAKE_RESOLUTION",            "secretRef": "bakeresolution"},
                    ],
                }
            ]
        }

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                logger.info(
                    f"[USDZ Job {job_id}] Triggering Azure Container Apps Job "
                    f"(attempt {attempt}/{self._MAX_RETRIES})"
                )
                token = DefaultAzureCredential().get_token(_ARM_SCOPE).token
                response = requests.post(
                    url,
                    json=body,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=30,
                )
                if not response.ok:
                    logger.error(
                        f"[USDZ Job {job_id}] Azure API error {response.status_code}: {response.text}"
                    )
                response.raise_for_status()
                logger.info(f"[USDZ Job {job_id}] Azure Container Apps Job triggered successfully.")
                return
            except Exception as e:
                last_error = e
                logger.warning(f"[USDZ Job {job_id}] Trigger attempt {attempt} failed: {e}")
                if attempt < self._MAX_RETRIES:
                    backoff = self._BASE_BACKOFF ** attempt
                    logger.info(f"[USDZ Job {job_id}] Retrying in {backoff}s...")
                    time.sleep(backoff)

        logger.error(
            f"[USDZ Job {job_id}] All {self._MAX_RETRIES} trigger attempts failed. "
            f"Last error: {last_error}. USDZ will not be generated for this product."
        )

    async def trigger_conversion(
        self,
        glb_blob_url: str,
        product_id: str,
        user_id: str,
        product_name: Optional[str] = None,
        output_blob_name: str = "model.usdz",
    ) -> None:
        """Fire-and-forget: schedules the Azure job trigger and returns immediately."""
        if not self._is_configured():
            logger.warning(
                "USDZ trigger skipped — AZURE_SUBSCRIPTION_ID / AZURE_RESOURCE_GROUP / "
                "AZURE_JOB_NAME / AZURE_JOB_IMAGE not configured, or azure-identity not installed."
            )
            return

        job_id = str(uuid.uuid4())
        safe_name = product_name or "product"
        loop = asyncio.get_event_loop()

        sync_call = partial(
            self._trigger_sync,
            glb_blob_url=glb_blob_url,
            product_id=product_id,
            user_id=user_id,
            product_name=safe_name,
            output_blob_name=output_blob_name,
            job_id=job_id,
        )
        loop.run_in_executor(None, sync_call)
        logger.info(
            f"[USDZ Job {job_id}] Trigger scheduled (fire-and-forget). "
            f"product={product_id} glb={glb_blob_url}"
        )


usdz_trigger_service = USDZTriggerService()
