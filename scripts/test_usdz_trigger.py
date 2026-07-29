"""
Manual end-to-end test for the USDZ trigger — WITHOUT calling fal.ai.

Point it at a GLB that already exists in Azure Blob Storage (e.g. one from a
product you created earlier) and it will start the Azure Container Apps Job
exactly the way /createProductFal does, then report success/failure.

Unlike the real code path (fire-and-forget via run_in_executor), this script
calls the trigger SYNCHRONOUSLY so you can see the outcome before it exits.

Usage (from repo root, with the venv active and .env populated):

    python -m scripts.test_usdz_trigger \
        --glb-url "https://dev-cdn-....azurefd.net/dev/<user>/<product>/model<suffix>.glb" \
        --product-id "<existing product uuid>" \
        --user-id "<existing user uuid>" \
        --product-name "My Test Product"

Notes:
  * --glb-url accepts EITHER the CDN url (what is stored in the DB) OR the
    direct blob url. If you pass a CDN url it is auto-converted to the direct
    blob url, because the container downloads via the Azure Blob SDK.
  * Use a REAL product-id/user-id so the container's DB write lands on an
    existing row.
  * Requires the 4 AZURE_JOB_* settings + a service principal
    (AZURE_CLIENT_ID / AZURE_CLIENT_SECRET / AZURE_TENANT_ID) that has
    permission to start the job (Microsoft.App/jobs/start/action).
"""

import argparse
import logging
import uuid

from app.core.config import settings
from app.services.usdz_trigger_service import usdz_trigger_service


def _to_blob_url(glb_url: str) -> str:
    """Convert a CDN url to the direct blob url if needed.

    CDN:  {CDN_BASE_URL}/{container}/{path}
    Blob: {AZURE_BLOB_BASE_URL}/{container}/{path}
    They share the same /{container}/{path} tail, so this is a host swap.
    """
    cdn_base = (settings.CDN_BASE_URL or "").rstrip("/")
    blob_base = (settings.AZURE_BLOB_BASE_URL or "").rstrip("/")

    if cdn_base and glb_url.startswith(cdn_base + "/"):
        if not blob_base:
            raise SystemExit(
                "Cannot convert CDN url to blob url: AZURE_BLOB_BASE_URL is not set "
                "(and could not be derived from AZURE_STORAGE_ACCOUNT / conn string)."
            )
        tail = glb_url[len(cdn_base):]  # keeps leading '/'
        return f"{blob_base}{tail}"

    # Already a blob url (or some other absolute url) — pass through unchanged.
    return glb_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Fire the USDZ converter job for an existing GLB.")
    parser.add_argument("--glb-url", required=True, help="CDN or direct blob url of an existing GLB.")
    parser.add_argument("--product-id", required=True, help="Existing product UUID.")
    parser.add_argument("--user-id", required=True, help="Existing user UUID.")
    parser.add_argument("--product-name", default="USDZ Trigger Test", help="Product name.")
    parser.add_argument("--output-blob-name", default="model.usdz", help="Output USDZ blob name.")
    args = parser.parse_args()

    # Surface the [USDZ Job ...] logs on the console.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not usdz_trigger_service._is_configured():
        raise SystemExit(
            "Trigger is NOT configured. Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, "
            "AZURE_JOB_NAME, AZURE_JOB_IMAGE in .env (and ensure azure-identity is installed)."
        )

    glb_blob_url = _to_blob_url(args.glb_url)
    job_id = str(uuid.uuid4())

    print("\n--- USDZ trigger test -------------------------------------------")
    print(f"  job-id           : {job_id}")
    print(f"  glb (blob) url   : {glb_blob_url}")
    print(f"  product-id       : {args.product_id}")
    print(f"  user-id          : {args.user_id}")
    print(f"  product-name     : {args.product_name}")
    print(f"  output-blob-name : {args.output_blob_name}")
    print(f"  job              : {settings.AZURE_JOB_NAME} (rg={settings.AZURE_RESOURCE_GROUP})")
    print("-----------------------------------------------------------------\n")

    # Call the SYNC path directly so we block and see the outcome (the real code
    # path schedules this on an executor and returns immediately).
    usdz_trigger_service._trigger_sync(
        glb_blob_url=glb_blob_url,
        product_id=args.product_id,
        user_id=args.user_id,
        product_name=args.product_name,
        output_blob_name=args.output_blob_name,
        job_id=job_id,
    )

    print(
        "\nDone. If you saw 'triggered successfully' above, check the job execution "
        "in Azure Portal (Container Apps Job -> execution history), then verify the "
        f"USDZ blob and its DB url for product {args.product_id}.\n"
    )


if __name__ == "__main__":
    main()
