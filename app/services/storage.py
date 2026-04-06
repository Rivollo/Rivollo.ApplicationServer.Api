import uuid
import os
import re
from datetime import datetime, timedelta
from typing import Optional, BinaryIO, List, Dict

from app.core.config import settings

try:
	from azure.storage.blob import BlobServiceClient, ContentSettings
	from azure.core.credentials import AzureNamedKeyCredential
	_AZURE_AVAILABLE = True
except Exception:
	_AZURE_AVAILABLE = False
	BlobServiceClient = None  # type: ignore
	ContentSettings = None  # type: ignore
	AzureNamedKeyCredential = None  # type: ignore


class StorageService:
	def __init__(self) -> None:
		self._blob_client: Optional[BlobServiceClient] = None

	@staticmethod
	def _sanitize_filename(filename: str) -> str:
		"""Return a blob-safe filename — URL-clean and path-traversal-safe.

		Rules:
		  1. Strip any directory component (basename only) — neutralises
		     path traversal attempts like ../../etc/passwd.
		  2. Replace every character that is not alphanumeric, dash,
		     underscore, or dot with an underscore.
		  3. Strip leading dots — prevents hidden-file names and any
		     residual traversal artefacts.
		  4. Fall back to "file" if the result is empty.

		The resulting name contains only [A-Za-z0-9_.-] and never starts
		with a dot, so the blob path and its URL are identical — no
		URL-encoding is needed and external callers receive a URL they can
		use as-is.

		Examples:
		    "chair unmasked.webp"   → "chair_unmasked.webp"
		    "chair+mask.webp"       → "chair_mask.webp"
		    "my file (1).png"       → "my_file__1_.png"
		    "archive.tar.gz"        → "archive.tar.gz"
		    "normal_name.jpg"       → "normal_name.jpg"
		    "../secret.jpg"         → "secret.jpg"
		    "../../etc/passwd"      → "passwd"
		"""
		# 1. Strip directory components
		filename = os.path.basename(filename)
		# 2. Split name and extension
		name, ext = os.path.splitext(filename)
		# 3. Allow only alphanumeric characters
		sanitized = re.sub(r"[^A-Za-z0-9]", "", name)
		# 4. Strip leading dots
		sanitized = sanitized.lstrip(".")
		# 5. Fallback
		sanitized = sanitized or "file"
		# 6. Append 5-char alphanumeric suffix
		suffix = uuid.uuid4().hex[:5]
		return f"{sanitized}{suffix}{ext}"

	@staticmethod
	def _cdn_url(container: str, blob_path: str) -> str:
		"""Build a CDN URL for a blob.

		Args:
			container: The Azure Blob Storage container name (e.g. "dev", "uploads").
			blob_path: The path within the container, without leading slash.

		The resulting URL is:
		    {CDN_BASE_URL}/{container}/{blob_path}

		This mirrors the structure of the direct blob URL:
		    https://{account}.blob.core.windows.net/{container}/{blob_path}

		Raises RuntimeError when CDN_BASE_URL is not configured so that
		misconfigured environments fail fast rather than silently returning
		broken URLs to clients.
		"""
		base = (settings.CDN_BASE_URL or "").rstrip("/")
		if not base:
			raise RuntimeError(
				"CDN_BASE_URL is not configured. "
				"Set it to your Azure CDN / Front Door hostname in the environment."
			)
		return f"{base}/{container}/{blob_path}"

	def _get_blob_service_client(self) -> BlobServiceClient:
		if not _AZURE_AVAILABLE:
			raise RuntimeError("Azure SDK not available. Ensure azure-storage-blob is installed.")
		if self._blob_client is not None:
			return self._blob_client
		if settings.AZURE_STORAGE_CONN_STRING:
			self._blob_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONN_STRING)
			return self._blob_client
		if settings.AZURE_STORAGE_ACCOUNT and settings.AZURE_STORAGE_KEY:
			account_url = f"https://{settings.AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
			credential = AzureNamedKeyCredential(settings.AZURE_STORAGE_ACCOUNT, settings.AZURE_STORAGE_KEY)  # type: ignore
			self._blob_client = BlobServiceClient(account_url=account_url, credential=credential)
			return self._blob_client
		raise RuntimeError("Azure Storage is not configured. Set AZURE_STORAGE_CONN_STRING or AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_KEY.")

	def create_presigned_upload(self, user_id: str, filename: str) -> tuple[str, str]:
		container = settings.STORAGE_CONTAINER_UPLOADS or "uploads"
		upload_id = str(uuid.uuid4())
		blob_path = f"users/{user_id}/uploads/{upload_id}/{self._sanitize_filename(filename)}"
		cdn_file_url = self._cdn_url(container, blob_path)

		# Build a real SAS URL for PUT to the Azure Blob endpoint
		client = self._get_blob_service_client()
		blob_client = client.get_blob_client(container=container, blob=blob_path)

		# Generate SAS using SDK helper if available; otherwise raise a clear error
		try:
			from azure.storage.blob import generate_blob_sas, BlobSasPermissions  # type: ignore
			from azure.storage.blob import ResourceTypes  # type: ignore
		except Exception:
			raise RuntimeError("Azure SDK does not expose SAS helpers. Ensure azure-storage-blob is installed.")

		expiry_time = datetime.utcnow() + timedelta(minutes=60)
		starts_on = datetime.utcnow()

		account_name = getattr(client, "account_name", None)  # type: ignore
		sas_token: str
		account_key = None
		if settings.AZURE_STORAGE_KEY:
			account_key = settings.AZURE_STORAGE_KEY
		else:
			account_key = getattr(getattr(client, "credential", object()), "account_key", None)  # type: ignore

		if account_key:
			# Use account key SAS
			sas_token = generate_blob_sas(
				account_name=account_name,
				container_name=container,
				blob_name=blob_path,
				account_key=account_key,
				permission=BlobSasPermissions(write=True, create=True),
				expiry=expiry_time,
				start=starts_on,
			)
		else:
			# Fallback: use user delegation SAS (Managed Identity / AAD)
			try:
				udk = client.get_user_delegation_key(starts_on=starts_on, expires_on=expiry_time)  # type: ignore
				sas_token = generate_blob_sas(
					account_name=account_name,
					container_name=container,
					blob_name=blob_path,
					user_delegation_key=udk,
					permission=BlobSasPermissions(write=True, create=True),
					expiry=expiry_time,
					start=starts_on,
				)
			except Exception as ex:
				raise RuntimeError("Unable to generate SAS: set AZURE_STORAGE_KEY or grant Managed Identity Blob Data Contributor.") from ex

		upload_url = f"{blob_client.url}?{sas_token}"
		return upload_url, cdn_file_url

	def upload_file_content(self, user_id: str, filename: str, content_type: Optional[str], stream: BinaryIO) -> tuple[str, str]:
		"""Upload file content and return (cdn_url, blob_url)."""
		client = self._get_blob_service_client()
		container = settings.STORAGE_CONTAINER_UPLOADS or "uploads"
		upload_id = str(uuid.uuid4())
		blob_path = f"users/{user_id}/uploads/{upload_id}/{self._sanitize_filename(filename)}"
		blob_client = client.get_blob_client(container=container, blob=blob_path)
		settings_obj = ContentSettings(content_type=content_type or "application/octet-stream")  # type: ignore
		blob_client.upload_blob(stream, overwrite=True, content_settings=settings_obj)

		cdn_url = self._cdn_url(container, blob_path)
		blob_url = blob_client.url
		return cdn_url, blob_url

	def upload_asset_file(self, user_id: str, asset_id: str, file_extension: str, content_type: Optional[str], stream: BinaryIO) -> tuple[str, str]:
		"""Upload asset file and return (cdn_url, blob_url)."""
		client = self._get_blob_service_client()
		container = settings.STORAGE_CONTAINER_UPLOADS or "uploads"
		blob_path = f"users/{user_id}/models/{asset_id}.{file_extension}"
		blob_client = client.get_blob_client(container=container, blob=blob_path)
		settings_obj = ContentSettings(content_type=content_type or "application/octet-stream")  # type: ignore
		blob_client.upload_blob(stream, overwrite=True, content_settings=settings_obj)

		cdn_url = self._cdn_url(container, blob_path)
		blob_url = blob_client.url
		return cdn_url, blob_url

	def download_upload_blob_bytes(self, file_url: str) -> tuple[bytes, Optional[str], str]:
		"""Download a blob addressed via a CDN URL.

		The CDN URL has the form:
		    {CDN_BASE_URL}/{container}/{blob_path}

		The container is parsed directly from the URL — this correctly handles
		both uploads (STORAGE_CONTAINER_UPLOADS) and media/product images
		(STORAGE_CONTAINER_MEDIA) without hardcoding either.

		Returns (content_bytes, content_type, filename).
		Raises RuntimeError if CDN_BASE_URL is not set or the URL doesn't match.
		"""
		cdn_base = (settings.CDN_BASE_URL or "").rstrip("/")
		if not cdn_base:
			raise RuntimeError("CDN_BASE_URL is not configured")

		cdn_prefix = f"{cdn_base}/"
		if not file_url.startswith(cdn_prefix):
			raise RuntimeError(
				f"file_url does not start with CDN_BASE_URL ({cdn_base}); cannot infer blob path"
			)

		# Remainder is "{container}/{blob_path}"
		remainder = file_url[len(cdn_prefix):]
		slash = remainder.find("/")
		if slash == -1:
			raise RuntimeError(f"file_url has no blob path after container: {file_url}")

		container = remainder[:slash]
		blob_path = remainder[slash + 1:]

		client = self._get_blob_service_client()
		blob_client = client.get_blob_client(container=container, blob=blob_path)
		downloader = blob_client.download_blob()
		content_bytes = downloader.readall()
		content_type: Optional[str] = None
		try:
			props = blob_client.get_blob_properties()
			content_type = getattr(getattr(props, "content_settings", None), "content_type", None)  # type: ignore
		except Exception:
			pass
		filename = os.path.basename(blob_path)
		return content_bytes, content_type, filename

	def upload_dual_format_files(self, user_id: str, base_filename: str, files: List[Dict[str, any]]) -> tuple[List[str], List[str], str]:
		"""Upload multiple files with the same base name but different extensions.

		Args:
			user_id: User ID
			base_filename: Base filename without extension
			files: List of dicts with 'extension', 'content_type', 'stream' keys

		Returns:
			Tuple of (cdn_urls, blob_urls, asset_url_without_extension)
		"""
		client = self._get_blob_service_client()
		container = settings.STORAGE_CONTAINER_UPLOADS or "uploads"
		upload_id = str(uuid.uuid4())

		# Sanitize base_filename once so all files and the base URL share the same suffix
		sanitized_base = self._sanitize_filename(base_filename)

		cdn_urls = []
		blob_urls = []
		for file_info in files:
			extension = file_info['extension']
			content_type = file_info['content_type']
			stream = file_info['stream']

			blob_path = f"users/{user_id}/uploads/{upload_id}/{sanitized_base}.{extension}"
			blob_client = client.get_blob_client(container=container, blob=blob_path)
			settings_obj = ContentSettings(content_type=content_type or "application/octet-stream")  # type: ignore
			blob_client.upload_blob(stream, overwrite=True, content_settings=settings_obj)

			cdn_urls.append(self._cdn_url(container, blob_path))
			blob_urls.append(blob_client.url)

		# Base URL for the asset (no extension) — same container + path prefix
		asset_url_base = self._cdn_url(container, f"users/{user_id}/uploads/{upload_id}/{sanitized_base}")

		return cdn_urls, blob_urls, asset_url_base

	def upload_dual_asset_files(self, user_id: str, asset_id: str, base_name: str, files: List[Dict[str, any]]) -> tuple[List[str], List[str], str]:
		"""Upload multiple asset files with the same base name but different extensions.

		Args:
			user_id: User ID
			asset_id: Asset ID
			base_name: Base name for the files
			files: List of dicts with 'extension', 'content_type', 'stream' keys

		Returns:
			Tuple of (cdn_urls, blob_urls, asset_url_without_extension)
		"""
		client = self._get_blob_service_client()
		container = settings.STORAGE_CONTAINER_UPLOADS or "uploads"

		cdn_urls = []
		blob_urls = []
		for file_info in files:
			extension = file_info['extension']
			content_type = file_info['content_type']
			stream = file_info['stream']

			blob_path = f"users/{user_id}/models/{asset_id}_{base_name}.{extension}"
			blob_client = client.get_blob_client(container=container, blob=blob_path)
			settings_obj = ContentSettings(content_type=content_type or "application/octet-stream")  # type: ignore
			blob_client.upload_blob(stream, overwrite=True, content_settings=settings_obj)

			cdn_urls.append(self._cdn_url(container, blob_path))
			blob_urls.append(blob_client.url)

		asset_url_base = self._cdn_url(container, f"users/{user_id}/models/{asset_id}_{base_name}")

		return cdn_urls, blob_urls, asset_url_base

	def _media_container(self) -> str:
		"""Resolve the container for product/background images."""
		return settings.STORAGE_CONTAINER_MEDIA or settings.STORAGE_CONTAINER_UPLOADS or "uploads"

	def upload_product_image(self, user_id: str, product_id: str, filename: str, content_type: Optional[str], stream: BinaryIO) -> tuple[str, str]:
		"""Upload product image. Returns (cdn_url, blob_url)."""
		client = self._get_blob_service_client()
		container = self._media_container()
		blob_path = f"{user_id}/{product_id}/{self._sanitize_filename(filename)}"
		blob_client = client.get_blob_client(container=container, blob=blob_path)
		settings_obj = ContentSettings(content_type=content_type or "application/octet-stream")  # type: ignore
		blob_client.upload_blob(stream, overwrite=True, content_settings=settings_obj)
		cdn_url = self._cdn_url(container, blob_path)
		blob_url = blob_client.url
		return cdn_url, blob_url

	def upload_background_image(self, user_id: str, product_id: str, filename: str, content_type: Optional[str], stream: BinaryIO) -> tuple[str, str]:
		"""Upload background image. Returns (cdn_url, blob_url)."""
		client = self._get_blob_service_client()
		container = self._media_container()
		blob_path = f"{user_id}/{product_id}/backgrounds/{self._sanitize_filename(filename)}"
		blob_client = client.get_blob_client(container=container, blob=blob_path)
		settings_obj = ContentSettings(content_type=content_type or "application/octet-stream")  # type: ignore
		blob_client.upload_blob(stream, overwrite=True, content_settings=settings_obj)
		cdn_url = self._cdn_url(container, blob_path)
		blob_url = blob_client.url
		return cdn_url, blob_url


storage_service = StorageService()
