import uuid
import os
from urllib.parse import quote
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
		upload_id = str(uuid.uuid4())
		blob_path = f"users/{user_id}/uploads/{upload_id}/{quote(filename)}"
		cdn_file_url = f"{settings.CDN_BASE_URL}/{blob_path}"

		# Build a real SAS URL for PUT to the Azure Blob endpoint
		client = self._get_blob_service_client()
		container = settings.STORAGE_CONTAINER_UPLOADS or "uploads"
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
		"""Upload file content and return both CDN URL and blob URL."""
		client = self._get_blob_service_client()
		container = settings.STORAGE_CONTAINER_UPLOADS or "uploads"
		upload_id = str(uuid.uuid4())
		blob_path = f"users/{user_id}/uploads/{upload_id}/{quote(filename)}"
		blob_client = client.get_blob_client(container=container, blob=blob_path)
		settings_obj = ContentSettings(content_type=content_type or "application/octet-stream")  # type: ignore
		blob_client.upload_blob(stream, overwrite=True, content_settings=settings_obj)
		
		cdn_url = f"{settings.CDN_BASE_URL}/{blob_path}"
		blob_url = blob_client.url
		return cdn_url, blob_url

	def upload_asset_file(self, user_id: str, asset_id: str, file_extension: str, content_type: Optional[str], stream: BinaryIO) -> tuple[str, str]:
		"""Upload asset file and return both CDN URL and blob URL."""
		client = self._get_blob_service_client()
		container = settings.STORAGE_CONTAINER_UPLOADS or "uploads"
		blob_path = f"users/{user_id}/models/{asset_id}.{file_extension}"
		blob_client = client.get_blob_client(container=container, blob=blob_path)
		settings_obj = ContentSettings(content_type=content_type or "application/octet-stream")  # type: ignore
		blob_client.upload_blob(stream, overwrite=True, content_settings=settings_obj)
		
		cdn_url = f"{settings.CDN_BASE_URL}/{blob_path}"
		blob_url = blob_client.url
		return cdn_url, blob_url

	def download_upload_blob_bytes(self, file_url: str) -> tuple[bytes, Optional[str], str]:
		"""Download a blob that was previously addressed via CDN_BASE_URL/users/... path.

		Returns a tuple of (content_bytes, content_type, filename).
		Raises RuntimeError if Azure is not configured or the URL doesn't match CDN_BASE_URL.
		"""
		base = (settings.CDN_BASE_URL or "").rstrip("/")
		if not base:
			raise RuntimeError("CDN_BASE_URL is not configured")
		prefix = f"{base}/"
		if not file_url.startswith(prefix):
			raise RuntimeError("file_url does not match CDN_BASE_URL; cannot infer blob path")
		blob_path = file_url[len(prefix):]
		container = settings.STORAGE_CONTAINER_UPLOADS or "uploads"

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
		
		cdn_urls = []
		blob_urls = []
		for file_info in files:
			extension = file_info['extension']
			content_type = file_info['content_type']
			stream = file_info['stream']
			
			filename = f"{base_filename}.{extension}"
			blob_path = f"users/{user_id}/uploads/{upload_id}/{quote(filename)}"
			blob_client = client.get_blob_client(container=container, blob=blob_path)
			settings_obj = ContentSettings(content_type=content_type or "application/octet-stream")  # type: ignore
			blob_client.upload_blob(stream, overwrite=True, content_settings=settings_obj)
			
			cdn_url = f"{settings.CDN_BASE_URL}/{blob_path}"
			blob_url = blob_client.url
			cdn_urls.append(cdn_url)
			blob_urls.append(blob_url)
		
		# Asset URL without extension
		asset_url_base = f"{settings.CDN_BASE_URL}/users/{user_id}/uploads/{upload_id}/{quote(base_filename)}"
		
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
			
			cdn_url = f"{settings.CDN_BASE_URL}/{blob_path}"
			blob_url = blob_client.url
			cdn_urls.append(cdn_url)
			blob_urls.append(blob_url)
		
		# Asset URL without extension
		asset_url_base = f"{settings.CDN_BASE_URL}/users/{user_id}/models/{asset_id}_{base_name}"
		
		return cdn_urls, blob_urls, asset_url_base

	def _media_container(self) -> str:
		"""Resolve the container for product/background images."""
		return settings.STORAGE_CONTAINER_MEDIA or settings.STORAGE_CONTAINER_UPLOADS or "uploads"

	def upload_product_image(self, user_id: str, product_id: str, filename: str, content_type: Optional[str], stream: BinaryIO) -> tuple[str, str]:
		"""Upload product image with path structure: {userId}/{productId}/filename

		Args:
			user_id: User ID
			product_id: Product ID
			filename: Image filename
			content_type: Content type of the image
			stream: Binary stream of the image

		Returns:
			Tuple of (cdn_url, blob_url)
		"""
		client = self._get_blob_service_client()
		container = self._media_container()
		blob_path = f"{user_id}/{product_id}/{quote(filename)}"
		blob_client = client.get_blob_client(container=container, blob=blob_path)
		settings_obj = ContentSettings(content_type=content_type or "application/octet-stream")  # type: ignore
		blob_client.upload_blob(stream, overwrite=True, content_settings=settings_obj)
		cdn_url = f"{settings.CDN_BASE_URL}/{blob_path}"
		blob_url = blob_client.url
		return cdn_url, blob_url

	def upload_background_image(self, user_id: str, product_id: str, filename: str, content_type: Optional[str], stream: BinaryIO) -> tuple[str, str]:
		"""Upload background image with path structure: {userId}/{productId}/backgrounds/filename

		Args:
			user_id: User ID
			product_id: Product ID
			filename: Image filename
			content_type: Content type of the image
			stream: Binary stream of the image

		Returns:
			Tuple of (cdn_url, blob_url)
		"""
		client = self._get_blob_service_client()
		container = self._media_container()
		blob_path = f"{user_id}/{product_id}/backgrounds/{quote(filename)}"
		blob_client = client.get_blob_client(container=container, blob=blob_path)
		settings_obj = ContentSettings(content_type=content_type or "application/octet-stream")  # type: ignore
		blob_client.upload_blob(stream, overwrite=True, content_settings=settings_obj)
		cdn_url = f"{settings.CDN_BASE_URL}/{blob_path}"
		blob_url = blob_client.url
		return cdn_url, blob_url


storage_service = StorageService()
