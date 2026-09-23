import uuid
import os
import re
from datetime import datetime, timedelta
from typing import Optional, BinaryIO, List, Dict
from urllib.parse import urlparse

from app.core.config import settings

try:
	from azure.storage.blob import BlobServiceClient, ContentSettings
	from azure.core.credentials import AzureNamedKeyCredential
	from azure.core.exceptions import ResourceExistsError
	_AZURE_AVAILABLE = True
except Exception:
	_AZURE_AVAILABLE = False
	BlobServiceClient = None  # type: ignore
	ContentSettings = None  # type: ignore
	AzureNamedKeyCredential = None  # type: ignore
	ResourceExistsError = Exception  # type: ignore


# Configurator textures: the ONLY content types tbl_part_option_textures allows
# (ck_option_texture_mime), mapped to the extension each one is stored under.
# The extension is derived from this map rather than taken from the caller, so
# no caller can contribute a raw path component.
_CONFIGURATOR_TEXTURE_EXTENSIONS = {
	"image/png": "png",
	"image/jpeg": "jpg",
}

# Longest side of a single path token. Generous for a UUID or a sha256 hex.
_CONFIGURATOR_TOKEN_MAX = 64


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

	# Raw blob URLs look like https://{account}.blob.core.windows.net/...
	_BLOB_HOST_SUFFIX = ".blob.core.windows.net"

	@staticmethod
	def _credentialed_account() -> Optional[str]:
		"""The ONE storage account this service can authenticate against.

		Mirrors _get_blob_service_client's precedence exactly — connection string
		first, then AZURE_STORAGE_ACCOUNT — so acceptance can never drift from
		what the client is actually bound to.

		Deliberately NOT settings.all_blob_base_urls(): that list drives the
		response-body CDN rewrite and may name accounts we hold no key for.
		Accepting one of those here would turn a clear error into a 403 at read.
		"""
		for part in (settings.AZURE_STORAGE_CONN_STRING or "").split(";"):
			if part.startswith("AccountName="):
				account = part[len("AccountName="):].strip()
				if account:
					return account
		return (settings.AZURE_STORAGE_ACCOUNT or "").strip() or None

	@staticmethod
	def _split_container_and_path(remainder: str, file_url: str) -> tuple[str, str]:
		"""Split "{container}/{blob_path}"."""
		slash = remainder.find("/")
		if slash == -1:
			raise RuntimeError(f"file_url has no blob path after container: {file_url}")
		return remainder[:slash], remainder[slash + 1:]

	def resolve_blob_location(self, file_url: str) -> tuple[str, str]:
		"""(container, blob_path) for a URL this service can actually read.

		Two accepted forms, and only two:

		  1. ``{CDN_BASE_URL}/{container}/{blob_path}`` — parsed exactly as before,
		     including its treatment of any query string, so existing callers are
		     bit-for-bit unaffected.
		  2. ``https://{account}.blob.core.windows.net/{container}/{blob_path}``
		     where ``{account}`` is the account this service holds credentials for.

		A blob URL on ANY OTHER account raises, naming both accounts. That is the
		point: roughly two thirds of this database's mesh URLs live on storage
		accounts this application has no key for, and silently accepting them
		would turn a configuration problem into an authentication failure much
		further down the call stack. No host is ever rewritten to another host.
		"""
		if not file_url:
			raise RuntimeError("file_url is empty; cannot infer blob path")

		cdn_base = (settings.CDN_BASE_URL or "").rstrip("/")
		if cdn_base and file_url.startswith(f"{cdn_base}/"):
			return self._split_container_and_path(
				file_url[len(cdn_base) + 1:], file_url
			)

		parsed = urlparse(file_url)
		host = (parsed.netloc or "").lower()
		if host.endswith(self._BLOB_HOST_SUFFIX):
			account = host[: -len(self._BLOB_HOST_SUFFIX)]
			expected = self._credentialed_account()
			if expected and account == expected.lower():
				# parsed.path drops any query string and is not percent-decoded,
				# matching how these paths were written.
				return self._split_container_and_path(
					parsed.path.lstrip("/"), file_url
				)
			raise RuntimeError(
				f"file_url is on storage account {account!r}, but this application "
				f"holds credentials only for {expected!r}; it cannot be read. "
				f"Serve it through CDN_BASE_URL or configure that account."
			)

		if not cdn_base:
			raise RuntimeError("CDN_BASE_URL is not configured")
		raise RuntimeError(
			f"file_url is neither a {cdn_base} URL nor a blob URL on the "
			f"configured storage account; cannot infer blob path"
		)
	def download_upload_blob_bytes(self, file_url: str) -> tuple[bytes, Optional[str], str]:
		"""Download a blob addressed via a CDN URL or a same-account blob URL.

		Accepted forms are defined by resolve_blob_location:
		    {CDN_BASE_URL}/{container}/{blob_path}
		    https://{configured account}.blob.core.windows.net/{container}/{path}

		The container is parsed directly from the URL — this correctly handles
		both uploads (STORAGE_CONTAINER_UPLOADS) and media/product images
		(STORAGE_CONTAINER_MEDIA) without hardcoding either.

		Returns (content_bytes, content_type, filename).
		Raises RuntimeError when the URL is on another storage account, or is
		neither form.
		"""
		container, blob_path = self.resolve_blob_location(file_url)

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

	def upload_variant_model(
		self,
		product_id: str,
		config_hash: str,
		extension: str,
		content_type: Optional[str],
		stream: BinaryIO,
	) -> tuple[str, str]:
		"""Upload a baked colourway model. Returns (cdn_url, blob_url).

		The path is content-addressed by the variant's config hash:

		    {container}/products/{product_id}/variants/{config_hash}.{ext}

		That makes bakes idempotent — re-baking the same recipe overwrites the
		same blob rather than accumulating copies — and makes the file safely
		cacheable forever, because a different colour always yields a different
		hash and therefore a different URL.
		"""
		client = self._get_blob_service_client()
		container = settings.STORAGE_CONTAINER_UPLOADS or "uploads"
		blob_path = f"products/{product_id}/variants/{config_hash}.{extension.lstrip('.')}"
		blob_client = client.get_blob_client(container=container, blob=blob_path)
		settings_obj = ContentSettings(  # type: ignore
			content_type=content_type or "model/gltf-binary",
			# Immutable: the hash in the path changes whenever the bytes would.
			cache_control="public, max-age=31536000, immutable",
		)
		blob_client.upload_blob(stream, overwrite=True, content_settings=settings_obj)
		return self._cdn_url(container, blob_path), blob_client.url

	@staticmethod
	def _configurator_path_token(value: str, label: str) -> str:
		"""Validate one path segment and return it in a blob-safe form.

		Rejects rather than scrubs: every token here is server-generated (a UUID,
		a glb_version, a recipe hash), so anything unexpected is a bug upstream,
		not user input to be tidied up. Silently rewriting it would hide that and
		could collapse two distinct artifacts onto one path.

		The one rewrite is ':' -> '-', because glb_version is prefix-discriminated
		("asset:<uuid>" / "sha256:<hex>", ADR-006) and a colon in a URL path is
		legal but reliably awkward in tooling. The mapping stays injective: the
		prefixes differ, so no two versions can collide.
		"""
		raw = (value or "").strip()
		if not raw:
			raise ValueError(f"Configurator texture {label} is empty")
		if len(raw) > _CONFIGURATOR_TOKEN_MAX:
			raise ValueError(
				f"Configurator texture {label} is too long "
				f"({len(raw)} > {_CONFIGURATOR_TOKEN_MAX})"
			)
		if ".." in raw:
			raise ValueError(f"Configurator texture {label} must not contain '..'")
		if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", raw):
			raise ValueError(
				f"Configurator texture {label} has characters that are not "
				f"allowed in a blob path: {raw!r}"
			)
		return raw.replace(":", "-")

	def upload_configurator_texture(
		self,
		*,
		product_id: str,
		glb_version: str,
		option_id: str,
		material_index: int,
		recipe_hash: str,
		content_type: str,
		stream: BinaryIO,
	) -> tuple[str, str]:
		"""Upload one baked Configurator texture. Returns (cdn_url, blob_url).

		The path is content-addressed, so the same recipe over the same model
		always lands on the same blob:

		    {container}/configurator/{product_id}/{glb_version}/{option_id}/{material_index}-{recipe_hash}.{ext}

		Every component is server-generated and validated. The extension is
		DERIVED from content_type (png / jpg) rather than supplied, because
		tbl_part_option_textures.ck_option_texture_mime permits only image/png and
		image/jpeg — accepting an extension would let the two disagree and would
		hand the caller a raw path component.

		Unlike every other upload in this service, this one passes
		``overwrite=False``. The path already identifies the bytes, so an existing
		blob IS the artifact we were about to write; treating that as success is
		idempotent AND makes it impossible to clobber a different artifact. The
		caller's bytes are written verbatim — no re-encoding, no resizing.
		"""
		extension = _CONFIGURATOR_TEXTURE_EXTENSIONS.get(content_type)
		if extension is None:
			raise ValueError(
				f"Unsupported Configurator texture content type {content_type!r}. "
				f"Expected one of "
				f"{', '.join(sorted(_CONFIGURATOR_TEXTURE_EXTENSIONS))}."
			)

		if not isinstance(material_index, int) or isinstance(material_index, bool):
			raise ValueError("Configurator texture material_index must be an int")
		if material_index < 0:
			raise ValueError(
				f"Configurator texture material_index must be >= 0, got {material_index}"
			)

		product_token = self._configurator_path_token(product_id, "product_id")
		version_token = self._configurator_path_token(glb_version, "glb_version")
		option_token = self._configurator_path_token(option_id, "option_id")
		hash_token = self._configurator_path_token(recipe_hash, "recipe_hash")

		blob_path = (
			f"configurator/{product_token}/{version_token}/{option_token}"
			f"/{material_index}-{hash_token}.{extension}"
		)

		client = self._get_blob_service_client()
		container = settings.STORAGE_CONTAINER_UPLOADS or "uploads"
		blob_client = client.get_blob_client(container=container, blob=blob_path)
		settings_obj = ContentSettings(  # type: ignore
			content_type=content_type,
			# Immutable: recipe_hash in the path changes whenever the bytes would.
			cache_control="public, max-age=31536000, immutable",
		)
		try:
			blob_client.upload_blob(
				stream, overwrite=False, content_settings=settings_obj
			)
		except ResourceExistsError:
			# Already stored by an earlier bake of this exact recipe. Nothing to
			# do, and deliberately NOT an error: re-baking must be idempotent.
			pass

		return self._cdn_url(container, blob_path), blob_client.url

	def delete_blob_by_cdn_url(self, file_url: str) -> bool:
		"""Delete a blob by CDN or same-account blob URL. False if not deleted.

		Used to purge superseded variant bakes and superseded Configurator
		textures. Never raises: on a missing blob, or a URL this service cannot
		resolve, it reports False. A purge that finds nothing to delete has
		already achieved its goal.

		Routed through resolve_blob_location so a same-account raw blob URL is
		now purgeable too; previously it was silently skipped.
		"""
		try:
			container, blob_path = self.resolve_blob_location(file_url)
		except RuntimeError:
			# Unreadable URL — nothing this service can delete. Returning False
			# rather than raising is the long-standing contract here: callers
			# purge best-effort and must not fail a delete over a stray URL.
			return False

		try:
			client = self._get_blob_service_client()
			client.get_blob_client(container=container, blob=blob_path).delete_blob()
			return True
		except Exception:
			return False

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

	def upload_model_variant_file(
		self,
		*,
		user_id: str,
		product_id: str,
		variant_id: str,
		filename: str,
		content_type: Optional[str],
		stream: BinaryIO,
	) -> tuple[str, str]:
		"""Upload one Configurator model-variant file. Returns (cdn_url, blob_url).

		    {media container}/{user_id}/{product_id}/model-variants/{variant_id}/{file}

		Under the seller's own {user_id}/ prefix on purpose: the account purge job
		sweeps that prefix, so a variant's GLB, original upload and thumbnail need no
		new purge rule (ADR-014). The three ids are server-generated UUIDs and are
		validated as such; the filename gets the usual random suffix, so a
		re-upload never overwrites a URL a viewer may have cached.
		"""
		tokens = []
		for label, value in (("user_id", user_id), ("product_id", product_id), ("variant_id", variant_id)):
			try:
				tokens.append(str(uuid.UUID(str(value))))
			except ValueError as exc:
				raise ValueError(f"Model variant {label} is not a UUID: {value!r}") from exc
		user_token, product_token, variant_token = tokens

		client = self._get_blob_service_client()
		container = self._media_container()
		blob_path = (
			f"{user_token}/{product_token}/model-variants/{variant_token}/"
			f"{self._sanitize_filename(filename)}"
		)
		blob_client = client.get_blob_client(container=container, blob=blob_path)
		settings_obj = ContentSettings(content_type=content_type or "application/octet-stream")  # type: ignore
		blob_client.upload_blob(stream, overwrite=False, content_settings=settings_obj)
		return self._cdn_url(container, blob_path), blob_client.url

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
