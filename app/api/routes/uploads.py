import io
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user_id
from app.core.db import get_db
from app.models.models import Job, Upload
from app.schemas.uploads import (
	BackgroundRemovalRequest,
	BackgroundRemovalResponse,
	UploadContentResponse,
	UploadInitRequest,
	UploadInitResponse,
)
from app.services.licensing_service import LicensingService
from app.services.model_converter import model_converter
from app.services.storage import storage_service
from app.utils.envelopes import api_success

router = APIRouter(tags=["uploads"])

_UPLOAD_URL_TTL_MINUTES = 60
_VALID_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_DIRECT_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".glb", ".gltf", ".usdz"}

_logger = logging.getLogger(__name__)


def _normalize_prefixed_uuid(value: Optional[str], prefix: str) -> Optional[uuid.UUID]:
	if not value:
		return None
	raw_value = value.strip()
	if raw_value.startswith(prefix):
		raw_value = raw_value[len(prefix):]
	try:
		return uuid.UUID(raw_value)
	except ValueError:
		return None


def _extract_upload_identifier(file_url: str) -> str:
	parsed = urlparse(file_url)
	path_parts = [part for part in parsed.path.split("/") if part]
	try:
		uploads_index = path_parts.index("uploads")
		upload_segment = path_parts[uploads_index + 1]
	except (ValueError, IndexError):
		upload_segment = uuid.uuid4().hex
	return f"upload-{upload_segment}"


def _validate_filename(filename: str, allowed_extensions: Optional[set[str]] = None) -> tuple[str, str]:
	name = filename.strip()
	if not name or "." not in name or len(name) > 255:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid filename")
	extension = os.path.splitext(name)[1].lower()
	if allowed_extensions is not None and extension not in allowed_extensions:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type")
	return name, extension


def _build_upload_content_response(
	upload_id: str,
	url: str,
	public_url: Optional[str],
	content_type: Optional[str],
	size_bytes: int,
	formats: Optional[Dict[str, str]] = None,
	blob_urls: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
	payload = UploadContentResponse(
		upload_id=upload_id,
		url=url,
		image_url=url,
		public_url=public_url,
		content_type=content_type,
		size_bytes=size_bytes,
		formats=formats,
		blob_urls=blob_urls,
	)
	return payload.model_dump(by_alias=True)


@router.post("/uploads/init")
async def init_upload(
	payload: UploadInitRequest,
	user_id: str = Depends(get_current_user_id),
	db: Session = Depends(get_db),
):
	filename, extension = _validate_filename(payload.filename, _VALID_UPLOAD_EXTENSIONS)

	try:
		upload_url, file_url = storage_service.create_presigned_upload(user_id=user_id, filename=filename)
	except Exception as exc:
		_logger.exception("Failed to create presigned upload for %s: %s", filename, exc)
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to initialize upload")

	rec = Upload(filename=filename, upload_url=upload_url, file_url=file_url, created_by=user_id)
	db.add(rec)

	job_uuid = _normalize_prefixed_uuid(payload.job_id, "job-")
	model_uuid = _normalize_prefixed_uuid(payload.model_id, "prod-")

	if job_uuid and model_uuid:
		job = db.query(Job).filter(Job.id == job_uuid, Job.created_by == user_id).one_or_none()
		if job is not None:
			meta = dict(getattr(job, "meta", {}) or {})
			try:
				job.modelid = model_uuid  # type: ignore[attr-defined]
				_logger.info("Linked model %s to job %s via upload init", job.modelid, job.id)
			except Exception:
				_logger.exception("Failed to set job.modelid for job %s", job.id)
			meta["modelid"] = str(model_uuid)
			job.meta = meta  # type: ignore[assignment]
			db.add(job)
			try:
				db.commit()
			except Exception:
				_logger.exception("Failed to commit job metadata update")
				db.rollback()

	try:
		db.commit()
	except Exception:
		_logger.exception("Failed to commit upload record")
		db.rollback()
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unable to persist upload")

	expires_at = datetime.utcnow() + timedelta(minutes=_UPLOAD_URL_TTL_MINUTES)
	response_payload = UploadInitResponse(
		upload_id=_extract_upload_identifier(file_url),
		upload_url=upload_url,
		image_url=file_url,
		expires_at=expires_at,
	).model_dump(by_alias=True)
	return api_success(response_payload)


@router.post("/uploads/content")
async def upload_content(
	file: UploadFile = File(...),
	user_id: str = Depends(get_current_user_id),
	db: Session = Depends(get_db),
):
	if not file.filename:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required")
	filename, extension = _validate_filename(file.filename, _DIRECT_UPLOAD_EXTENSIONS)
	logger = logging.getLogger(__name__)

	# USDZ conversion is a paid-plan feature, so only GLB uploads need the plan
	# lookup. A GLB from a non-eligible plan skips the conversion branch and is
	# stored as-is by the plain upload path below.
	is_glb = model_converter.is_glb_file(filename)
	usdz_allowed = await LicensingService.can_create_usdz(db, user_id) if is_glb else False
	if is_glb and not usdz_allowed:
		logger.info(
			"USDZ conversion skipped for %s — user %s is not on a USDZ-eligible "
			"plan. Uploading GLB only.",
			filename, user_id
		)

	if is_glb and usdz_allowed:
		try:
			glb_content = await file.read()
			glb_stream = io.BytesIO(glb_content)
			usdz_bytes, usdz_content_type = model_converter.convert_glb_to_usdz(glb_stream, filename)

			files_to_upload = [
				{"extension": "glb", "content_type": file.content_type or "model/gltf-binary", "stream": io.BytesIO(glb_content)},
				{"extension": "usdz", "content_type": usdz_content_type, "stream": io.BytesIO(usdz_bytes)},
			]

			cdn_urls, blob_urls, asset_url_base = storage_service.upload_dual_format_files(
				user_id=user_id,
				base_filename=os.path.splitext(filename)[0],
				files=files_to_upload,
			)

			glb_url, usdz_url = cdn_urls
			glb_blob_url, usdz_blob_url = blob_urls

			meta_common = {
				"original_filename": filename,
				"asset_url_base": asset_url_base,
			}

			glb_rec = Upload(
				filename=f"{os.path.splitext(filename)[0]}.glb",
				upload_url=None,
				file_url=glb_url,
				created_by=user_id,
				meta={
					**meta_common,
					"has_converted_formats": True,
					"converted_formats": ["usdz"],
					"usdz_url": usdz_url,
					"blob_url": glb_blob_url,
				},
			)
			db.add(glb_rec)

			usdz_rec = Upload(
				filename=f"{os.path.splitext(filename)[0]}.usdz",
				upload_url=None,
				file_url=usdz_url,
				created_by=user_id,
				meta={
					**meta_common,
					"converted_from": "glb",
					"source_file_url": glb_url,
					"is_converted_format": True,
					"blob_url": usdz_blob_url,
				},
			)
			db.add(usdz_rec)
			db.commit()

			logger.info("Uploaded GLB and converted USDZ for %s", filename)

			response_payload = _build_upload_content_response(
				upload_id=_extract_upload_identifier(glb_url),
				url=glb_url,
				public_url=glb_blob_url,
				content_type=file.content_type,
				size_bytes=len(glb_content),
				formats={"glb": glb_url, "usdz": usdz_url},
				blob_urls={"glb": glb_blob_url, "usdz": usdz_blob_url},
			)
			response_payload["assetUrl"] = asset_url_base  # type: ignore[index]
			response_payload["hasMultipleFormats"] = True  # type: ignore[index]
			return api_success(response_payload)

		except Exception as exc:
			logger.exception("GLB conversion failed, falling back to raw upload: %s", exc)
			stream = io.BytesIO(glb_content)
			stream.seek(0)
			file_url, blob_url = storage_service.upload_file_content(
				user_id=user_id,
				filename=filename,
				content_type=file.content_type,
				stream=stream,
			)

			rec = Upload(
				filename=filename,
				upload_url=None,
				file_url=file_url,
				created_by=user_id,
				meta={
					"conversion_attempted": True,
					"conversion_failed": True,
					"conversion_error": str(exc),
					"blob_url": blob_url,
					"original_filename": filename,
				},
			)
			db.add(rec)
			db.commit()

			response_payload = _build_upload_content_response(
				upload_id=_extract_upload_identifier(file_url),
				url=file_url,
				public_url=blob_url,
				content_type=file.content_type,
				size_bytes=len(glb_content),
				formats={"glb": file_url},
				blob_urls={"glb": blob_url} if blob_url else None,
			)
			response_payload["hasMultipleFormats"] = False  # type: ignore[index]
			response_payload["conversionStatus"] = {  # type: ignore[index]
				"usdz": {"attempted": True, "successful": False, "error": str(exc)}
			}
			return api_success(response_payload)

	content_bytes = await file.read()
	stream = io.BytesIO(content_bytes)
	stream.seek(0)

	try:
		file_url, blob_url = storage_service.upload_file_content(
			user_id=user_id,
			filename=filename,
			content_type=file.content_type,
			stream=stream,
		)
	except Exception as exc:
		logger.exception("Failed to upload file content: %s", exc)
		raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="File upload failed")

	rec = Upload(
		filename=filename,
		upload_url=None,
		file_url=file_url,
		created_by=user_id,
		meta={"blob_url": blob_url},
	)
	db.add(rec)
	db.commit()

	response_payload = _build_upload_content_response(
		upload_id=_extract_upload_identifier(file_url),
		url=file_url,
		public_url=blob_url,
		content_type=file.content_type,
		size_bytes=len(content_bytes),
		formats={"glb": file_url} if is_glb else None,
		blob_urls={"glb": blob_url} if is_glb and blob_url else None,
	)
	if is_glb and not usdz_allowed:
		# The GLB is stored and returned as normal; this tells the caller why no
		# USDZ came with it, so the UI can show an upgrade prompt.
		response_payload["hasMultipleFormats"] = False  # type: ignore[index]
		response_payload["conversionStatus"] = {  # type: ignore[index]
			"usdz": {
				"attempted": False,
				"successful": False,
				"error": None,
				"reason": "plan_not_eligible",
				"requiredPlans": list(LicensingService.USDZ_PLAN_CODES),
				"message": LicensingService.USDZ_UPGRADE_MESSAGE,
			}
		}
	return api_success(response_payload)


@router.post("/uploads/remove-background")
async def remove_background(
	payload: BackgroundRemovalRequest,
	user_id: str = Depends(get_current_user_id),
	db: Session = Depends(get_db),
):
	source_url = str(payload.image_url)
	try:
		async with httpx.AsyncClient(timeout=30.0) as client:
			resp = await client.get(source_url)
	except httpx.RequestError as exc:
		_logger.warning("Failed to download image for background removal: %s", exc)
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to download imageURL")

	if resp.status_code != 200 or not resp.content:
		_logger.warning("Background removal download returned status %s", resp.status_code)
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to download imageURL")

	image_bytes = resp.content
	original_filename = os.path.basename(urlparse(source_url).path) or "image.png"
	base_name, ext = os.path.splitext(original_filename)
	target_ext = ext.lower() if ext.lower() in {".png", ".webp"} else ".png"
	content_type = resp.headers.get("content-type", "image/png" if target_ext == ".png" else "image/webp")

	clean_filename = f"{base_name}-clean{target_ext}"
	stream = io.BytesIO(image_bytes)
	stream.seek(0)

	try:
		clean_url, clean_blob_url = storage_service.upload_file_content(
			user_id=user_id,
			filename=clean_filename,
			content_type=content_type,
			stream=stream,
		)

		rec = Upload(
			filename=clean_filename,
			upload_url=None,
			file_url=clean_url,
			created_by=user_id,
			meta={
				"blob_url": clean_blob_url,
				"source_image_url": source_url,
				"background_removed": True,
				"refine_edges": payload.refine_edges,
				"restore_shadow": payload.restore_shadow,
			},
		)
		db.add(rec)
		db.commit()
		quality_score = 0.9
	except Exception as exc:
		_logger.exception("Failed to store cleaned image, returning original: %s", exc)
		clean_url = source_url
		clean_blob_url = None
		quality_score = 0.0

	response_payload = BackgroundRemovalResponse(
		originalImageURL=source_url,
		cleanedImageURL=clean_url,
		maskURL=None,
		qualityScore=quality_score,
	).model_dump(by_alias=True)

	return api_success(response_payload)
