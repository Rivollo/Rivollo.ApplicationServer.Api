from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
import uuid
import logging
import os
from urllib.parse import urlparse

import httpx
import io
import mimetypes

from app.api.deps import get_current_user_id
from app.core.db import get_db
from app.core.config import settings
from app.models.models import Job, JobStatusEnum, Asset, AssetPart
from app.schemas.jobs import CreateJobRequest, JobStatusResponse, CreateJobResponse
from app.utils.envelopes import api_success
from app.services.storage import storage_service
from app.services.model_converter import model_converter

router = APIRouter(tags=["jobs"])


@router.get("/jobs/debug/test")
def debug_test_endpoint():
	"""Debug endpoint to test response format"""
	logger = logging.getLogger(__name__)
	logger.info("=== DEBUG TEST ENDPOINT CALLED ===")
	
	test_data = {
		"test": "response",
		"timestamp": "2024-01-01T12:00:00Z",
		"message": "This is a test response to verify API format"
	}
	
	response = api_success(test_data)
	logger.info("=== DEBUG TEST RESPONSE: %s ===", response)
	
	return response


@router.get("/jobs/debug/inference/{provider_uid}")
def debug_inference_endpoint(provider_uid: str):
	"""Debug endpoint to test inference server directly"""
	logger = logging.getLogger(__name__)
	logger.info("=== DEBUG INFERENCE ENDPOINT CALLED for UID: %s ===", provider_uid)
	
	inference_url = f"http://74.225.34.67:8081/status/{provider_uid}"
	logger.info("Testing inference server URL: %s", inference_url)
	
	try:
		with httpx.Client(timeout=httpx.Timeout(200.0)) as client:
			resp = client.get(inference_url)
			logger.info("Raw inference response: status=%s, headers=%s", resp.status_code, dict(resp.headers))
			logger.info("Raw inference response body (first 1000 chars): %s", resp.text[:1000])
			
			result = {
				"inference_url": inference_url,
				"status_code": resp.status_code,
				"headers": dict(resp.headers),
				"response_text": resp.text,
				"response_length": len(resp.text) if resp.text else 0
			}
			
			try:
				result["response_json"] = resp.json()
			except Exception as json_error:
				result["json_parse_error"] = str(json_error)
			
			logger.info("=== DEBUG INFERENCE RESULT: %s ===", result)
			return api_success(result)
			
	except Exception as e:
		logger.error("=== DEBUG INFERENCE ERROR: %s ===", str(e))
		return api_success({
			"error": str(e),
			"inference_url": inference_url
		})


def _job_public_id(job_id: uuid.UUID) -> str:
	return f"job-{job_id}"


def _parse_job_id(raw_id: str) -> uuid.UUID:
	value = raw_id
	if value.startswith("job-"):
		value = value[4:]
	return uuid.UUID(value)


async def _process_completed_job(job: Job, resp, user_id: str, db: AsyncSession, logger):
	"""Process a completed job by uploading the asset and creating database records"""
	try:
		# Create asset record first
		asset = Asset(
			title=f"Generated Model - {job.id}",
			source_image_url=job.image_url,
			created_from_job=job.id,
			created_by=user_id
		)
		db.add(asset)
		await db.flush()  # Get the asset ID
		
		# Determine file extension from content type
		content_type = resp.headers.get("content-type", "").lower()
		original_extension = None
		part_name = "model"
		
		if "glb" in content_type or "gltf-binary" in content_type:
			original_extension = "glb"
		elif "gltf" in content_type:
			original_extension = "gltf"
		else:
			original_extension = "glb"  # Default to GLB
		
		# Convert GLB to USDZ and upload both formats if it's a GLB file
		if original_extension == "glb":
			try:
				# Convert GLB to USDZ
				glb_stream = io.BytesIO(resp.content)
				usdz_bytes, usdz_content_type = model_converter.convert_glb_to_usdz(
					glb_stream, f"model.{original_extension}"
				)
				
				# Prepare both files for upload
				files_to_upload = [
					{
						'extension': 'glb',
						'content_type': content_type or 'model/gltf-binary',
						'stream': io.BytesIO(resp.content)
					},
					{
						'extension': 'usdz',
						'content_type': usdz_content_type,
						'stream': io.BytesIO(usdz_bytes)
					}
				]
				
				# Upload both files to storage
				cdn_urls, blob_urls, asset_url_base = storage_service.upload_dual_asset_files(
					user_id=user_id,
					asset_id=str(asset.id),
					base_name="model",
					files=files_to_upload
				)
				
				glb_url, usdz_url = cdn_urls
				glb_blob_url, usdz_blob_url = blob_urls
				
				# Create asset part record for GLB
				glb_asset_part = AssetPart(
					asset_id=asset.id,
					part_name="model_glb",
					url=glb_url,
					blob_url=glb_blob_url,
					mime_type=content_type or "model/gltf-binary",
					size_bytes=len(resp.content),
					position=0,
					meta={
						"format": "glb",
						"has_converted_formats": True,
						"converted_formats": ["usdz"],
						"asset_url_base": asset_url_base,
					},
				)
				db.add(glb_asset_part)
				# Create asset part record for USDZ
				usdz_asset_part = AssetPart(
					asset_id=asset.id,
					part_name="model_usdz",
					url=usdz_url,
					blob_url=usdz_blob_url,
					mime_type=usdz_content_type,
					size_bytes=len(usdz_bytes),
					position=1,
					meta={
						"format": "usdz",
						"converted_from": "glb",
						"asset_url_base": asset_url_base,
					},
				)
				db.add(usdz_asset_part)
				
				logger.info(
					"Successfully converted GLB model to USDZ for job %s. GLB URL: %s, USDZ URL: %s",
					job.id, glb_url, usdz_url
				)
				
				# Update job with primary asset information
				job.asset_id = asset.id
				job.status = JobStatusEnum.ready
				db.add(job)
				
				# Commit all changes
				await db.commit()
				await db.refresh(asset)
				await db.refresh(glb_asset_part)
				await db.refresh(usdz_asset_part)
				await db.refresh(job)
				
				logger.info(
					"Successfully processed completed job %s, created asset %s with GLB URL %s and USDZ URL %s",
					job.id, asset.id, glb_url, usdz_url
				)
				
				return api_success({
					"id": _job_public_id(job.id),
					"status": job.status.value,
					"assetId": str(asset.id),
					"glburl": glb_url,
					"usdzURL": usdz_url,
					"conversionStatus": {
						"usdz": {
							"attempted": True,
							"successful": True,
							"error": None,
						}
					},
				})
				
			except Exception as e:
				logger.warning(
					"Failed to convert GLB to USDZ for job %s: %s. Uploading GLB only.",
					job.id, str(e)
				)
				# Fall back to uploading only the original GLB file
				asset_stream = io.BytesIO(resp.content)
				file_url, blob_url = storage_service.upload_asset_file(
					user_id=user_id,
					asset_id=str(asset.id),
					file_extension=original_extension,
					content_type=content_type or "model/gltf-binary",
					stream=asset_stream
				)
				
				# Create asset part record with conversion failure info
				asset_part = AssetPart(
					asset_id=asset.id,
					part_name=part_name,
					url=file_url,
					blob_url=blob_url,
					mime_type=content_type or "model/gltf-binary",
					size_bytes=len(resp.content),
					position=0,
					meta={
						"format": original_extension,
						"conversion_attempted": True,
						"conversion_failed": True,
						"conversion_error": str(e),
					},
				)
				db.add(asset_part)
				
				# Update job with asset_id and status (for failed conversion fallback)
				job.asset_id = asset.id
				job.status = JobStatusEnum.ready
				db.add(job)
				
				# Commit all changes
				await db.commit()
				await db.refresh(asset)
				await db.refresh(asset_part)
				await db.refresh(job)

				logger.info(
					"Successfully processed job %s with GLB fallback (USDZ conversion failed). GLB URL: %s",
					job.id, file_url
				)
				
				return api_success({
					"id": _job_public_id(job.id),
					"status": job.status.value,
					"assetId": str(asset.id),
					"glburl": file_url,
					"usdzURL": None,
					"conversionStatus": {
						"usdz": {
							"attempted": True,
							"successful": False,
							"error": str(e),
						}
					},
				})
		else:
			# Handle non-GLB files normally (GLTF, etc.)
			asset_stream = io.BytesIO(resp.content)
			file_url, blob_url = storage_service.upload_asset_file(
				user_id=user_id,
				asset_id=str(asset.id),
				file_extension=original_extension,
				content_type=content_type or "model/gltf-binary",
				stream=asset_stream
			)
			
			# Create asset part record
			asset_part = AssetPart(
				asset_id=asset.id,
				part_name=part_name,
				url=file_url,
				blob_url=blob_url,
				mime_type=content_type or "model/gltf-binary",
				size_bytes=len(resp.content),
				position=0,
				meta={"format": original_extension},
			)
			db.add(asset_part)
		
			# Update job with asset_id and status (for non-GLB files)
			job.asset_id = asset.id
			job.status = JobStatusEnum.ready
			db.add(job)
			
			# Commit all changes
			await db.commit()
			await db.refresh(asset)
			await db.refresh(asset_part)
			await db.refresh(job)

			logger.info("Successfully processed completed job %s, created asset %s with streaming URL %s", 
					   job.id, asset.id, file_url)
			
			return api_success({
				"id": _job_public_id(job.id),
				"status": job.status.value,
				"assetId": str(asset.id),
				"glburl": file_url,
				"usdzURL": None,
			})
		
	except Exception as ex:
		logger.exception("Failed to process completed job %s", job.id)
		# Mark job as failed
		try:
			job.status = JobStatusEnum.failed
			job.error_message = "Failed to process completed asset"
			db.add(job)
			await db.commit()
		except Exception:
			logger.exception("Failed to mark job as failed")
		
		raise HTTPException(
			status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
			detail="Failed to process completed asset"
		) from ex


@router.post("/jobs")
async def create_job(payload: CreateJobRequest, user_id: str = Depends(get_current_user_id), db: AsyncSession = Depends(get_db)):
    logger = logging.getLogger(__name__)
    image_url = str(payload.imageURL)
    if not image_url.startswith("http"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid imageURL")

    logger.info("Create job requested by user %s for imageURL=%s", user_id, image_url)

    # Create the DB job immediately with status=queued (created)
    job = Job(image_url=image_url, status=JobStatusEnum.queued, created_by=user_id)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    async def _fail_job(error_message: str) -> None:
        try:
            job.status = JobStatusEnum.failed
            job.error_message = error_message
            db.add(job)
            await db.commit()
        except Exception:
            logger.exception("Failed to mark job as failed in DB")

    # 1) Download the image from the provided URL (prefer Azure blob if URL matches our CDN)
    try:
        image_bytes: bytes
        content_type: str
        filename: str
        base = settings.CDN_BASE_URL.rstrip("/") if settings.CDN_BASE_URL else ""
        if base and image_url.startswith(f"{base}/"):
            image_bytes, ct, filename = storage_service.download_upload_blob_bytes(image_url)
            content_type = ct or "application/octet-stream"
        else:
            with httpx.Client(timeout=httpx.Timeout(200.0)) as client:
                resp = client.get(image_url)
                if resp.status_code != 200:
                    logger.warning("Failed to download image: status=%s url=%s", resp.status_code, image_url)
                    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to download imageURL")
                image_bytes = resp.content
                content_type = resp.headers.get("content-type", "application/octet-stream")
                parsed = urlparse(image_url)
                filename = os.path.basename(parsed.path) or "image.png"
    except HTTPException as ex:
        await _fail_job("Unable to download imageURL")
        raise ex
    except Exception as ex:
        logger.exception("Error downloading image from %s", image_url)
        await _fail_job("Unable to download imageURL")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to download imageURL") from ex

    # 2) Send to inference server as multipart/form-data
    inference_url = settings.MODEL_SERVICE_URL if str(settings.MODEL_SERVICE_URL).startswith("http") else "http://74.225.34.67:8081/send"
    form_data = {
        "texture": "true",
        "type": "glb",
        "face_count": "10000",
        "octree_resolution": "128",
        "num_inference_steps": "5",
        "guidance_scale": "5.0",
        "mc_algo": "mc",
    }

    # Improve content-type detection from filename if missing/unknown
    if not content_type or content_type == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            content_type = guessed

    # Validate we actually have content
    if not image_bytes or len(image_bytes) == 0:
        logger.warning("Downloaded image has zero bytes; url=%s filename=%s", image_url, filename)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Downloaded image is empty")

    # Persist a copy to the local Downloads folder and use that file for upload
    try:
        downloads_dir = os.path.expanduser("~/Downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        local_path = os.path.join(downloads_dir, filename)
        if os.path.exists(local_path):
            name, ext = os.path.splitext(filename)
            local_path = os.path.join(downloads_dir, f"{name}-{uuid.uuid4().hex}{ext}")
        with open(local_path, "wb") as out_f:
            out_f.write(image_bytes)
        logger.info("Saved downloaded image to %s (%d bytes, content_type=%s)", local_path, len(image_bytes), content_type)
    except Exception:
        logger.exception("Failed to persist image to Downloads folder")
        await _fail_job("Failed to store image locally")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to store image locally")

    # Log the outgoing HTTP request details (without dumping binary content)
    logger.info(
        "Inference HTTP request: POST %s | headers=%s | form_fields=%s | file(name=%s, path=%s, content_type=%s, size_bytes=%d)",
        inference_url,
        {"Accept": "application/json"},
        form_data,
        filename,
        local_path,
        content_type,
        len(image_bytes),
    )

    try:
        with httpx.Client(timeout=httpx.Timeout(200.0)) as client:
            with open(local_path, "rb") as f:
                files = {"image": (filename, f, content_type)}
                req = client.build_request(
                    "POST",
                    inference_url,
                    data=form_data,
                    files=files,
                    headers={"Accept": "application/json"},
                )
                # Log full prepared request headers including multipart boundary
                logger.info("Inference HTTP prepared headers: %s", dict(req.headers))
                r = client.send(req)
            if r.status_code >= 400:
                logger.warning("Inference server error status=%s body=%s", r.status_code, r.text)
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Inference server error")
            data = r.json()
            uid = data.get("uid")
            if not uid:
                logger.warning("Inference server did not return uid: body=%s", r.text)
                raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Invalid response from inference server")
            logger.info("Inference HTTP response: status=%s uid=%s", r.status_code, uid)
    except HTTPException as ex:
        await _fail_job("Inference server error")
        raise ex
    except Exception as ex:
        logger.exception("Error sending image to inference server")
        await _fail_job("Inference server error")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Inference server error") from ex

    # 3) Store model job id in physical column (if present) and meta for redundancy
    try:
        model_uuid = uuid.UUID(str(uid))
        logger.info("Parsed provider uid as UUID: raw=%s parsed=%s", uid, model_uuid)
    except Exception:
        model_uuid = uuid.uuid4()
        logger.warning("Provider uid is not a UUID: raw=%s. Generated fallback UUID=%s", uid, model_uuid)
    try:
        job.modelid = model_uuid
        logger.info("Setting job.modelid=%s for job.id=%s", job.modelid, job.id)
    except Exception:
        logger.exception("Failed to set job.modelid for job.id=%s", job.id)
    meta = dict(job.meta or {})
    meta["modelid"] = str(model_uuid)
    job.meta = meta
    job.status = JobStatusEnum.processing
    db.add(job)
    await db.commit()
    logger.info("Committed job update: id=%s modelid=%s status=%s", job.id, job.modelid, job.status.value)

    logger.info("Job created id=%s (model_id=%s) for user=%s", job.id, uid, user_id)

    return api_success({"id": _job_public_id(job.id), "status": job.status.value, "assetId": None})


@router.get("/jobs/{id}")
async def get_job(id: str, user_id: str = Depends(get_current_user_id), db: AsyncSession = Depends(get_db)):
	logger = logging.getLogger(__name__)
	logger.info("=== GET /jobs/%s called by user %s ===", id, user_id)
	
	try:
		job_id = _parse_job_id(id)
		logger.info("Parsed job ID: %s", job_id)
	except ValueError:
		logger.error("Invalid job ID format: %s", id)
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
	
	result = await db.execute(select(Job).where(Job.id == job_id, Job.created_by == user_id))
	job = result.scalar_one_or_none()
	if job is None:
		logger.error("Job not found: %s for user %s", job_id, user_id)
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
	
	logger.info("Found job: id=%s, status=%s, modelid=%s, asset_id=%s", 
		job.id, job.status.value, getattr(job, "modelid", None), getattr(job, "asset_id", None))

	# If we have a provider model job id, get status from inference server and return as-is
	provider_uid = str(job.modelid) if getattr(job, "modelid", None) else None
	logger = logging.getLogger(__name__)
	logger.info("GET /jobs/%s: job.modelid=%s, provider_uid=%s", id, getattr(job, "modelid", None), provider_uid)
	
	if provider_uid:
		# Use the exact inference server URL format as specified
		inference_url = f"http://74.225.34.67:8081/status/{provider_uid}"
		logger.info("=== QUERYING INFERENCE SERVER: %s ===", inference_url)
		
		try:
			with httpx.Client(timeout=httpx.Timeout(200.0)) as client:
				logger.info("Sending GET request to inference server...")
				resp = client.get(inference_url)
				logger.info("Inference server response: status=%s, content-type=%s, content-length=%s", 
					resp.status_code, 
					resp.headers.get("content-type", "unknown"),
					resp.headers.get("content-length", "unknown"))
				
				if resp.status_code < 400:
					# Check if response is JSON or binary
					content_type = resp.headers.get("content-type", "").lower()
					logger.info("Response content type detected: %s", content_type)
					
					if "application/json" in content_type or "text/" in content_type:
						# Try to parse as JSON
						try:
							logger.info("Attempting to parse JSON response...")
							json_data = resp.json()
							logger.info("Successfully parsed JSON response from inference server: %s", json_data)
							
							# Check if json_data is None or empty
							if json_data is None:
								logger.warning("JSON data is None, returning error response")
								return api_success({"error": "Inference server returned null response", "status": "null_response"})
							
							final_response = api_success(json_data)
							logger.info("=== RETURNING JSON RESPONSE: %s ===", final_response)
							
							# Double-check the response before returning
							if final_response is None:
								logger.error("Final response is None! This should never happen")
								return api_success({"error": "Response serialization failed", "original_data": str(json_data)})
							
							return final_response
							
						except Exception as json_error:
							logger.error("Failed to parse response as JSON: %s", str(json_error))
							logger.error("Raw response text: %s", resp.text[:500])  # First 500 chars
							
							# Return raw text if JSON parsing fails
							error_response = api_success({"raw_response": resp.text, "status": "parsing_error"})
							logger.info("=== RETURNING ERROR RESPONSE: %s ===", error_response)
							return error_response
					else:
						# Binary response - this means the job is completed, process the asset
						logger.info("Received binary response (content-type: %s), processing completed asset for job %s", content_type, job.id)
						completed_response = await _process_completed_job(job, resp, user_id, db, logger)
						logger.info("=== RETURNING COMPLETED JOB RESPONSE ===")
						return completed_response
				else:
					logger.warning("Inference server returned error status: %s, body: %s", resp.status_code, resp.text)
					# Return error status wrapped in api_success for consistency
				return api_success({
					"error": "Inference server error",
					"status_code": resp.status_code,
					"message": resp.text,
					"jobId": _job_public_id(job.id)
				})
		except Exception as e:
			logger.warning("Provider status request failed for url=%s: %s", inference_url, str(e))
			# Return error information wrapped in api_success
		return api_success({
			"error": "Connection to inference server failed",
			"message": str(e),
			"jobId": _job_public_id(job.id),
			"inference_url": inference_url
		})

	# If no provider UID or all requests failed, return basic job status
	logger.info("=== RETURNING FALLBACK RESPONSE ===")
	logger.info("No provider UID or inference server failed for job %s with status %s", id, job.status.value)
	
	# If job has an asset_id, get the asset information
	asset_id = None
	file_url = None
	usdz_url = None
	blob_urls = {}
	asset_url = None
	has_multiple_formats = False
	
	if job.asset_id:
		asset_result = await db.execute(select(Asset).where(Asset.id == job.asset_id))
		asset = asset_result.scalar_one_or_none()
		if asset:
			asset_id = str(asset.id)
			logger.info("Found asset %s for job %s", asset_id, job.id)
			
			# Get all asset parts ordered by position
			parts_result = await db.execute(select(AssetPart).where(AssetPart.asset_id == asset.id).order_by(AssetPart.position.asc()))
			asset_parts = parts_result.scalars().all()

			formats = {}
			for part in asset_parts:
				part_meta = part.meta or {}
				part_format = part_meta.get("format", "unknown")
				# part.url is always the CDN URL
				formats[part_format] = part.url

				# Store asset base URL if available
				if part_meta.get("asset_url_base") and not asset_url:
					asset_url = part_meta["asset_url_base"]

				# Set primary CDN URL (GLB takes precedence)
				if part_format == "glb" or file_url is None:
					file_url = part.url

				# Set USDZ CDN URL if available
				if part_format == "usdz":
					usdz_url = part.url

			has_multiple_formats = len(formats) > 1
			logger.info("Asset %s has formats: %s", asset_id, list(formats.keys()))

	glb_response_url = file_url
	usdz_response_url = usdz_url
	
	response_data = {
		"id": _job_public_id(job.id), 
		"status": job.status.value, 
		"assetId": asset_id, 
		"glburl": glb_response_url,
		"usdzURL": usdz_response_url
	}
	
	logger.info("=== FINAL FALLBACK RESPONSE DATA: %s ===", response_data)
	
	final_fallback_response = api_success(response_data)
	logger.info("=== FINAL FALLBACK RESPONSE WRAPPED: %s ===", final_fallback_response)
	
	return final_fallback_response
