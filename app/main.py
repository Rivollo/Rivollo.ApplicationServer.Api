from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from typing import Optional
import os

from opentelemetry.trace import get_current_span

from app.core.config import settings
from app.api.routes.auth import router as auth_router
from app.api.routes.users import router as users_router
from app.api.routes.products import router as products_router, public_router as public_products_router, public_noauth_router as public_products_noauth_router
from app.api.routes.uploads import router as uploads_router
from app.api.routes.jobs import router as jobs_router
from app.api.routes.assets import router as assets_router
from app.api.routes.subscriptions import router as subscriptions_router
from app.api.routes.galleries import router as galleries_router
from app.api.routes.branding import router as branding_router
from app.api.routes.analytics import router as analytics_router
from app.api.routes.dashboard import router as dashboard_router
from app.api.routes.search import router as search_router
from app.api.routes.health import router as health_router
from app.api.routes.hotspots import router as hotspots_router
from app.api.routes.dimensions import router as dimensions_router
from app.api.routes.product_links import router as product_links_router
from app.api.routes.support import router as support_router
from app.api.routes.link_share import router as link_share_router
from app.utils.envelopes import api_success, api_error
from app.core.db import init_engine_and_session


app = FastAPI(title=settings.APP_NAME)

# Telemetry / Azure Monitor (optional)
_logger = logging.getLogger("rivollo.api")

# Ensure we also write logs to a local file `.server.log`
def _ensure_file_logging() -> None:
	root_logger = logging.getLogger()
	server_log_path = os.path.join(os.getcwd(), ".server.log")
	already_added = False
	for h in root_logger.handlers:
		try:
			base = getattr(h, "baseFilename", None)
			if base and os.path.abspath(base) == os.path.abspath(server_log_path):
				already_added = True
				break
		except Exception:
			continue
	if not already_added:
		fh = logging.FileHandler(server_log_path, encoding="utf-8")
		fh.setLevel(logging.INFO)
		formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
		fh.setFormatter(formatter)
		root_logger.addHandler(fh)
		# Make sure INFO logs propagate to file
		if root_logger.level > logging.INFO or root_logger.level == logging.NOTSET:
			root_logger.setLevel(logging.INFO)
try:
	if settings.ENABLE_APP_INSIGHTS and settings.AZURE_MONITOR_CONN_STR:
		from azure.monitor.opentelemetry import configure_azure_monitor
		from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
		from opentelemetry.instrumentation.logging import LoggingInstrumentor

		configure_azure_monitor(
			connection_string=settings.AZURE_MONITOR_CONN_STR,
			sampling_ratio=settings.SAMPLING_RATIO,
			enable_live_metrics=True,
		)
		# Include trace/span ids in stdlib logging records
		LoggingInstrumentor().instrument(set_logging_format=True)
		# Instrument FastAPI to automatically capture request traces/metrics
		FastAPIInstrumentor.instrument_app(app)
		_logger.info("Azure Monitor telemetry is enabled")
except Exception as telemetry_exc:
	# Do not block app startup if telemetry fails
	logging.getLogger(__name__).warning("Failed to initialize Azure Monitor telemetry: %s", telemetry_exc)

# Initialize file logging regardless of telemetry
_ensure_file_logging()

# CORS (adjust for real origins later)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Normalize API prefix (must not end with '/')
_api_prefix = settings.API_PREFIX.rstrip("/")

# Include routers
app.include_router(health_router, prefix=_api_prefix)
app.include_router(auth_router, prefix=_api_prefix)
app.include_router(users_router, prefix=_api_prefix)
app.include_router(subscriptions_router, prefix=_api_prefix)
app.include_router(products_router, prefix=_api_prefix)
app.include_router(public_products_router, prefix=_api_prefix)
app.include_router(public_products_noauth_router, prefix=_api_prefix)
app.include_router(galleries_router, prefix=_api_prefix)
app.include_router(branding_router, prefix=_api_prefix)
app.include_router(analytics_router, prefix=_api_prefix)
app.include_router(dashboard_router, prefix=_api_prefix)
app.include_router(search_router, prefix=_api_prefix)
app.include_router(uploads_router, prefix=_api_prefix)
app.include_router(jobs_router, prefix=_api_prefix)
app.include_router(assets_router, prefix=_api_prefix)
app.include_router(hotspots_router, prefix=_api_prefix)
app.include_router(dimensions_router, prefix=_api_prefix)
app.include_router(product_links_router, prefix=_api_prefix)
app.include_router(support_router, prefix=_api_prefix)
app.include_router(link_share_router, prefix=_api_prefix)



# Structured request logging (includes trace correlation where available)
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
	start_time = time.perf_counter()
	client_ip: Optional[str] = request.headers.get("x-forwarded-for") or request.client.host if request.client else None
	user_agent: Optional[str] = request.headers.get("user-agent")
	try:
		response = await call_next(request)
		return response
	except Exception as e:
		# Log here as well; handler below will also run
		_current_span = get_current_span()
		trace_id_int = _current_span.get_span_context().trace_id if _current_span else 0
		trace_id = f"{trace_id_int:032x}" if trace_id_int else None
		_logger.exception(
			"Unhandled exception during request",
			extra={
				"http.method": request.method,
				"http.route": request.url.path,
				"net.peer.ip": client_ip,
				"http.user_agent": user_agent,
				"trace_id": trace_id,
			},
		)
		raise
	finally:
		elapsed_ms = (time.perf_counter() - start_time) * 1000.0
		_current_span = get_current_span()
		trace_id_int = _current_span.get_span_context().trace_id if _current_span else 0
		trace_id = f"{trace_id_int:032x}" if trace_id_int else None
		try:
			status_code = response.status_code  # type: ignore[name-defined]
		except Exception:
			status_code = None
		_logger.info(
			"HTTP request",
			extra={
				"http.method": request.method,
				"http.route": request.url.path,
				"http.status_code": status_code,
				"http.duration_ms": round(elapsed_ms, 2),
				"net.peer.ip": client_ip,
				"http.user_agent": user_agent,
				"trace_id": trace_id,
			},
		)


@app.on_event("startup")
def on_startup() -> None:
	init_engine_and_session()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
	_current_span = get_current_span()
	trace_id_int = _current_span.get_span_context().trace_id if _current_span else 0
	trace_id = f"{trace_id_int:032x}" if trace_id_int else None
	_logger.exception(
		"Unhandled exception",
		extra={
			"http.method": request.method,
			"http.route": request.url.path,
			"trace_id": trace_id,
		},
	)
	return JSONResponse(status_code=500, content=api_error(code="INTERNAL_SERVER_ERROR", message="An unexpected error occurred"))


@app.get("/")
async def root():
	return api_success({"service": settings.APP_NAME, "status": "ok"})
