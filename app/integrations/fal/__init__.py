"""fal.ai image-to-3D integration.

Two pieces, deliberately separated:

  registry     WHAT differs per model — endpoint, request body, result shape,
               credit cost. Database-backed (tbl_mstr_3d_models); adding a
               model is a row, not a code change.
  queue_client HOW every fal model is called — submit, poll, fetch, download.
               Model-agnostic; never edited when a model is added.
"""

from app.integrations.fal.queue_client import (
    FalGenerateResponse,
    FalQueueClient,
    fal_queue_client,
)
from app.integrations.fal.registry import (
    FalModelSpec,
    clear_model_registry_cache,
    get_model_spec,
    get_model_spec_any,
    list_model_specs,
)

__all__ = [
    "FalGenerateResponse",
    "FalModelSpec",
    "FalQueueClient",
    "clear_model_registry_cache",
    "fal_queue_client",
    "get_model_spec",
    "get_model_spec_any",
    "list_model_specs",
]
