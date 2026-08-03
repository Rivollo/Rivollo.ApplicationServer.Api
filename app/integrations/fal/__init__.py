"""fal.ai image-to-3D integration.

Two pieces, deliberately separated:

  registry     WHAT differs per model — endpoint, request body, result shape,
               credit cost. Adding a model is one entry here.
  queue_client HOW every fal model is called — submit, poll, fetch, download.
               Model-agnostic; never edited when a model is added.
"""

from app.integrations.fal.queue_client import (
    FalGenerateResponse,
    FalQueueClient,
    fal_queue_client,
)
from app.integrations.fal.registry import (
    DEFAULT_MODEL_KEY,
    FAL_MODELS,
    FalModelSpec,
    get_model_spec,
    list_model_specs,
)

__all__ = [
    "DEFAULT_MODEL_KEY",
    "FAL_MODELS",
    "FalGenerateResponse",
    "FalModelSpec",
    "FalQueueClient",
    "fal_queue_client",
    "get_model_spec",
    "list_model_specs",
]
