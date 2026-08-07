"""Tripo direct-API integration (openapi.tripo3d.ai).

Separate from ``app.integrations.fal`` on purpose. fal wraps one Tripo task and
exposes no segmentation; only this direct API has ``generate_parts``. The two
providers speak different protocols, use different auth and bill separately, so
they are different integrations rather than two entries in one registry.

  client    protocol  — Bearer auth, task_id polling, progress, download
  tasks     payloads  — one builder per Tripo task we use
  pipeline  sequence  — geometry then texture, as one operation
"""

from app.integrations.tripo.client import (
    TripoClient,
    TripoError,
    TripoTaskResult,
    tripo_client,
)
from app.integrations.tripo.pipeline import (
    PartsGenerationResult,
    tripo_parts_pipeline,
)

__all__ = [
    "PartsGenerationResult",
    "TripoClient",
    "TripoError",
    "TripoTaskResult",
    "tripo_client",
    "tripo_parts_pipeline",
]
