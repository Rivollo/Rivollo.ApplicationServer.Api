"""3D-generation model registry — database-backed configuration.

One row per selectable image-to-3D model. Adding, repricing, reordering, or
disabling a model is a row change here, not a code change — see
``app/integrations/fal/registry.py`` for how a row becomes a usable spec, and
``sql/create_3d_model_registry.sql`` for the table + the current seed data.

Every model this app calls today speaks fal.ai's queue protocol (submit ->
poll -> result -> download), so ``provider`` is currently always
``'fal_queue'``. It exists as its own column — rather than folding that
assumption into the table name or schema — because a genuinely different
protocol (Tripo's own direct API: bearer auth, a ``code == 0`` success
envelope, task polling with real progress, a two-stage chained pipeline)
cannot be expressed as request-body JSON no matter how the schema is shaped.
A future non-fal-queue provider gets a new ``provider`` value and its own
driver in ``app/integrations``, not a new table or a migration to widen this
one — that is the entire reason this column and ``provider_config`` (rather
than fal-specific top-level columns) exist from day one, even though only
one provider exists right now.
"""

import uuid
from typing import Any, Optional

from sqlalchemy import Boolean, Index, Integer, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import text

from app.models.base import Base
from app.models.models import AuditMixin, UUIDMixin


class Model3DConfig(UUIDMixin, AuditMixin, Base):
    """One selectable image-to-3D generation model."""

    __tablename__ = "tbl_mstr_3d_models"
    __table_args__ = (
        # Enforced in the database, not just by convention: at most one row
        # can be the default at a time. A plain boolean column alone would
        # let a careless UPDATE leave two rows both `is_default = true`.
        Index(
            "ux_3d_models_single_default",
            "is_default",
            unique=True,
            postgresql_where=text("is_default"),
        ),
    )

    key: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'fal_queue'")
    )
    label: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    endpoint_id: Mapped[str] = mapped_column(Text, nullable=False)
    credit_cost: Mapped[int] = mapped_column(Integer, nullable=False)
    baseline_estimate_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    max_wait_seconds: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("600")
    )
    free_plan_eligible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    # Exactly one active row should carry this — see the partial unique index
    # above. Resolved dynamically (WHERE is_default), never hardcoded as a
    # key string in Python, so changing the default model is a data change.
    is_default: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    # Display order in the model picker.
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    # Soft-disable without deleting: an inactive model drops out of the
    # picker and can no longer be selected for a NEW generation, but a row
    # keyed by an in-flight job or a past ModelGenerationStat entry must
    # still resolve — see get_model_spec vs get_model_spec_any in
    # app/integrations/fal/registry.py for the two lookup modes this implies.
    isactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    # Provider-specific request/response shape, interpreted by the driver
    # named in `provider`. For provider='fal_queue' this holds:
    #   image_url_field        which JSON key carries the image URL (vendors
    #                           disagree — Hunyuan wants "input_image_url",
    #                           everyone else wants "image_url")
    #   request_body_template  every other field of the request, as a static
    #                           object — every current model's request is
    #                           this template plus the image URL substituted
    #                           in, nothing computed per-call
    #   glb_url_paths          ordered dot-paths tried against the result
    #                           JSON for the finished GLB (e.g.
    #                           "model_urls.glb.url"); a segment suffixed
    #                           "[]" means "iterate this list, first match
    #                           wins" — see _resolve_path in registry.py
    #   usdz_url_paths          same, for a vendor-supplied USDZ; empty means
    #                           this model doesn't export one itself
    provider_config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    # Hard-won operational knowledge that used to live as Python comments
    # next to each model's request builder — e.g. "quad topology makes Tripo
    # return FBX, not GLB" or "Trellis at the API's default 500k vertices
    # fails UV-unwrap with HTTP 500". A JSONB template can't carry comments,
    # so this column exists specifically so that knowledge isn't lost when a
    # model's config moves from code into a row. See the seed migration for
    # what each current model's notes say.
    notes: Mapped[Optional[str]] = mapped_column(Text)
