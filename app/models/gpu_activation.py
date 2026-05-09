"""GpuActivation model — tracks when a GPU cold-start was initiated."""

from datetime import datetime

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql.sqltypes import TIMESTAMP

from app.models.base import Base


class GpuActivation(Base):
    """One row per 3D service — upserted each time a cold-start is detected.
    Used to calculate the remaining warm-up time for subsequent users."""
    __tablename__ = "tbl_gpu_activation"

    service: Mapped[str] = mapped_column(String(50), primary_key=True)
    activation_started_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False
    )
