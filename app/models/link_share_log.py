import uuid
from sqlalchemy import Column, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from app.models.base import Base



class LinkShareLog(Base):
    __tablename__ = "tbl_link_share_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    product_id = Column(UUID(as_uuid=True), nullable=True)
    shared_by_user_id = Column(UUID(as_uuid=True), nullable=True)

    link = Column(Text, nullable=False)
    channel = Column(Text, nullable=False)
    destination = Column(Text, nullable=False)

    template_variant = Column(Text, nullable=False)
    external_message_id = Column(Text, nullable=True)

    status = Column(Text, nullable=False)
    error_message = Column(Text, nullable=True)

    created_at = Column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
