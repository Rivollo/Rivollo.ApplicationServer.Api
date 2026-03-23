"""Service layer for support operations - contains business logic."""

from typing import Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Support
from app.repositories.support_repository import SupportRepository
from app.services.email_service import EmailService


class SupportService:
    """Service layer for support business logic."""

    def __init__(self, db: AsyncSession):
        """Initialize service with database session."""
        self.repository = SupportRepository(db)

    async def create_support_request(
        self,
        fullname: str,
        comment: Optional[str],
        user_id: Optional[UUID],
        user_email: Optional[str] = None,
    ) -> Support:
        """Create a new support request with business logic validation."""
        # Validate fullname
        if not fullname or not fullname.strip():
            raise ValueError("Fullname is required and cannot be empty")

        # Trim whitespace
        fullname = fullname.strip()
        if comment:
            comment = comment.strip()

        # Create support entry through repository
        support_entry = await self.repository.create(
            fullname=fullname,
            comment=comment,
            user_id=user_id,
        )

        await EmailService.send_support_contact_email(
            fullname=fullname,
            comment=comment,
            user_email=user_email or "",
        )

        return support_entry

    async def get_support_by_id(self, support_id: int) -> Optional[Support]:
        """Get support entry by ID."""
        return await self.repository.get_by_id(support_id)

    async def get_user_support_requests(
        self, user_id: UUID, limit: int = 100
    ) -> list[Support]:
        """Get all support requests for a user."""
        return await self.repository.get_by_user_id(user_id, limit)

