import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps import CurrentUser, DB
from app.schemas.support import SupportCreateRequest, SupportResponse
from app.services.support_service import SupportService
from app.utils.envelopes import api_error, api_success

router = APIRouter(tags=["support"])


@router.post("/support/contact", response_model=dict, status_code=status.HTTP_201_CREATED)
async def create_support_contact(
    payload: SupportCreateRequest,
    current_user: CurrentUser,
    db: DB,
):
    """Create a new support contact entry."""
    try:
        # Parse user ID from payload (for validation)
        try:
            payload_user_id = uuid.UUID(payload.userid)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user ID format. Must be a valid UUID.",
            )

        # Use authenticated user's ID for security (ignore payload userid if different)
        # This ensures users can only create support requests for themselves
        user_id = current_user.id

        # Optionally validate that payload userid matches authenticated user
        if payload_user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User ID in request does not match authenticated user.",
            )

        # Initialize service
        support_service = SupportService(db)

        # Create support entry
        support_entry = await support_service.create_support_request(
            fullname=payload.fullname,
            comment=payload.comment,
            user_id=user_id,
            user_email=current_user.email,
        )

        # Build response
        response_data = SupportResponse(
            id=support_entry.id,
            fullname=support_entry.fullname,
            comment=support_entry.comment,
            isactive=support_entry.isactive,
            created_by=str(support_entry.created_by) if support_entry.created_by else None,
            created_date=support_entry.created_date,
            updated_by=str(support_entry.updated_by) if support_entry.updated_by else None,
            updated_date=support_entry.updated_date,
        )

        return api_success(response_data.model_dump())

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while creating support entry: {str(e)}",
        )

