from fastapi import APIRouter

from app.api.deps import CurrentUser, DB
from app.schemas.auth import UserResponse, UserUpdateRequest
from app.utils.envelopes import api_success

router = APIRouter(tags=["users"])

# Note: User management endpoints (get/update current user) are protected and require authentication.
@router.get("/users/me", response_model=dict)
async def get_current_user_endpoint(current_user: CurrentUser):
	user_data = UserResponse(
		id=str(current_user.id),
		email=current_user.email,
		name=current_user.name,
		avatar_url=current_user.avatar_url,
		bio=current_user.bio,
		created_at=current_user.created_at,
		updated_at=current_user.updated_at,
	)
	return api_success(user_data.model_dump())


@router.patch("/users/me", response_model=dict)
async def update_current_user_endpoint(
	payload: UserUpdateRequest,
	current_user: CurrentUser,
	db: DB,
):
	updated = False
	if payload.name is not None and payload.name != current_user.name:
		current_user.name = payload.name
		updated = True
	if payload.bio is not None and payload.bio != current_user.bio:
		current_user.bio = payload.bio
		updated = True
	if payload.avatar_url is not None and payload.avatar_url != current_user.avatar_url:
		current_user.avatar_url = payload.avatar_url
		updated = True

	if updated:
		await db.commit()
		await db.refresh(current_user)

	user_data = UserResponse(
		id=str(current_user.id),
		email=current_user.email,
		name=current_user.name,
		avatar_url=current_user.avatar_url,
		bio=current_user.bio,
		created_at=current_user.created_at,
		updated_at=current_user.updated_at,
	)
	return api_success(user_data.model_dump())
