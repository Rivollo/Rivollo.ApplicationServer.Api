from fastapi import APIRouter

from app.api.deps import CurrentUser, DB
from app.schemas.auth import UserResponse, UserUpdateRequest
from app.schemas.users import DeleteAccountRequest, DeleteAccountResponse
from app.services.account_service import AccountService
from app.utils.envelopes import api_success

router = APIRouter(tags=["users"])


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
        auth_provider="email" if current_user.password_hash else "google",
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
        auth_provider="email" if current_user.password_hash else "google",
    )
    return api_success(user_data.model_dump())


@router.delete("/users/me/account", response_model=dict)
async def delete_account_endpoint(
    payload: DeleteAccountRequest,
    current_user: CurrentUser,
    db: DB,
):
    """Schedule the authenticated user's account for deletion.

    The account and all its data are deactivated immediately but retained for 30
    days, during which the account can still be restored. Nothing is erased here.

    Email/password users: send { "password": "current_password" }
    Google OAuth users:   send { "confirmation": "DELETE MY ACCOUNT" }
    """
    result = await AccountService.delete_account(
        db=db,
        user=current_user,
        password=payload.password,
        confirmation=payload.confirmation,
    )
    # "has been deleted" was accurate when deletion was immediate and is not any
    # more — the account is recoverable for the whole retention window, and copy
    # that says otherwise talks people out of restoring an account they still
    # have. Say what is true and name the way back.
    response = DeleteAccountResponse(
        message=(
            "Your account is scheduled for deletion and all your data has been "
            "deactivated. You can restore it until the recovery period ends."
        ),
        deleted_at=result.deleted_at,
        purge_after=result.purge_after,
    )
    return api_success(response.model_dump())
