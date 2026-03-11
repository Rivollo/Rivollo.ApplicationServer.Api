"""Seed database with initial data (plans, demo users)."""

import asyncio
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.core.config import settings
from app.models.models import Plan, User, Organization, OrgMember, Subscription, LicenseAssignment, OrgRole
from app.core.security import hash_password


async def seed_plans(session: AsyncSession) -> None:
    """Create or update subscription plans."""
    plans_data = [
        {
            "code": "free",
            "name": "Free",
        },
        {
            "code": "pro",
            "name": "Pro",
        },
        {
            "code": "enterprise",
            "name": "Enterprise",
        },
    ]

    for plan_data in plans_data:
        result = await session.execute(select(Plan).where(Plan.code == plan_data["code"]))
        existing_plan = result.scalar_one_or_none()

        if existing_plan:
            # Update existing plan name only (quotas column no longer exists)
            existing_plan.name = plan_data["name"]
            print(f"✓ Updated plan: {plan_data['name']}")
        else:
            # Create new plan (no quotas column in schema)
            plan = Plan(code=plan_data["code"], name=plan_data["name"])
            session.add(plan)
            print(f"✓ Created plan: {plan_data['name']}")

    await session.commit()


async def seed_demo_user(session: AsyncSession) -> None:
    """Create a demo user with free plan."""
    demo_email = "demo@rivollo.com"

    result = await session.execute(select(User).where(User.email == demo_email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        print(f"✓ Demo user already exists: {demo_email}")
        return

    # Create demo user
    demo_user = User(
        email=demo_email,
        password_hash=hash_password("demo123456"),
        name="Demo User",
    )
    session.add(demo_user)
    await session.flush()

    # Create organization
    demo_org = Organization(
        name="Demo Organization",
        slug=f"demo-org-{str(demo_user.id)[:8]}",
        branding={},
    )
    session.add(demo_org)
    await session.flush()

    # Add user as org owner
    org_member = OrgMember(
        org_id=demo_org.id,
        user_id=demo_user.id,
        role=OrgRole.OWNER,
    )
    session.add(org_member)

    # Get free plan
    result = await session.execute(select(Plan).where(Plan.code == "free"))
    free_plan = result.scalar_one_or_none()

    if free_plan:
        # Create subscription
        subscription = Subscription(
            user_id=demo_user.id,
            plan_id=free_plan.id,
            status="active",
            seats_purchased=1,
        )
        session.add(subscription)
        await session.flush()

        # Create license with individual limit/usage columns (quotas column removed)
        license_assignment = LicenseAssignment(
            subscription_id=subscription.id,
            user_id=demo_user.id,
            status="active",
            limit_max_products=2,
            limit_max_ai_credits=5,
            limit_max_public_views=1000,
            limit_max_galleries=0,
            usage_products=0,
            usage_ai_credits=0,
            usage_public_views=0,
            usage_galleries=0,
        )
        session.add(license_assignment)

    await session.commit()

    print(f"✓ Created demo user: {demo_email} (password: demo123456)")


async def main() -> None:
    """Run all seed operations."""
    print("🌱 Seeding database...")

    # Ensure async URL (handle common provider variants)
    database_url = settings.DATABASE_URL
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgresql+psycopg2://"):
        database_url = database_url.replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgresql+psycopg://"):
        database_url = database_url.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(database_url, echo=False)
    async_session = async_sessionmaker(engine, expire_on_commit=False)

    async with async_session() as session:
        await seed_plans(session)
        await seed_demo_user(session)

    await engine.dispose()

    print("✅ Database seeding completed!")


if __name__ == "__main__":
    asyncio.run(main())
