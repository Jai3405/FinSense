"""
User and authentication related models.
"""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.models.portfolio import Portfolio
    from app.models.watchlist import Watchlist


class User(Base):
    """User model - synced with Clerk."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)  # Clerk user ID
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    first_name: Mapped[str | None] = mapped_column(String(100))
    last_name: Mapped[str | None] = mapped_column(String(100))
    image_url: Mapped[str | None] = mapped_column(Text)

    # Subscription
    plan: Mapped[str] = mapped_column(
        Enum("free", "pro", "pro_plus", "premium", name="plan_type"),
        default="free",
    )
    plan_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Risk Profile
    risk_score: Mapped[int | None] = mapped_column(Integer)
    risk_category: Mapped[str | None] = mapped_column(
        Enum(
            "conservative",
            "moderate",
            "aggressive",
            "very_aggressive",
            name="risk_category",
        )
    )
    investment_horizon: Mapped[str | None] = mapped_column(
        Enum("short", "medium", "long", name="investment_horizon")
    )

    # Onboarding
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    onboarding_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Relationships
    portfolios: Mapped[list["Portfolio"]] = relationship(
        "Portfolio", back_populates="user", cascade="all, delete-orphan"
    )
    watchlists: Mapped[list["Watchlist"]] = relationship(
        "Watchlist", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.email}>"
