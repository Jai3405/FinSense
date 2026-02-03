"""
Watchlist models.
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.models.user import User


class Watchlist(Base):
    """User watchlist model."""

    __tablename__ = "watchlists"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(100), default="My Watchlist")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="watchlists")
    stocks: Mapped[list["WatchlistStock"]] = relationship(
        "WatchlistStock", back_populates="watchlist", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Watchlist {self.name}>"


class WatchlistStock(Base):
    """Stock in a watchlist."""

    __tablename__ = "watchlist_stocks"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    watchlist_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("watchlists.id", ondelete="CASCADE"),
        index=True,
    )
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    alert_price: Mapped[float | None] = mapped_column(Float)
    alert_direction: Mapped[str | None] = mapped_column(String(10))  # "above" or "below"
    notes: Mapped[str | None] = mapped_column(Text)

    # Timestamps
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    watchlist: Mapped["Watchlist"] = relationship(
        "Watchlist", back_populates="stocks"
    )

    def __repr__(self) -> str:
        return f"<WatchlistStock {self.symbol}>"
