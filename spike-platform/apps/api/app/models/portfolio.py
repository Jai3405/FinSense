"""
Portfolio and holdings models.
"""

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

if TYPE_CHECKING:
    from app.models.user import User


class Portfolio(Base):
    """User portfolio model."""

    __tablename__ = "portfolios"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(255), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(100), default="My Portfolio")
    description: Mapped[str | None] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="portfolios")
    holdings: Mapped[list["Holding"]] = relationship(
        "Holding", back_populates="portfolio", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Portfolio {self.name}>"


class Holding(Base):
    """Portfolio holding model."""

    __tablename__ = "holdings"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    portfolio_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("portfolios.id", ondelete="CASCADE"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    quantity: Mapped[int] = mapped_column(Integer)
    avg_price: Mapped[float] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    portfolio: Mapped["Portfolio"] = relationship("Portfolio", back_populates="holdings")

    def __repr__(self) -> str:
        return f"<Holding {self.symbol} x {self.quantity}>"


class Transaction(Base):
    """Portfolio transaction history."""

    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4())
    )
    portfolio_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("portfolios.id", ondelete="CASCADE"), index=True
    )
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    transaction_type: Mapped[str] = mapped_column(String(10))  # "buy" or "sell"
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str | None] = mapped_column(Text)

    # Timestamps
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<Transaction {self.transaction_type} {self.symbol}>"
