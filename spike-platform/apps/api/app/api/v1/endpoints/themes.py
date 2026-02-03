"""
Smart Themes endpoints - AI-curated investment baskets.
"""

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.api.v1.deps.auth import get_current_user, get_optional_user, ClerkUser

router = APIRouter()


class ThemeStock(BaseModel):
    """Stock in a theme."""

    symbol: str
    name: str
    allocation: float
    finscore: float
    change_percent: float


class Theme(BaseModel):
    """Investment theme."""

    id: str
    name: str
    description: str
    category: str
    stocks: list[ThemeStock]
    returns_1m: float
    returns_3m: float
    returns_1y: float
    risk_level: str
    min_investment: int
    finscore: float
    popularity: int


@router.get("/", response_model=list[Theme])
async def get_themes(
    category: str | None = Query(None),
    risk_level: str | None = Query(None, regex="^(low|medium|high)$"),
    user: ClerkUser | None = Depends(get_optional_user),
) -> list[Theme]:
    """Get all available investment themes."""
    themes = [
        Theme(
            id="quality-compounders",
            name="Quality Compounders",
            description="High-quality companies with consistent earnings growth and strong moats",
            category="Quality",
            stocks=[
                ThemeStock(
                    symbol="RELIANCE",
                    name="Reliance Industries",
                    allocation=20.0,
                    finscore=8.4,
                    change_percent=1.17,
                ),
                ThemeStock(
                    symbol="TCS",
                    name="Tata Consultancy",
                    allocation=18.0,
                    finscore=7.9,
                    change_percent=-0.32,
                ),
            ],
            returns_1m=5.2,
            returns_3m=12.8,
            returns_1y=28.5,
            risk_level="medium",
            min_investment=10000,
            finscore=8.2,
            popularity=1250,
        ),
        Theme(
            id="india-2030-growth",
            name="India 2030 Growth",
            description="Companies positioned to benefit from India's growth story",
            category="Growth",
            stocks=[
                ThemeStock(
                    symbol="HDFCBANK",
                    name="HDFC Bank",
                    allocation=15.0,
                    finscore=8.1,
                    change_percent=0.95,
                ),
            ],
            returns_1m=4.8,
            returns_3m=15.2,
            returns_1y=32.1,
            risk_level="high",
            min_investment=25000,
            finscore=7.8,
            popularity=890,
        ),
        Theme(
            id="dividend-aristocrats",
            name="Dividend Aristocrats",
            description="Consistent dividend payers with strong yield",
            category="Income",
            stocks=[],
            returns_1m=2.1,
            returns_3m=6.5,
            returns_1y=15.2,
            risk_level="low",
            min_investment=50000,
            finscore=7.5,
            popularity=650,
        ),
    ]

    if category:
        themes = [t for t in themes if t.category.lower() == category.lower()]
    if risk_level:
        themes = [t for t in themes if t.risk_level == risk_level]

    return themes


@router.get("/{theme_id}", response_model=Theme)
async def get_theme(
    theme_id: str,
    user: ClerkUser | None = Depends(get_optional_user),
) -> Theme:
    """Get detailed information about a specific theme."""
    return Theme(
        id=theme_id,
        name="Quality Compounders",
        description="High-quality companies with consistent earnings growth",
        category="Quality",
        stocks=[
            ThemeStock(
                symbol="RELIANCE",
                name="Reliance Industries",
                allocation=20.0,
                finscore=8.4,
                change_percent=1.17,
            ),
            ThemeStock(
                symbol="TCS",
                name="Tata Consultancy",
                allocation=18.0,
                finscore=7.9,
                change_percent=-0.32,
            ),
            ThemeStock(
                symbol="HDFCBANK",
                name="HDFC Bank",
                allocation=15.0,
                finscore=8.1,
                change_percent=0.95,
            ),
        ],
        returns_1m=5.2,
        returns_3m=12.8,
        returns_1y=28.5,
        risk_level="medium",
        min_investment=10000,
        finscore=8.2,
        popularity=1250,
    )


@router.post("/{theme_id}/invest")
async def invest_in_theme(
    theme_id: str,
    amount: int = Query(..., ge=1000),
    user: ClerkUser = Depends(get_current_user),
) -> dict:
    """Invest in a theme (requires broker integration)."""
    return {
        "status": "pending",
        "theme_id": theme_id,
        "amount": amount,
        "message": "Investment requires broker authorization.",
    }
