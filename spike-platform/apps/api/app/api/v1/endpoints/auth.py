"""
Authentication endpoints.
Integrates with Clerk for user authentication.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.v1.deps.auth import get_current_user, ClerkUser

router = APIRouter()


class UserResponse(BaseModel):
    """User profile response."""

    id: str
    email: str
    first_name: str | None
    last_name: str | None
    image_url: str | None
    plan: str = "free"


class RiskProfileRequest(BaseModel):
    """Risk profiling questionnaire request."""

    investment_horizon: str  # "short", "medium", "long"
    risk_tolerance: int  # 1-5
    monthly_income: int | None = None
    existing_investments: int | None = None
    financial_goals: list[str] = []
    age: int | None = None
    dependents: int = 0


class RiskProfileResponse(BaseModel):
    """Risk profile assessment result."""

    score: int  # 1-10
    category: str  # "conservative", "moderate", "aggressive", "very_aggressive"
    recommended_allocation: dict[str, float]
    insights: list[str]


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    user: ClerkUser = Depends(get_current_user),
) -> UserResponse:
    """
    Get the current authenticated user's profile.
    """
    return UserResponse(
        id=user.id,
        email=user.email,
        first_name=user.first_name,
        last_name=user.last_name,
        image_url=user.image_url,
        plan=user.plan,
    )


@router.post("/risk-profile", response_model=RiskProfileResponse)
async def assess_risk_profile(
    request: RiskProfileRequest,
    user: ClerkUser = Depends(get_current_user),
) -> RiskProfileResponse:
    """
    Assess user's risk profile based on questionnaire.
    SEBI requires risk profiling before providing investment advice.
    """
    # Calculate risk score based on inputs
    score = 5  # Default moderate

    # Adjust based on investment horizon
    if request.investment_horizon == "long":
        score += 2
    elif request.investment_horizon == "short":
        score -= 2

    # Adjust based on risk tolerance
    score += (request.risk_tolerance - 3)

    # Age adjustment
    if request.age:
        if request.age < 30:
            score += 1
        elif request.age > 50:
            score -= 1

    # Clamp score between 1-10
    score = max(1, min(10, score))

    # Determine category
    if score <= 3:
        category = "conservative"
        allocation = {"equity": 30, "debt": 60, "gold": 10}
    elif score <= 5:
        category = "moderate"
        allocation = {"equity": 50, "debt": 40, "gold": 10}
    elif score <= 7:
        category = "aggressive"
        allocation = {"equity": 70, "debt": 25, "gold": 5}
    else:
        category = "very_aggressive"
        allocation = {"equity": 85, "debt": 10, "gold": 5}

    insights = [
        f"Based on your profile, you have a {category} risk appetite.",
        f"We recommend {allocation['equity']}% allocation to equities.",
        "Your portfolio should be reviewed quarterly.",
    ]

    if request.investment_horizon == "long":
        insights.append("Long-term horizon allows for higher equity exposure.")

    return RiskProfileResponse(
        score=score,
        category=category,
        recommended_allocation=allocation,
        insights=insights,
    )


@router.post("/onboarding/complete")
async def complete_onboarding(
    user: ClerkUser = Depends(get_current_user),
) -> dict:
    """
    Mark user onboarding as complete.
    """
    # TODO: Update user record in database
    return {"status": "completed", "redirect": "/dashboard"}
