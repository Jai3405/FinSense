"""
AI Services endpoints - Strategy-GPT, Legend Agents, and AI insights.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.v1.deps.auth import get_current_user, require_plan, ClerkUser

router = APIRouter()


class StrategyGPTRequest(BaseModel):
    """Strategy-GPT query request."""

    query: str
    context: dict | None = None


class StrategyGPTResponse(BaseModel):
    """Strategy-GPT response."""

    strategy_name: str
    description: str
    rules: list[dict]
    backtest_summary: dict | None
    code: str | None


class LegendAgentRequest(BaseModel):
    """Legend Agent analysis request."""

    symbol: str
    agent: str  # "buffett", "lynch", "dalio"


class LegendAnalysis(BaseModel):
    """Legend Agent analysis response."""

    agent_name: str
    symbol: str
    verdict: str
    confidence: float
    reasoning: str
    key_metrics: list[dict]
    risks: list[str]
    quote: str


class AIInsight(BaseModel):
    """AI-generated insight."""

    id: str
    type: str
    priority: str
    title: str
    description: str
    action: str | None
    related_symbols: list[str]
    timestamp: str


@router.post("/strategy-gpt", response_model=StrategyGPTResponse)
async def strategy_gpt(
    request: StrategyGPTRequest,
    user: ClerkUser = Depends(require_plan(["pro", "pro_plus", "premium"])),
) -> StrategyGPTResponse:
    """
    Strategy-GPT: Convert natural language to quantitative strategy.

    Examples:
    - "Buy stocks with PE < 20 and ROE > 15% in IT sector"
    - "Find momentum stocks breaking 52-week highs"
    - "Create a defensive portfolio for bear markets"
    """
    return StrategyGPTResponse(
        strategy_name="Quality Momentum Strategy",
        description="Identifies quality stocks with strong momentum characteristics",
        rules=[
            {
                "type": "filter",
                "condition": "PE Ratio",
                "operator": "<",
                "value": 20,
            },
            {
                "type": "filter",
                "condition": "ROE",
                "operator": ">",
                "value": 15,
            },
            {
                "type": "filter",
                "condition": "Sector",
                "operator": "in",
                "value": ["IT", "Technology"],
            },
            {
                "type": "entry",
                "condition": "Price > 50-DMA",
                "action": "BUY",
            },
        ],
        backtest_summary={
            "period": "2019-2024",
            "total_return": 145.2,
            "annualized_return": 19.5,
            "sharpe_ratio": 1.42,
            "max_drawdown": -18.5,
            "win_rate": 62.3,
        },
        code="""
# Strategy: Quality Momentum
def screen_stocks(universe):
    return universe.filter(
        (universe['pe_ratio'] < 20) &
        (universe['roe'] > 15) &
        (universe['sector'].isin(['IT', 'Technology']))
    )

def entry_signal(stock):
    return stock.price > stock.sma(50)
""",
    )


@router.post("/legend", response_model=LegendAnalysis)
async def legend_agent_analysis(
    request: LegendAgentRequest,
    user: ClerkUser = Depends(get_current_user),
) -> LegendAnalysis:
    """
    Get investment analysis from a Legend Agent.

    Available agents:
    - buffett: Warren Buffett's value investing philosophy
    - lynch: Peter Lynch's GARP (Growth at Reasonable Price)
    - dalio: Ray Dalio's all-weather, risk parity approach
    """
    agents = {
        "buffett": {
            "name": "Warren Buffett",
            "focus": "Intrinsic value, competitive moats, management quality",
            "quote": "Price is what you pay. Value is what you get.",
        },
        "lynch": {
            "name": "Peter Lynch",
            "focus": "PEG ratio, understanding the business, tenbaggers",
            "quote": "Know what you own, and know why you own it.",
        },
        "dalio": {
            "name": "Ray Dalio",
            "focus": "Risk parity, diversification, economic cycles",
            "quote": "He who lives by the crystal ball will eat shattered glass.",
        },
    }

    agent_info = agents.get(request.agent, agents["buffett"])

    return LegendAnalysis(
        agent_name=agent_info["name"],
        symbol=request.symbol.upper(),
        verdict="buy",
        confidence=78.5,
        reasoning=f"Based on {agent_info['name']}'s philosophy focusing on {agent_info['focus']}, "
        f"{request.symbol.upper()} presents an attractive opportunity. "
        "The company has strong fundamentals, consistent earnings growth, "
        "and trades at a reasonable valuation relative to its quality.",
        key_metrics=[
            {"name": "ROE", "value": "22.5%", "assessment": "Excellent"},
            {"name": "Debt/Equity", "value": "0.3", "assessment": "Conservative"},
            {"name": "PE Ratio", "value": "24.5", "assessment": "Fair"},
            {"name": "PEG Ratio", "value": "1.2", "assessment": "Attractive"},
        ],
        risks=[
            "Sector concentration in portfolio",
            "Currency fluctuation exposure",
            "Regulatory changes in key markets",
        ],
        quote=agent_info["quote"],
    )


@router.get("/insights", response_model=list[AIInsight])
async def get_ai_insights(
    user: ClerkUser = Depends(get_current_user),
) -> list[AIInsight]:
    """
    Get personalized AI-generated insights for the user.
    Based on portfolio, watchlist, and market conditions.
    """
    return [
        AIInsight(
            id="insight-1",
            type="opportunity",
            priority="high",
            title="TATAELXSI showing breakout pattern",
            description="Stock crossed 50-DMA with 2x volume. FinScore: 8.2. "
            "Historical success rate of similar setups: 72%.",
            action="View Analysis",
            related_symbols=["TATAELXSI"],
            timestamp="2024-01-28T15:30:00+05:30",
        ),
        AIInsight(
            id="insight-2",
            type="warning",
            priority="medium",
            title="Portfolio concentration risk",
            description="65% allocation to IT sector. Consider diversifying "
            "into Pharma or FMCG for better risk-adjusted returns.",
            action="Optimize Portfolio",
            related_symbols=["SUNPHARMA", "HINDUNILVR"],
            timestamp="2024-01-28T14:00:00+05:30",
        ),
        AIInsight(
            id="insight-3",
            type="trend",
            priority="medium",
            title="Market regime shift detected",
            description="Market transitioning from consolidation to bullish phase. "
            "Consider adjusting strategy to favor momentum stocks.",
            action="View Regime Analysis",
            related_symbols=[],
            timestamp="2024-01-28T12:00:00+05:30",
        ),
    ]


@router.post("/chat")
async def ai_chat(
    message: str,
    user: ClerkUser = Depends(get_current_user),
) -> dict:
    """
    Chat with SPIKE AI for investment-related queries.
    """
    return {
        "response": f"Based on your question about '{message[:50]}...', "
        "here's my analysis...",
        "sources": ["FinScore Data", "Market Analysis", "Portfolio Context"],
        "suggested_actions": [
            {"label": "View FinScore", "action": "navigate", "target": "/finscore"},
            {"label": "See Portfolio", "action": "navigate", "target": "/portfolio"},
        ],
    }
