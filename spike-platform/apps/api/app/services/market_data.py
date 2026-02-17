"""Unified market data service for Indian stock markets."""

import logging
from dataclasses import dataclass
from datetime import datetime

import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class StockQuote:
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    high: float
    low: float
    open: float
    prev_close: float
    market_cap: float | None = None
    pe_ratio: float | None = None
    week_52_high: float | None = None
    week_52_low: float | None = None


@dataclass
class OHLCV:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class IndexData:
    symbol: str
    name: str
    value: float
    change: float
    change_percent: float


@dataclass
class SectorData:
    name: str
    change_percent: float
    top_gainer: str
    top_loser: str


@dataclass
class SearchResult:
    symbol: str
    name: str
    exchange: str
    type: str


class MarketDataService:
    """Unified market data. yfinance primary, Angel One optional premium."""

    # Major NSE indices
    INDICES = {
        "^NSEI": "NIFTY 50",
        "^BSESN": "SENSEX",
        "^NSEBANK": "BANK NIFTY",
        "^CNXIT": "NIFTY IT",
    }

    # Sector ETFs/indices for sector performance
    SECTORS = {
        "NIFTY_IT": {"symbol": "^CNXIT", "name": "IT"},
        "NIFTY_BANK": {"symbol": "^NSEBANK", "name": "Banking"},
        "NIFTY_PHARMA": {"symbol": "^CNXPHARMA", "name": "Pharma"},
        "NIFTY_AUTO": {"symbol": "^CNXAUTO", "name": "Auto"},
        "NIFTY_FMCG": {"symbol": "^CNXFMCG", "name": "FMCG"},
        "NIFTY_METAL": {"symbol": "^CNXMETAL", "name": "Metal"},
        "NIFTY_REALTY": {"symbol": "^CNXREALTY", "name": "Realty"},
        "NIFTY_ENERGY": {"symbol": "^CNXENERGY", "name": "Energy"},
    }

    # Popular stocks for trending/search
    POPULAR_STOCKS = [
        "RELIANCE.NS",
        "TCS.NS",
        "HDFCBANK.NS",
        "INFY.NS",
        "ICICIBANK.NS",
        "HINDUNILVR.NS",
        "ITC.NS",
        "SBIN.NS",
        "BHARTIARTL.NS",
        "KOTAKBANK.NS",
        "LT.NS",
        "AXISBANK.NS",
        "ASIANPAINT.NS",
        "MARUTI.NS",
        "TITAN.NS",
        "SUNPHARMA.NS",
        "ULTRACEMCO.NS",
        "WIPRO.NS",
        "HCLTECH.NS",
        "BAJFINANCE.NS",
        "TATAMOTORS.NS",
        "ADANIENT.NS",
        "ADANIPORTS.NS",
        "NTPC.NS",
        "POWERGRID.NS",
        "TATASTEEL.NS",
        "ONGC.NS",
        "COALINDIA.NS",
        "JSWSTEEL.NS",
        "TECHM.NS",
    ]

    def __init__(self, angel_one_service=None):
        self._angel_one = angel_one_service

    async def get_stock_quote(self, symbol: str) -> StockQuote:
        """Get real-time quote for a stock."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            fast_info = ticker.fast_info

            price = (
                fast_info.get("lastPrice", 0)
                or info.get("currentPrice", 0)
                or info.get("regularMarketPrice", 0)
            )
            prev_close = info.get("previousClose", 0) or info.get(
                "regularMarketPreviousClose", 0
            )
            change = price - prev_close if price and prev_close else 0
            change_pct = (change / prev_close * 100) if prev_close else 0

            return StockQuote(
                symbol=symbol,
                name=info.get("shortName", "") or info.get("longName", symbol),
                price=round(price, 2),
                change=round(change, 2),
                change_percent=round(change_pct, 2),
                volume=info.get("volume", 0) or info.get("regularMarketVolume", 0),
                high=info.get("dayHigh", 0) or info.get("regularMarketDayHigh", 0),
                low=info.get("dayLow", 0) or info.get("regularMarketDayLow", 0),
                open=info.get("open", 0) or info.get("regularMarketOpen", 0),
                prev_close=round(prev_close, 2),
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                week_52_high=info.get("fiftyTwoWeekHigh"),
                week_52_low=info.get("fiftyTwoWeekLow"),
            )
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            raise

    async def get_historical(
        self, symbol: str, period: str = "1mo", interval: str = "1d"
    ) -> list[OHLCV]:
        """Get historical OHLCV data."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            return [
                OHLCV(
                    timestamp=idx.to_pydatetime(),
                    open=round(row["Open"], 2),
                    high=round(row["High"], 2),
                    low=round(row["Low"], 2),
                    close=round(row["Close"], 2),
                    volume=int(row["Volume"]),
                )
                for idx, row in df.iterrows()
            ]
        except Exception as e:
            logger.error(f"Error fetching history for {symbol}: {e}")
            raise

    async def get_market_indices(self) -> list[IndexData]:
        """Get major Indian market indices."""
        results = []
        for symbol, name in self.INDICES.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                fast_info = ticker.fast_info

                price = fast_info.get("lastPrice", 0) or info.get(
                    "regularMarketPrice", 0
                )
                prev_close = info.get("previousClose", 0) or info.get(
                    "regularMarketPreviousClose", 0
                )
                change = price - prev_close if price and prev_close else 0
                change_pct = (change / prev_close * 100) if prev_close else 0

                results.append(
                    IndexData(
                        symbol=symbol,
                        name=name,
                        value=round(price, 2),
                        change=round(change, 2),
                        change_percent=round(change_pct, 2),
                    )
                )
            except Exception as e:
                logger.warning(f"Could not fetch index {symbol}: {e}")
        return results

    async def get_sector_performance(self) -> list[SectorData]:
        """Get sector performance data."""
        results = []
        for key, sector in self.SECTORS.items():
            try:
                ticker = yf.Ticker(sector["symbol"])
                info = ticker.info
                prev_close = info.get("previousClose", 0)
                price = info.get("regularMarketPrice", 0)
                change_pct = (
                    ((price - prev_close) / prev_close * 100) if prev_close else 0
                )

                results.append(
                    SectorData(
                        name=sector["name"],
                        change_percent=round(change_pct, 2),
                        top_gainer="",
                        top_loser="",
                    )
                )
            except Exception as e:
                logger.warning(f"Could not fetch sector {key}: {e}")
        return results

    async def get_trending(self, limit: int = 10) -> list[StockQuote]:
        """Get trending stocks (top movers by change percent from popular stocks)."""
        quotes = []
        for symbol in self.POPULAR_STOCKS[:limit]:
            try:
                quote = await self.get_stock_quote(symbol)
                quotes.append(quote)
            except Exception:
                continue
        # Sort by absolute change percent (most moved)
        quotes.sort(key=lambda q: abs(q.change_percent), reverse=True)
        return quotes

    async def search_stocks(self, query: str) -> list[SearchResult]:
        """Search stocks by name or symbol."""
        query_upper = query.upper()
        results = []
        for symbol in self.POPULAR_STOCKS:
            clean = symbol.replace(".NS", "")
            if query_upper in clean:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    results.append(
                        SearchResult(
                            symbol=symbol,
                            name=info.get("shortName", "")
                            or info.get("longName", clean),
                            exchange="NSE",
                            type="Equity",
                        )
                    )
                except Exception:
                    results.append(
                        SearchResult(
                            symbol=symbol,
                            name=clean,
                            exchange="NSE",
                            type="Equity",
                        )
                    )
        return results

    async def get_market_status(self) -> dict:
        """Check if Indian markets are open."""
        from zoneinfo import ZoneInfo

        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)

        # Market hours: 9:15 AM - 3:30 PM IST, Monday-Friday
        is_weekday = now.weekday() < 5
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        is_market_hours = market_open <= now <= market_close

        return {
            "is_open": is_weekday and is_market_hours,
            "current_time": now.isoformat(),
            "next_open": "09:15 IST",
            "next_close": "15:30 IST",
        }


# Singleton
_market_data_service: MarketDataService | None = None


def get_market_data_service() -> MarketDataService:
    """Get or create the market data service singleton."""
    global _market_data_service
    if _market_data_service is None:
        _market_data_service = MarketDataService()
    return _market_data_service
