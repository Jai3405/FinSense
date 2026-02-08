"use client";

import { useState } from "react";
import { TrendingUp, TrendingDown, ChevronDown } from "lucide-react";

// Color constants
const colors = {
  bg: "#FFFFFF",
  bgHover: "#F0FBF9",
  bgMint: "#F5FFFC",
  border: "#B8DDD7",
  textPrimary: "#0D3331",
  textSecondary: "#3D6B66",
  textMuted: "#6B9B94",
  accent: "#00897B",
  gain: "#00B386",
  gainBg: "#E6F9F4",
  loss: "#F45B69",
  lossBg: "#FEF0F1",
};

// Mock portfolio data
const portfolioSummary = {
  totalValue: 1247850.75,
  investedValue: 1050000,
  dayChange: 12450.25,
  dayChangePercent: 1.01,
  totalReturns: 197850.75,
  totalReturnsPercent: 18.84,
};

const holdings = [
  { symbol: "HDFCBANK", name: "HDFC Bank Ltd", qty: 50, avgPrice: 1520.45, currentPrice: 1685.30, dayChange: 1.2 },
  { symbol: "TCS", name: "Tata Consultancy Services", qty: 25, avgPrice: 3450.00, currentPrice: 3892.50, dayChange: -0.5 },
  { symbol: "RELIANCE", name: "Reliance Industries", qty: 40, avgPrice: 2380.25, currentPrice: 2545.80, dayChange: 0.8 },
  { symbol: "INFY", name: "Infosys Ltd", qty: 60, avgPrice: 1420.00, currentPrice: 1565.25, dayChange: 1.5 },
  { symbol: "ICICIBANK", name: "ICICI Bank Ltd", qty: 80, avgPrice: 925.50, currentPrice: 1012.40, dayChange: -0.3 },
  { symbol: "TATAMOTORS", name: "Tata Motors Ltd", qty: 100, avgPrice: 625.00, currentPrice: 742.35, dayChange: 2.1 },
  { symbol: "SUNPHARMA", name: "Sun Pharma Industries", qty: 45, avgPrice: 1180.00, currentPrice: 1245.60, dayChange: 0.4 },
  { symbol: "BHARTIARTL", name: "Bharti Airtel Ltd", qty: 35, avgPrice: 1050.00, currentPrice: 1185.90, dayChange: -0.2 },
];

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 }).format(value);
}

export default function PortfolioPage() {
  const [sortBy, setSortBy] = useState<string>("value");
  const isPositiveDay = portfolioSummary.dayChangePercent >= 0;
  const isPositiveTotal = portfolioSummary.totalReturnsPercent >= 0;

  const sortedHoldings = [...holdings].sort((a, b) => {
    const aValue = a.qty * a.currentPrice;
    const bValue = b.qty * b.currentPrice;
    return bValue - aValue;
  });

  return (
    <div className="max-w-6xl">
      {/* Portfolio Summary */}
      <div className="mb-6">
        <div className="flex items-baseline gap-3 mb-1">
          <h1 className="text-2xl font-semibold" style={{ color: colors.textPrimary }}>
            {formatCurrency(portfolioSummary.totalValue)}
          </h1>
          <div className="flex items-center gap-1">
            {isPositiveDay ? (
              <TrendingUp className="w-4 h-4" style={{ color: colors.gain }} />
            ) : (
              <TrendingDown className="w-4 h-4" style={{ color: colors.loss }} />
            )}
            <span
              className="text-sm font-medium"
              style={{ color: isPositiveDay ? colors.gain : colors.loss }}
            >
              {isPositiveDay ? "+" : ""}
              {formatCurrency(portfolioSummary.dayChange)} ({portfolioSummary.dayChangePercent.toFixed(2)}%)
            </span>
          </div>
        </div>
        <p className="text-sm" style={{ color: colors.textMuted }}>
          Invested {formatCurrency(portfolioSummary.investedValue)}
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <div
          className="rounded-xl p-4"
          style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
        >
          <p className="text-xs uppercase tracking-wide mb-1" style={{ color: colors.textMuted }}>
            Current Value
          </p>
          <p className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
            {formatCurrency(portfolioSummary.totalValue)}
          </p>
        </div>
        <div
          className="rounded-xl p-4"
          style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
        >
          <p className="text-xs uppercase tracking-wide mb-1" style={{ color: colors.textMuted }}>
            Invested
          </p>
          <p className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
            {formatCurrency(portfolioSummary.investedValue)}
          </p>
        </div>
        <div
          className="rounded-xl p-4"
          style={{
            backgroundColor: isPositiveTotal ? colors.gainBg : colors.lossBg,
            border: `1px solid ${isPositiveTotal ? colors.gain : colors.loss}20`,
          }}
        >
          <p className="text-xs uppercase tracking-wide mb-1" style={{ color: colors.textMuted }}>
            Total Returns
          </p>
          <p className="text-lg font-semibold" style={{ color: isPositiveTotal ? colors.gain : colors.loss }}>
            {isPositiveTotal ? "+" : ""}{formatCurrency(portfolioSummary.totalReturns)}
            <span className="text-sm font-normal ml-1">
              ({isPositiveTotal ? "+" : ""}{portfolioSummary.totalReturnsPercent.toFixed(2)}%)
            </span>
          </p>
        </div>
      </div>

      {/* Holdings Section */}
      <div
        className="rounded-xl overflow-hidden"
        style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
      >
        {/* Header */}
        <div
          className="px-4 py-3 flex items-center justify-between"
          style={{ borderBottom: `1px solid ${colors.border}` }}
        >
          <h2 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
            Holdings ({holdings.length})
          </h2>
          <button
            className="flex items-center gap-1 text-xs px-2 py-1 rounded-md transition-colors"
            style={{ color: colors.textSecondary }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
          >
            Sort by: Value
            <ChevronDown className="w-3 h-3" />
          </button>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr style={{ backgroundColor: colors.bgMint, borderBottom: `1px solid ${colors.border}` }}>
                <th
                  className="text-left text-xs font-medium uppercase tracking-wide px-4 py-3"
                  style={{ color: colors.textMuted }}
                >
                  Stock
                </th>
                <th
                  className="text-right text-xs font-medium uppercase tracking-wide px-4 py-3"
                  style={{ color: colors.textMuted }}
                >
                  Qty
                </th>
                <th
                  className="text-right text-xs font-medium uppercase tracking-wide px-4 py-3"
                  style={{ color: colors.textMuted }}
                >
                  Avg. Price
                </th>
                <th
                  className="text-right text-xs font-medium uppercase tracking-wide px-4 py-3"
                  style={{ color: colors.textMuted }}
                >
                  LTP
                </th>
                <th
                  className="text-right text-xs font-medium uppercase tracking-wide px-4 py-3"
                  style={{ color: colors.textMuted }}
                >
                  Current Value
                </th>
                <th
                  className="text-right text-xs font-medium uppercase tracking-wide px-4 py-3"
                  style={{ color: colors.textMuted }}
                >
                  P&L
                </th>
                <th
                  className="text-right text-xs font-medium uppercase tracking-wide px-4 py-3"
                  style={{ color: colors.textMuted }}
                >
                  Day Change
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedHoldings.map((holding, index) => {
                const currentValue = holding.qty * holding.currentPrice;
                const investedValue = holding.qty * holding.avgPrice;
                const pnl = currentValue - investedValue;
                const pnlPercent = ((currentValue - investedValue) / investedValue) * 100;
                const isPositive = pnl >= 0;
                const isDayPositive = holding.dayChange >= 0;

                return (
                  <tr
                    key={holding.symbol}
                    className="transition-colors cursor-pointer"
                    style={{ borderBottom: `1px solid ${colors.border}` }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                  >
                    <td className="px-4 py-3">
                      <div>
                        <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>
                          {holding.symbol}
                        </p>
                        <p className="text-xs" style={{ color: colors.textMuted }}>
                          {holding.name}
                        </p>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right text-sm" style={{ color: colors.textPrimary }}>
                      {holding.qty}
                    </td>
                    <td className="px-4 py-3 text-right text-sm" style={{ color: colors.textSecondary }}>
                      {formatNumber(holding.avgPrice)}
                    </td>
                    <td className="px-4 py-3 text-right text-sm font-medium" style={{ color: colors.textPrimary }}>
                      {formatNumber(holding.currentPrice)}
                    </td>
                    <td className="px-4 py-3 text-right text-sm font-medium" style={{ color: colors.textPrimary }}>
                      {formatCurrency(currentValue)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div>
                        <p
                          className="text-sm font-medium"
                          style={{ color: isPositive ? colors.gain : colors.loss }}
                        >
                          {isPositive ? "+" : ""}{formatCurrency(pnl)}
                        </p>
                        <p className="text-xs" style={{ color: isPositive ? colors.gain : colors.loss }}>
                          {isPositive ? "+" : ""}{pnlPercent.toFixed(2)}%
                        </p>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span
                        className="text-sm font-medium"
                        style={{ color: isDayPositive ? colors.gain : colors.loss }}
                      >
                        {isDayPositive ? "+" : ""}{holding.dayChange.toFixed(2)}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
