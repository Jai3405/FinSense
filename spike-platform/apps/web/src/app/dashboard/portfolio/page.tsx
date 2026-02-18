"use client";

import { useState } from "react";
import { TrendingUp, TrendingDown, ChevronDown } from "lucide-react";
import { usePortfolioSummary, useHoldings } from "@/lib/api/hooks";

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

function LoadingSkeleton() {
  return (
    <div className="max-w-6xl space-y-6">
      <div className="rounded-2xl p-6" style={{ backgroundColor: "#FFFFFF", border: "1px solid #B8DDD7" }}>
        <div className="animate-pulse space-y-4">
          <div className="h-5 w-40 rounded" style={{ backgroundColor: "#F5FFFC" }} />
          <div className="h-4 w-full rounded" style={{ backgroundColor: "#F5FFFC" }} />
          <div className="h-4 w-3/4 rounded" style={{ backgroundColor: "#F5FFFC" }} />
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="rounded-2xl p-6" style={{ backgroundColor: "#FFFFFF", border: "1px solid #B8DDD7" }}>
            <div className="animate-pulse space-y-4">
              <div className="h-5 w-40 rounded" style={{ backgroundColor: "#F5FFFC" }} />
              <div className="h-4 w-full rounded" style={{ backgroundColor: "#F5FFFC" }} />
            </div>
          </div>
        ))}
      </div>
      <div className="rounded-2xl p-6" style={{ backgroundColor: "#FFFFFF", border: "1px solid #B8DDD7" }}>
        <div className="animate-pulse space-y-4">
          <div className="h-5 w-40 rounded" style={{ backgroundColor: "#F5FFFC" }} />
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="h-4 w-full rounded" style={{ backgroundColor: "#F5FFFC" }} />
          ))}
        </div>
      </div>
    </div>
  );
}

export default function PortfolioPage() {
  const [sortBy, setSortBy] = useState<string>("value");
  const { data: portfolioSummary, isLoading: summaryLoading } = usePortfolioSummary();
  const { data: holdings, isLoading: holdingsLoading } = useHoldings();

  const isLoading = summaryLoading || holdingsLoading;

  if (isLoading) {
    return <LoadingSkeleton />;
  }

  if (!portfolioSummary || !holdings || holdings.length === 0) {
    return (
      <div className="max-w-6xl">
        <div
          className="rounded-xl p-12 text-center"
          style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
        >
          <p className="text-lg font-semibold mb-2" style={{ color: colors.textPrimary }}>
            No holdings yet
          </p>
          <p className="text-sm" style={{ color: colors.textMuted }}>
            Add stocks to your portfolio to see them here.
          </p>
        </div>
      </div>
    );
  }

  const isPositiveDay = portfolioSummary.today_change_percent >= 0;
  const isPositiveTotal = portfolioSummary.returns_percent >= 0;

  const sortedHoldings = [...holdings].sort((a, b) => {
    return b.current_value - a.current_value;
  });

  return (
    <div className="max-w-6xl">
      {/* Portfolio Summary */}
      <div className="mb-6">
        <div className="flex items-baseline gap-3 mb-1">
          <h1 className="text-2xl font-semibold" style={{ color: colors.textPrimary }}>
            {formatCurrency(portfolioSummary.total_value)}
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
              {formatCurrency(portfolioSummary.today_change)} ({portfolioSummary.today_change_percent.toFixed(2)}%)
            </span>
          </div>
        </div>
        <p className="text-sm" style={{ color: colors.textMuted }}>
          Invested {formatCurrency(portfolioSummary.invested)}
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
            {formatCurrency(portfolioSummary.total_value)}
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
            {formatCurrency(portfolioSummary.invested)}
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
            {isPositiveTotal ? "+" : ""}{formatCurrency(portfolioSummary.returns)}
            <span className="text-sm font-normal ml-1">
              ({isPositiveTotal ? "+" : ""}{portfolioSummary.returns_percent.toFixed(2)}%)
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
                  Returns %
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedHoldings.map((holding) => {
                const isPositive = holding.returns >= 0;

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
                      {holding.quantity}
                    </td>
                    <td className="px-4 py-3 text-right text-sm" style={{ color: colors.textSecondary }}>
                      {formatNumber(holding.avg_price)}
                    </td>
                    <td className="px-4 py-3 text-right text-sm font-medium" style={{ color: colors.textPrimary }}>
                      {formatNumber(holding.current_price)}
                    </td>
                    <td className="px-4 py-3 text-right text-sm font-medium" style={{ color: colors.textPrimary }}>
                      {formatCurrency(holding.current_value)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <div>
                        <p
                          className="text-sm font-medium"
                          style={{ color: isPositive ? colors.gain : colors.loss }}
                        >
                          {isPositive ? "+" : ""}{formatCurrency(holding.returns)}
                        </p>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span
                        className="text-sm font-medium"
                        style={{ color: isPositive ? colors.gain : colors.loss }}
                      >
                        {isPositive ? "+" : ""}{holding.returns_percent.toFixed(2)}%
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
