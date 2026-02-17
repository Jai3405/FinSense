"use client";

import { ArrowDownRight, ArrowUpRight, TrendingUp } from "lucide-react";
import { formatCurrency, formatPercent } from "@/lib/utils";
import { usePortfolioSummary } from "@/lib/api/hooks";

// Color constants
const colors = {
  bg: "#FFFFFF",
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

export function PortfolioSummary() {
  const { data: portfolioData, isLoading, error } = usePortfolioSummary();

  if (isLoading) {
    return (
      <div
        className="rounded-2xl p-6"
        style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
      >
        <div className="animate-pulse space-y-4">
          <div className="h-5 w-40 rounded" style={{ backgroundColor: colors.bgMint }} />
          <div className="h-4 w-full rounded" style={{ backgroundColor: colors.bgMint }} />
          <div className="h-4 w-3/4 rounded" style={{ backgroundColor: colors.bgMint }} />
        </div>
      </div>
    );
  }

  if (error || !portfolioData) {
    return (
      <div
        className="rounded-2xl p-6"
        style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
      >
        <h2 className="text-lg font-semibold mb-4" style={{ color: colors.textPrimary }}>
          Portfolio Summary
        </h2>
        <div className="text-center py-8">
          <p className="text-sm mb-2" style={{ color: colors.textMuted }}>
            No holdings yet
          </p>
          <p className="text-sm" style={{ color: colors.accent }}>
            Add your first holding to get started
          </p>
        </div>
      </div>
    );
  }

  const portfolio = {
    totalValue: portfolioData.total_value,
    invested: portfolioData.invested,
    returns: portfolioData.returns,
    returnsPercent: portfolioData.returns_percent,
    todayChange: portfolioData.today_change,
    todayChangePercent: portfolioData.today_change_percent,
  };

  const isPositive = portfolio.todayChange >= 0;

  return (
    <div
      className="rounded-2xl p-6"
      style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
    >
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
          Portfolio Summary
        </h2>
        <div
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg"
          style={{ backgroundColor: colors.gainBg }}
        >
          <TrendingUp className="w-4 h-4" style={{ color: colors.gain }} />
          <span className="text-sm font-medium" style={{ color: colors.gain }}>
            Outperforming NIFTY by 8.2%
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {/* Total Value */}
        <div>
          <p className="text-sm mb-1" style={{ color: colors.textMuted }}>Total Value</p>
          <p className="text-2xl font-bold" style={{ color: colors.textPrimary }}>
            {formatCurrency(portfolio.totalValue)}
          </p>
        </div>

        {/* Invested */}
        <div>
          <p className="text-sm mb-1" style={{ color: colors.textMuted }}>Invested</p>
          <p className="text-2xl font-bold" style={{ color: colors.textPrimary }}>
            {formatCurrency(portfolio.invested)}
          </p>
        </div>

        {/* Total Returns */}
        <div>
          <p className="text-sm mb-1" style={{ color: colors.textMuted }}>Total Returns</p>
          <div className="flex items-baseline gap-2">
            <p className="text-2xl font-bold" style={{ color: portfolio.returns >= 0 ? colors.gain : colors.loss }}>
              {formatCurrency(portfolio.returns)}
            </p>
            <span className="text-sm" style={{ color: portfolio.returns >= 0 ? colors.gain : colors.loss }}>
              {formatPercent(portfolio.returnsPercent)}
            </span>
          </div>
        </div>

        {/* Today's Change */}
        <div>
          <p className="text-sm mb-1" style={{ color: colors.textMuted }}>Today's Change</p>
          <div className="flex items-center gap-2">
            <div
              className="p-1 rounded-lg"
              style={{ backgroundColor: isPositive ? colors.gainBg : colors.lossBg }}
            >
              {isPositive ? (
                <ArrowUpRight className="w-4 h-4" style={{ color: colors.gain }} />
              ) : (
                <ArrowDownRight className="w-4 h-4" style={{ color: colors.loss }} />
              )}
            </div>
            <div>
              <p
                className="text-xl font-bold"
                style={{ color: isPositive ? colors.gain : colors.loss }}
              >
                {formatCurrency(Math.abs(portfolio.todayChange))}
              </p>
              <span
                className="text-sm"
                style={{ color: isPositive ? colors.gain : colors.loss }}
              >
                {formatPercent(portfolio.todayChangePercent)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Mini Chart Placeholder */}
      <div
        className="mt-6 h-24 rounded-xl flex items-end p-4"
        style={{ background: `linear-gradient(to right, ${colors.gainBg}, ${colors.bgMint})` }}
      >
        <div className="flex items-end gap-1 w-full">
          {[40, 45, 38, 52, 48, 60, 55, 65, 70, 68, 75, 80, 78, 85, 90].map(
            (h, i) => (
              <div
                key={i}
                className="flex-1 rounded-t"
                style={{ height: `${h}%`, backgroundColor: `${colors.gain}99` }}
              />
            )
          )}
        </div>
      </div>
    </div>
  );
}
