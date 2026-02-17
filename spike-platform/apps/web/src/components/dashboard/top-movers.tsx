"use client";

import { useState } from "react";
import { ArrowUpRight, ArrowDownRight, Target } from "lucide-react";
import { cn, formatCurrency, getFinScoreColor } from "@/lib/utils";
import { useTrendingStocks } from "@/lib/api/hooks";

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

interface Stock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  volume: string;
  finScore: number;
}

export function TopMovers() {
  const [activeTab, setActiveTab] = useState<"gainers" | "losers">("gainers");
  const { data: trendingData, isLoading, error } = useTrendingStocks(10);

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

  const allStocks: Stock[] = trendingData
    ? trendingData.map((t) => ({
        symbol: t.symbol,
        name: t.name,
        price: t.price,
        change: (t.change_percent / 100) * t.price, // derive absolute change from percent
        changePercent: t.change_percent,
        volume: t.volume,
        finScore: 0.0, // Phase 2
      }))
    : [];

  const gainers = allStocks
    .filter((s) => s.changePercent > 0)
    .sort((a, b) => b.changePercent - a.changePercent);

  const losers = allStocks
    .filter((s) => s.changePercent < 0)
    .sort((a, b) => a.changePercent - b.changePercent);

  const stocks = activeTab === "gainers" ? gainers : losers;

  return (
    <div
      className="rounded-2xl p-6"
      style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
    >
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
          Top Movers
        </h2>
        <div
          className="flex rounded-xl p-1"
          style={{ backgroundColor: colors.bgMint }}
        >
          <button
            onClick={() => setActiveTab("gainers")}
            className="px-4 py-2 rounded-lg text-sm font-medium transition"
            style={{
              backgroundColor: activeTab === "gainers" ? colors.gainBg : "transparent",
              color: activeTab === "gainers" ? colors.gain : colors.textMuted,
            }}
          >
            Gainers
          </button>
          <button
            onClick={() => setActiveTab("losers")}
            className="px-4 py-2 rounded-lg text-sm font-medium transition"
            style={{
              backgroundColor: activeTab === "losers" ? colors.lossBg : "transparent",
              color: activeTab === "losers" ? colors.loss : colors.textMuted,
            }}
          >
            Losers
          </button>
        </div>
      </div>

      {error || stocks.length === 0 ? (
        <p className="text-sm text-center py-4" style={{ color: colors.textMuted }}>
          No {activeTab} data available
        </p>
      ) : (
        <div className="space-y-3">
          {stocks.map((stock) => (
            <div
              key={stock.symbol}
              className="flex items-center justify-between p-4 rounded-xl transition cursor-pointer"
              style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}` }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = colors.bgMint)}
            >
              <div className="flex items-center gap-4">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ backgroundColor: colors.border }}
                >
                  <span className="text-xs font-bold" style={{ color: colors.textPrimary }}>
                    {stock.symbol.slice(0, 2)}
                  </span>
                </div>
                <div>
                  <p className="font-medium" style={{ color: colors.textPrimary }}>{stock.symbol}</p>
                  <p className="text-sm" style={{ color: colors.textMuted }}>{stock.name}</p>
                </div>
              </div>

              <div className="flex items-center gap-6">
                {/* FinScore */}
                <div className="flex items-center gap-1.5">
                  <Target className={cn("w-4 h-4", getFinScoreColor(stock.finScore))} />
                  <span className={cn("text-sm font-medium", getFinScoreColor(stock.finScore))}>
                    {stock.finScore.toFixed(1)}
                  </span>
                </div>

                {/* Volume */}
                <div className="text-right hidden sm:block">
                  <p className="text-sm" style={{ color: colors.textMuted }}>Vol</p>
                  <p className="text-sm" style={{ color: colors.textPrimary }}>{stock.volume}</p>
                </div>

                {/* Price & Change */}
                <div className="text-right min-w-[100px]">
                  <p className="font-medium" style={{ color: colors.textPrimary }}>
                    {formatCurrency(stock.price)}
                  </p>
                  <div
                    className="flex items-center justify-end gap-1 text-sm"
                    style={{ color: stock.change >= 0 ? colors.gain : colors.loss }}
                  >
                    {stock.change >= 0 ? (
                      <ArrowUpRight className="w-4 h-4" />
                    ) : (
                      <ArrowDownRight className="w-4 h-4" />
                    )}
                    <span>
                      {stock.change >= 0 ? "+" : ""}
                      {stock.changePercent.toFixed(2)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
