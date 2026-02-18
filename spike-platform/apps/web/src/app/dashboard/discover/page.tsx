"use client";

import { useState } from "react";
import {
  Search,
  TrendingUp,
  TrendingDown,
  Flame,
  Filter,
  Plus,
} from "lucide-react";
import { useTrendingStocks, useSectorPerformance } from "@/lib/api/hooks";

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
  inputBg: "#F8FFFE",
  orange: "#F5A623",
};

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(value);
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
      <div className="grid grid-cols-4 gap-3">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="rounded-2xl p-6" style={{ backgroundColor: "#FFFFFF", border: "1px solid #B8DDD7" }}>
            <div className="animate-pulse space-y-4">
              <div className="h-5 w-24 rounded" style={{ backgroundColor: "#F5FFFC" }} />
              <div className="h-4 w-full rounded" style={{ backgroundColor: "#F5FFFC" }} />
            </div>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-6">
        {[1, 2].map((i) => (
          <div key={i} className="rounded-2xl p-6" style={{ backgroundColor: "#FFFFFF", border: "1px solid #B8DDD7" }}>
            <div className="animate-pulse space-y-4">
              <div className="h-5 w-40 rounded" style={{ backgroundColor: "#F5FFFC" }} />
              {[1, 2, 3, 4, 5].map((j) => (
                <div key={j} className="h-4 w-full rounded" style={{ backgroundColor: "#F5FFFC" }} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function DiscoverPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const { data: trendingData, isLoading: trendingLoading } = useTrendingStocks(10);
  const { data: sectorsData, isLoading: sectorsLoading } = useSectorPerformance();

  const isLoading = trendingLoading || sectorsLoading;

  // Split trending stocks into display sections
  const trendingStocks = trendingData?.slice(0, 4) ?? [];
  const topGainers = (trendingData ?? [])
    .filter((s) => s.change_percent >= 0)
    .sort((a, b) => b.change_percent - a.change_percent)
    .slice(0, 5);
  const topLosers = (trendingData ?? [])
    .filter((s) => s.change_percent < 0)
    .sort((a, b) => a.change_percent - b.change_percent)
    .slice(0, 5);
  const sectors = sectorsData ?? [];

  if (isLoading) {
    return <LoadingSkeleton />;
  }

  return (
    <div className="max-w-6xl">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-xl font-semibold mb-1" style={{ color: colors.textPrimary }}>
          Discover
        </h1>
        <p className="text-sm" style={{ color: colors.textMuted }}>
          Explore trending stocks and market opportunities
        </p>
      </div>

      {/* Search */}
      <div className="relative mb-6">
        <Search
          className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4"
          style={{ color: colors.textMuted }}
        />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search stocks, ETFs, or mutual funds"
          className="w-full pl-10 pr-24 py-2.5 text-sm rounded-lg transition-colors focus:outline-none"
          style={{
            backgroundColor: colors.inputBg,
            border: `1px solid ${colors.border}`,
            color: colors.textPrimary,
          }}
          onFocus={(e) => {
            e.currentTarget.style.backgroundColor = colors.bg;
            e.currentTarget.style.borderColor = colors.accent;
          }}
          onBlur={(e) => {
            e.currentTarget.style.backgroundColor = colors.inputBg;
            e.currentTarget.style.borderColor = colors.border;
          }}
        />
        <button
          className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1 px-2.5 py-1.5 text-xs rounded-md transition-colors"
          style={{ color: colors.textSecondary }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
        >
          <Filter className="w-3 h-3" />
          Filters
        </button>
      </div>

      {/* Trending */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-3">
          <Flame className="w-4 h-4" style={{ color: colors.orange }} />
          <h2 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
            Trending Today
          </h2>
        </div>
        <div className="grid grid-cols-4 gap-3">
          {trendingStocks.map((stock) => {
            const isPositive = stock.change_percent >= 0;
            return (
              <div
                key={stock.symbol}
                className="rounded-xl p-3 cursor-pointer transition-colors"
                style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = colors.bg)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>
                      {stock.symbol}
                    </p>
                    <p
                      className="text-xs truncate max-w-[100px]"
                      style={{ color: colors.textMuted }}
                    >
                      {stock.name}
                    </p>
                  </div>
                  <button
                    className="p-1 rounded transition-colors"
                    style={{ color: colors.textMuted }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgMint)}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
                  >
                    <Plus className="w-3.5 h-3.5" />
                  </button>
                </div>
                <p className="text-base font-semibold mb-1" style={{ color: colors.textPrimary }}>
                  {formatCurrency(stock.price)}
                </p>
                <div className="flex items-center justify-between">
                  <span
                    className="text-xs font-medium"
                    style={{ color: isPositive ? colors.gain : colors.loss }}
                  >
                    {isPositive ? "+" : ""}{stock.change_percent.toFixed(2)}%
                  </span>
                  <span className="text-xs" style={{ color: colors.textMuted }}>
                    Vol: {stock.volume}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Top Movers */}
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* Gainers */}
        <div
          className="rounded-xl overflow-hidden"
          style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
        >
          <div
            className="px-4 py-3 flex items-center gap-2"
            style={{ borderBottom: `1px solid ${colors.border}` }}
          >
            <TrendingUp className="w-4 h-4" style={{ color: colors.gain }} />
            <h2 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
              Top Gainers
            </h2>
          </div>
          <div>
            {topGainers.map((stock, index) => (
              <div
                key={stock.symbol}
                className="px-4 py-2.5 flex items-center justify-between cursor-pointer transition-colors"
                style={{ borderBottom: `1px solid ${colors.border}` }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
              >
                <div className="flex items-center gap-3">
                  <span className="text-xs w-4" style={{ color: colors.textMuted }}>
                    {index + 1}
                  </span>
                  <div>
                    <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>
                      {stock.symbol}
                    </p>
                    <p className="text-xs" style={{ color: colors.textMuted }}>
                      {stock.name}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm" style={{ color: colors.textPrimary }}>
                    {formatCurrency(stock.price)}
                  </p>
                  <p className="text-xs font-medium" style={{ color: colors.gain }}>
                    +{stock.change_percent.toFixed(2)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Losers */}
        <div
          className="rounded-xl overflow-hidden"
          style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
        >
          <div
            className="px-4 py-3 flex items-center gap-2"
            style={{ borderBottom: `1px solid ${colors.border}` }}
          >
            <TrendingDown className="w-4 h-4" style={{ color: colors.loss }} />
            <h2 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
              Top Losers
            </h2>
          </div>
          <div>
            {topLosers.map((stock, index) => (
              <div
                key={stock.symbol}
                className="px-4 py-2.5 flex items-center justify-between cursor-pointer transition-colors"
                style={{ borderBottom: `1px solid ${colors.border}` }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
              >
                <div className="flex items-center gap-3">
                  <span className="text-xs w-4" style={{ color: colors.textMuted }}>
                    {index + 1}
                  </span>
                  <div>
                    <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>
                      {stock.symbol}
                    </p>
                    <p className="text-xs" style={{ color: colors.textMuted }}>
                      {stock.name}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm" style={{ color: colors.textPrimary }}>
                    {formatCurrency(stock.price)}
                  </p>
                  <p className="text-xs font-medium" style={{ color: colors.loss }}>
                    {stock.change_percent.toFixed(2)}%
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Sectors */}
      <div
        className="rounded-xl overflow-hidden"
        style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
      >
        <div className="px-4 py-3" style={{ borderBottom: `1px solid ${colors.border}` }}>
          <h2 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
            Sector Performance
          </h2>
        </div>
        <div className="grid grid-cols-6">
          {sectors.map((sector, index) => {
            const isPositive = sector.change_percent >= 0;
            return (
              <div
                key={sector.name}
                className="p-4 cursor-pointer transition-colors text-center"
                style={{
                  borderRight: index < sectors.length - 1 ? `1px solid ${colors.border}` : "none",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
              >
                <p className="text-xs mb-1" style={{ color: colors.textSecondary }}>
                  {sector.name}
                </p>
                <p
                  className="text-lg font-semibold"
                  style={{ color: isPositive ? colors.gain : colors.loss }}
                >
                  {isPositive ? "+" : ""}{sector.change_percent.toFixed(1)}%
                </p>
                <p className="text-xs mt-1" style={{ color: colors.textMuted }}>
                  {sector.top_gainer}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
