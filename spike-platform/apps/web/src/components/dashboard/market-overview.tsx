"use client";

import { TrendingUp, TrendingDown, Activity } from "lucide-react";

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

interface SectorData {
  name: string;
  change: number;
}

export function MarketOverview() {
  const sectors: SectorData[] = [
    { name: "IT", change: 2.4 },
    { name: "Banks", change: -0.8 },
    { name: "Pharma", change: 1.2 },
    { name: "Auto", change: 0.5 },
    { name: "FMCG", change: -0.3 },
    { name: "Metal", change: 3.1 },
    { name: "Energy", change: 1.8 },
    { name: "Realty", change: -1.5 },
  ];

  const marketBreadth = {
    advances: 1245,
    declines: 892,
    unchanged: 156,
  };

  const regimeStatus = {
    current: "Bullish",
    confidence: 78,
    trend: "Uptrend",
  };

  return (
    <div
      className="rounded-2xl p-6"
      style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
    >
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
          Market Overview
        </h2>
        <div className="flex items-center gap-2 text-sm">
          <Activity className="w-4 h-4" style={{ color: colors.accent }} />
          <span style={{ color: colors.textMuted }}>Live</span>
        </div>
      </div>

      {/* Regime Indicator */}
      <div
        className="flex items-center gap-4 p-4 rounded-xl mb-6"
        style={{ backgroundColor: colors.gainBg, border: `1px solid ${colors.gain}30` }}
      >
        <div
          className="w-12 h-12 rounded-xl flex items-center justify-center"
          style={{ backgroundColor: `${colors.gain}20` }}
        >
          <TrendingUp className="w-6 h-6" style={{ color: colors.gain }} />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
              {regimeStatus.current} Regime
            </span>
            <span
              className="px-2 py-0.5 text-xs rounded-full"
              style={{ backgroundColor: `${colors.gain}20`, color: colors.gain }}
            >
              {regimeStatus.confidence}% confidence
            </span>
          </div>
          <p className="text-sm" style={{ color: colors.textSecondary }}>
            Market showing sustained {regimeStatus.trend.toLowerCase()} momentum
          </p>
        </div>
      </div>

      {/* Market Breadth */}
      <div className="mb-6">
        <p className="text-sm mb-3" style={{ color: colors.textMuted }}>Market Breadth</p>
        <div
          className="flex items-center gap-0.5 h-3 rounded-full overflow-hidden"
          style={{ backgroundColor: colors.border }}
        >
          <div
            className="h-full rounded-l-full"
            style={{
              backgroundColor: colors.gain,
              width: `${
                (marketBreadth.advances /
                  (marketBreadth.advances +
                    marketBreadth.declines +
                    marketBreadth.unchanged)) *
                100
              }%`,
            }}
          />
          <div
            className="h-full"
            style={{
              backgroundColor: colors.textMuted,
              width: `${
                (marketBreadth.unchanged /
                  (marketBreadth.advances +
                    marketBreadth.declines +
                    marketBreadth.unchanged)) *
                100
              }%`,
            }}
          />
          <div
            className="h-full rounded-r-full"
            style={{
              backgroundColor: colors.loss,
              width: `${
                (marketBreadth.declines /
                  (marketBreadth.advances +
                    marketBreadth.declines +
                    marketBreadth.unchanged)) *
                100
              }%`,
            }}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs">
          <span style={{ color: colors.gain }}>
            {marketBreadth.advances} Advances
          </span>
          <span style={{ color: colors.textMuted }}>
            {marketBreadth.unchanged} Unchanged
          </span>
          <span style={{ color: colors.loss }}>
            {marketBreadth.declines} Declines
          </span>
        </div>
      </div>

      {/* Sector Performance */}
      <div>
        <p className="text-sm mb-3" style={{ color: colors.textMuted }}>Sector Performance</p>
        <div className="grid grid-cols-4 gap-3">
          {sectors.map((sector) => (
            <div
              key={sector.name}
              className="p-3 rounded-xl text-center cursor-pointer transition-colors"
              style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}` }}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = colors.bgMint)}
            >
              <p className="text-sm mb-1" style={{ color: colors.textSecondary }}>{sector.name}</p>
              <p
                className="text-sm font-semibold"
                style={{ color: sector.change >= 0 ? colors.gain : colors.loss }}
              >
                {sector.change >= 0 ? "+" : ""}
                {sector.change.toFixed(1)}%
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
