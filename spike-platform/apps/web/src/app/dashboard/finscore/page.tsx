"use client";

import { useState } from "react";
import {
  Search,
  Shield,
  Zap,
  DollarSign,
  MessageSquare,
  AlertTriangle,
  Users,
  Compass,
  BarChart3,
  Activity,
  ChevronRight,
  Info,
} from "lucide-react";

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
  warning: "#F5A623",
  warningBg: "#FFF8E6",
  loss: "#F45B69",
  lossBg: "#FEF0F1",
  inputBg: "#F8FFFE",
};

// Mock FinScore rankings data
const topRankedStocks = [
  { symbol: "HDFCBANK", name: "HDFC Bank Ltd", score: 92, signal: "STRONG BUY", confidence: 89 },
  { symbol: "TCS", name: "Tata Consultancy Services", score: 89, signal: "BUY", confidence: 85 },
  { symbol: "INFY", name: "Infosys Ltd", score: 87, signal: "BUY", confidence: 82 },
  { symbol: "RELIANCE", name: "Reliance Industries", score: 85, signal: "BUY", confidence: 78 },
  { symbol: "ICICIBANK", name: "ICICI Bank Ltd", score: 84, signal: "BUY", confidence: 80 },
  { symbol: "BAJFINANCE", name: "Bajaj Finance Ltd", score: 82, signal: "BUY", confidence: 75 },
  { symbol: "KOTAKBANK", name: "Kotak Mahindra Bank", score: 80, signal: "HOLD", confidence: 72 },
  { symbol: "ASIANPAINT", name: "Asian Paints Ltd", score: 78, signal: "HOLD", confidence: 70 },
];

// Mock detailed FinScore data
const mockFinScoreDetail = {
  symbol: "HDFCBANK",
  name: "HDFC Bank Ltd",
  sector: "Banking",
  score: 92,
  signal: "STRONG BUY",
  confidence: 89,
  lastUpdated: "2 min ago",
  dimensions: [
    { name: "Quality", score: 95, icon: Shield },
    { name: "Momentum", score: 88, icon: Zap },
    { name: "Value", score: 82, icon: DollarSign },
    { name: "Sentiment", score: 90, icon: MessageSquare },
    { name: "Risk", score: 94, icon: AlertTriangle },
    { name: "Flow", score: 91, icon: Users },
    { name: "Regime", score: 85, icon: Compass },
    { name: "Sector", score: 88, icon: BarChart3 },
    { name: "Technical", score: 93, icon: Activity },
  ],
};

function getScoreColor(score: number): string {
  if (score >= 80) return colors.gain;
  if (score >= 60) return colors.warning;
  return colors.loss;
}

function getSignalStyle(signal: string): { bg: string; color: string } {
  switch (signal) {
    case "STRONG BUY":
    case "BUY":
      return { bg: colors.gainBg, color: colors.gain };
    case "HOLD":
      return { bg: colors.warningBg, color: colors.warning };
    case "SELL":
    case "STRONG SELL":
      return { bg: colors.lossBg, color: colors.loss };
    default:
      return { bg: colors.bgMint, color: colors.textSecondary };
  }
}

export default function FinScorePage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedStock, setSelectedStock] = useState("HDFCBANK");

  return (
    <div className="max-w-6xl">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-xl font-semibold mb-1" style={{ color: colors.textPrimary }}>
          FinScore
        </h1>
        <p className="text-sm" style={{ color: colors.textMuted }}>
          AI-powered stock ratings combining 9 dimensions of analysis
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
          placeholder="Search for a stock..."
          className="w-full pl-10 pr-4 py-2.5 text-sm rounded-lg transition-colors focus:outline-none"
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
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Rankings List */}
        <div
          className="lg:col-span-2 rounded-xl overflow-hidden"
          style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
        >
          <div className="px-4 py-3" style={{ borderBottom: `1px solid ${colors.border}` }}>
            <h2 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
              Top Rated
            </h2>
          </div>
          <div>
            {topRankedStocks.map((stock, index) => {
              const isSelected = stock.symbol === selectedStock;

              return (
                <div
                  key={stock.symbol}
                  className="px-4 py-3 cursor-pointer transition-colors"
                  style={{
                    backgroundColor: isSelected ? colors.bgMint : "transparent",
                    borderBottom: `1px solid ${colors.border}`,
                  }}
                  onClick={() => setSelectedStock(stock.symbol)}
                  onMouseEnter={(e) => {
                    if (!isSelected) e.currentTarget.style.backgroundColor = colors.bgHover;
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected) e.currentTarget.style.backgroundColor = "transparent";
                  }}
                >
                  <div className="flex items-center justify-between">
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
                    <div className="flex items-center gap-3">
                      <span
                        className="text-lg font-semibold"
                        style={{ color: getScoreColor(stock.score) }}
                      >
                        {stock.score}
                      </span>
                      <ChevronRight className="w-4 h-4" style={{ color: colors.border }} />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Detail View */}
        <div
          className="lg:col-span-3 rounded-xl p-5"
          style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
        >
          {/* Header */}
          <div className="flex items-start justify-between mb-6">
            <div>
              <h2 className="text-xl font-semibold" style={{ color: colors.textPrimary }}>
                {mockFinScoreDetail.symbol}
              </h2>
              <p className="text-sm" style={{ color: colors.textMuted }}>
                {mockFinScoreDetail.name}
              </p>
            </div>
            <span
              className="px-2.5 py-1 rounded-md text-xs font-semibold"
              style={{
                backgroundColor: getSignalStyle(mockFinScoreDetail.signal).bg,
                color: getSignalStyle(mockFinScoreDetail.signal).color,
              }}
            >
              {mockFinScoreDetail.signal}
            </span>
          </div>

          {/* Score Display */}
          <div
            className="flex items-center gap-6 mb-8 pb-6"
            style={{ borderBottom: `1px solid ${colors.border}` }}
          >
            <div
              className="w-24 h-24 rounded-full flex items-center justify-center"
              style={{
                border: `4px solid ${getScoreColor(mockFinScoreDetail.score)}`,
                backgroundColor: `${getScoreColor(mockFinScoreDetail.score)}10`,
              }}
            >
              <span
                className="text-3xl font-bold"
                style={{ color: getScoreColor(mockFinScoreDetail.score) }}
              >
                {mockFinScoreDetail.score}
              </span>
            </div>
            <div>
              <p className="text-sm mb-1" style={{ color: colors.textMuted }}>
                Confidence Level
              </p>
              <p className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
                {mockFinScoreDetail.confidence}%
              </p>
              <p className="text-xs mt-2" style={{ color: colors.textMuted }}>
                Updated {mockFinScoreDetail.lastUpdated}
              </p>
            </div>
          </div>

          {/* Dimensions */}
          <div>
            <h3
              className="text-xs font-semibold uppercase tracking-wide mb-4"
              style={{ color: colors.textMuted }}
            >
              9-Dimension Breakdown
            </h3>
            <div className="grid grid-cols-3 gap-3">
              {mockFinScoreDetail.dimensions.map((dim) => {
                const Icon = dim.icon;
                return (
                  <div
                    key={dim.name}
                    className="rounded-lg p-3"
                    style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}` }}
                  >
                    <div className="flex items-center gap-2 mb-2">
                      <Icon className="w-4 h-4" style={{ color: colors.textMuted }} />
                      <span className="text-xs" style={{ color: colors.textSecondary }}>
                        {dim.name}
                      </span>
                    </div>
                    <span
                      className="text-lg font-semibold"
                      style={{ color: getScoreColor(dim.score) }}
                    >
                      {dim.score}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* Info Banner */}
      <div
        className="mt-6 rounded-xl p-4"
        style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}` }}
      >
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: colors.textMuted }} />
          <div>
            <p className="text-sm" style={{ color: colors.textSecondary }}>
              FinScore is a proprietary 0-100 rating that combines fundamental analysis, technical indicators,
              market sentiment, and AI-driven insights to provide actionable investment signals.
            </p>
            <button className="text-sm mt-2 font-medium" style={{ color: colors.accent }}>
              Learn more about methodology
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
