"use client";

import { useState } from "react";
import {
  Bot,
  TrendingUp,
  TrendingDown,
  Users,
  Star,
  ChevronRight,
  Play,
  Pause,
  Settings,
  MessageSquare,
  BarChart3,
  Target,
  Shield,
  Sparkles,
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
  loss: "#F45B69",
  lossBg: "#FEF0F1",
};

interface LegendAgent {
  id: string;
  name: string;
  title: string;
  avatar: string;
  style: string;
  description: string;
  philosophy: string[];
  performance1Y: number;
  followers: string;
  winRate: number;
  avgHoldingPeriod: string;
  topPicks: { symbol: string; return: number }[];
  active: boolean;
  color: string;
}

const legendAgents: LegendAgent[] = [
  {
    id: "buffett",
    name: "Warren Buffett",
    title: "The Oracle of Omaha",
    avatar: "WB",
    style: "Value Investing",
    description: "Long-term value investing focused on quality businesses with durable competitive advantages at fair prices.",
    philosophy: ["Buy wonderful companies at fair prices", "Think long-term, invest for decades", "Focus on ROE and consistent earnings"],
    performance1Y: 28.5,
    followers: "45.2K",
    winRate: 78,
    avgHoldingPeriod: "5+ years",
    topPicks: [
      { symbol: "HDFCBANK", return: 24.5 },
      { symbol: "TCS", return: 18.2 },
      { symbol: "NESTLEIND", return: 15.8 },
    ],
    active: true,
    color: "#1E3A5F",
  },
  {
    id: "lynch",
    name: "Peter Lynch",
    title: "The Growth Hunter",
    avatar: "PL",
    style: "Growth at Reasonable Price",
    description: "GARP strategy combining growth potential with reasonable valuations. Invest in what you know.",
    philosophy: ["Invest in what you understand", "Look for 10-baggers", "PEG ratio < 1 is attractive"],
    performance1Y: 35.2,
    followers: "32.8K",
    winRate: 72,
    avgHoldingPeriod: "2-3 years",
    topPicks: [
      { symbol: "TITAN", return: 42.5 },
      { symbol: "DMART", return: 28.4 },
      { symbol: "PIDILITIND", return: 22.1 },
    ],
    active: true,
    color: "#2E7D32",
  },
  {
    id: "graham",
    name: "Benjamin Graham",
    title: "The Father of Value Investing",
    avatar: "BG",
    style: "Deep Value",
    description: "Conservative value investing with focus on margin of safety and intrinsic value.",
    philosophy: ["Margin of safety is paramount", "Mr. Market is emotional, be rational", "Net-net working capital stocks"],
    performance1Y: 18.4,
    followers: "28.5K",
    winRate: 82,
    avgHoldingPeriod: "3-5 years",
    topPicks: [
      { symbol: "COALINDIA", return: 32.1 },
      { symbol: "ONGC", return: 28.5 },
      { symbol: "NMDC", return: 24.2 },
    ],
    active: false,
    color: "#4A148C",
  },
  {
    id: "dalio",
    name: "Ray Dalio",
    title: "The All Weather Strategist",
    avatar: "RD",
    style: "Risk Parity",
    description: "Diversified portfolio approach balancing risk across asset classes for all market conditions.",
    philosophy: ["Diversification is key", "Balance risk, not capital", "Prepare for any economic environment"],
    performance1Y: 15.8,
    followers: "22.1K",
    winRate: 68,
    avgHoldingPeriod: "1-2 years",
    topPicks: [
      { symbol: "GOLDBEES", return: 12.5 },
      { symbol: "LIQUIDBEES", return: 6.8 },
      { symbol: "HDFCBANK", return: 18.2 },
    ],
    active: false,
    color: "#00695C",
  },
  {
    id: "munger",
    name: "Charlie Munger",
    title: "The Rational Thinker",
    avatar: "CM",
    style: "Quality at Fair Price",
    description: "Focus on quality businesses with strong moats, willing to pay fair prices for exceptional companies.",
    philosophy: ["Invert, always invert", "Quality over cheapness", "Mental models matter"],
    performance1Y: 26.2,
    followers: "18.9K",
    winRate: 76,
    avgHoldingPeriod: "10+ years",
    topPicks: [
      { symbol: "ASIANPAINT", return: 22.4 },
      { symbol: "BAJFINANCE", return: 28.5 },
      { symbol: "HINDUNILVR", return: 15.2 },
    ],
    active: false,
    color: "#BF360C",
  },
  {
    id: "jhunjhunwala",
    name: "Rakesh Jhunjhunwala",
    title: "The Big Bull",
    avatar: "RJ",
    style: "India Growth Story",
    description: "Bullish on India's long-term growth story. Focus on domestic consumption and financials.",
    philosophy: ["India's growth story is intact", "Buy on dips, hold for wealth", "Retail and banking are future"],
    performance1Y: 42.5,
    followers: "52.4K",
    winRate: 70,
    avgHoldingPeriod: "5-7 years",
    topPicks: [
      { symbol: "TITAN", return: 45.2 },
      { symbol: "TATAELXSI", return: 38.5 },
      { symbol: "CRISIL", return: 32.1 },
    ],
    active: true,
    color: "#E65100",
  },
];

export default function LegendsPage() {
  const [selectedAgent, setSelectedAgent] = useState<string | null>("buffett");
  const selectedLegend = legendAgents.find((a) => a.id === selectedAgent);

  return (
    <div className="max-w-6xl">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center"
            style={{ background: `linear-gradient(135deg, ${colors.accent}, ${colors.gain})` }}
          >
            <Bot className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-semibold" style={{ color: colors.textPrimary }}>
              Legend Agents
            </h1>
            <p className="text-sm" style={{ color: colors.textMuted }}>
              AI agents powered by strategies of legendary investors
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Agents List */}
        <div className="space-y-3">
          {legendAgents.map((agent) => {
            const isSelected = selectedAgent === agent.id;
            const isPositive = agent.performance1Y >= 0;
            return (
              <div
                key={agent.id}
                onClick={() => setSelectedAgent(agent.id)}
                className="rounded-xl p-4 cursor-pointer transition-all"
                style={{
                  backgroundColor: isSelected ? colors.bgMint : colors.bg,
                  border: `1px solid ${isSelected ? colors.accent : colors.border}`,
                }}
                onMouseEnter={(e) => {
                  if (!isSelected) e.currentTarget.style.backgroundColor = colors.bgHover;
                }}
                onMouseLeave={(e) => {
                  if (!isSelected) e.currentTarget.style.backgroundColor = colors.bg;
                }}
              >
                <div className="flex items-start gap-3">
                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center text-white font-bold text-sm"
                    style={{ backgroundColor: agent.color }}
                  >
                    {agent.avatar}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
                        {agent.name}
                      </h3>
                      {agent.active && (
                        <span
                          className="w-2 h-2 rounded-full"
                          style={{ backgroundColor: colors.gain }}
                        />
                      )}
                    </div>
                    <p className="text-xs" style={{ color: colors.textMuted }}>
                      {agent.style}
                    </p>
                    <div className="flex items-center gap-3 mt-2">
                      <span
                        className="text-sm font-semibold"
                        style={{ color: isPositive ? colors.gain : colors.loss }}
                      >
                        {isPositive ? "+" : ""}{agent.performance1Y}%
                      </span>
                      <span className="text-xs" style={{ color: colors.textMuted }}>
                        <Users className="w-3 h-3 inline mr-1" />
                        {agent.followers}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Agent Detail */}
        {selectedLegend && (
          <div className="col-span-2 space-y-4">
            {/* Agent Header */}
            <div
              className="rounded-xl p-6"
              style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-4">
                  <div
                    className="w-16 h-16 rounded-xl flex items-center justify-center text-white font-bold text-lg"
                    style={{ backgroundColor: selectedLegend.color }}
                  >
                    {selectedLegend.avatar}
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
                      {selectedLegend.name}
                    </h2>
                    <p className="text-sm" style={{ color: colors.textMuted }}>
                      {selectedLegend.title}
                    </p>
                    <p className="text-sm mt-2 max-w-md" style={{ color: colors.textSecondary }}>
                      {selectedLegend.description}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {selectedLegend.active ? (
                    <button
                      className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium"
                      style={{ backgroundColor: colors.lossBg, color: colors.loss }}
                    >
                      <Pause className="w-4 h-4" />
                      Pause Agent
                    </button>
                  ) : (
                    <button
                      className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium text-white"
                      style={{ backgroundColor: colors.accent }}
                    >
                      <Play className="w-4 h-4" />
                      Activate Agent
                    </button>
                  )}
                  <button
                    className="p-2 rounded-lg"
                    style={{ border: `1px solid ${colors.border}`, color: colors.textMuted }}
                  >
                    <Settings className="w-4 h-4" />
                  </button>
                </div>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-4 gap-4 mt-6 pt-6" style={{ borderTop: `1px solid ${colors.border}` }}>
                <div>
                  <p className="text-xs" style={{ color: colors.textMuted }}>1Y Return</p>
                  <p
                    className="text-xl font-semibold"
                    style={{ color: selectedLegend.performance1Y >= 0 ? colors.gain : colors.loss }}
                  >
                    {selectedLegend.performance1Y >= 0 ? "+" : ""}{selectedLegend.performance1Y}%
                  </p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: colors.textMuted }}>Win Rate</p>
                  <p className="text-xl font-semibold" style={{ color: colors.textPrimary }}>
                    {selectedLegend.winRate}%
                  </p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: colors.textMuted }}>Followers</p>
                  <p className="text-xl font-semibold" style={{ color: colors.textPrimary }}>
                    {selectedLegend.followers}
                  </p>
                </div>
                <div>
                  <p className="text-xs" style={{ color: colors.textMuted }}>Avg Hold Period</p>
                  <p className="text-xl font-semibold" style={{ color: colors.textPrimary }}>
                    {selectedLegend.avgHoldingPeriod}
                  </p>
                </div>
              </div>
            </div>

            {/* Philosophy */}
            <div
              className="rounded-xl p-5"
              style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
            >
              <h3 className="text-sm font-semibold mb-3" style={{ color: colors.textPrimary }}>
                Investment Philosophy
              </h3>
              <div className="space-y-2">
                {selectedLegend.philosophy.map((principle, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <div
                      className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0"
                      style={{ backgroundColor: colors.bgMint }}
                    >
                      <span className="text-xs font-medium" style={{ color: colors.accent }}>
                        {index + 1}
                      </span>
                    </div>
                    <p className="text-sm" style={{ color: colors.textSecondary }}>
                      {principle}
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Top Picks */}
            <div
              className="rounded-xl p-5"
              style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
                  Top Picks
                </h3>
                <button className="text-xs font-medium" style={{ color: colors.accent }}>
                  View All
                </button>
              </div>
              <div className="grid grid-cols-3 gap-3">
                {selectedLegend.topPicks.map((pick) => (
                  <div
                    key={pick.symbol}
                    className="p-4 rounded-xl"
                    style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}` }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold" style={{ color: colors.textPrimary }}>
                        {pick.symbol}
                      </span>
                      <Target className="w-4 h-4" style={{ color: colors.accent }} />
                    </div>
                    <p
                      className="text-lg font-semibold"
                      style={{ color: pick.return >= 0 ? colors.gain : colors.loss }}
                    >
                      {pick.return >= 0 ? "+" : ""}{pick.return}%
                    </p>
                    <p className="text-xs mt-1" style={{ color: colors.textMuted }}>
                      1Y Return
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-3">
              <button
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-medium text-white"
                style={{ background: `linear-gradient(135deg, ${colors.accent}, ${colors.gain})` }}
              >
                <MessageSquare className="w-4 h-4" />
                Chat with {selectedLegend.name.split(" ")[0]}
              </button>
              <button
                className="flex-1 flex items-center justify-center gap-2 py-3 rounded-xl text-sm font-medium"
                style={{ border: `1px solid ${colors.border}`, color: colors.textSecondary }}
              >
                <BarChart3 className="w-4 h-4" />
                View Full Portfolio
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
