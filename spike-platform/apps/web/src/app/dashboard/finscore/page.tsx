"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Search,
  TrendingUp,
  TrendingDown,
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
  Star,
  Info,
} from "lucide-react";

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
  { symbol: "MARUTI", name: "Maruti Suzuki India", score: 76, signal: "HOLD", confidence: 68 },
  { symbol: "LT", name: "Larsen & Toubro", score: 74, signal: "HOLD", confidence: 65 },
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
    { name: "Quality", score: 95, icon: Shield, description: "ROE, ROCE, Debt ratios" },
    { name: "Momentum", score: 88, icon: Zap, description: "RSI, MACD, Trend strength" },
    { name: "Value", score: 82, icon: DollarSign, description: "PE, PB, EV/EBITDA" },
    { name: "Sentiment", score: 90, icon: MessageSquare, description: "News & social analysis" },
    { name: "Risk", score: 94, icon: AlertTriangle, description: "Volatility, Beta, Drawdown" },
    { name: "Flow", score: 91, icon: Users, description: "FII/DII activity, Delivery %" },
    { name: "Regime", score: 85, icon: Compass, description: "Market alignment" },
    { name: "Sector", score: 88, icon: BarChart3, description: "Sector dynamics" },
    { name: "Technical", score: 93, icon: Activity, description: "Chart patterns, Signals" },
  ],
};

function getScoreColor(score: number): string {
  if (score >= 80) return "#22C55E"; // Green - Excellent
  if (score >= 60) return "#7DCEA0"; // Sage - Good
  if (score >= 40) return "#C9A96E"; // Gold - Neutral
  if (score >= 20) return "#D4845E"; // Orange - Poor
  return "#EF4444"; // Red - Bad
}

function getSignalColor(signal: string): { bg: string; text: string } {
  switch (signal) {
    case "STRONG BUY":
      return { bg: "rgba(34, 197, 94, 0.2)", text: "#22C55E" };
    case "BUY":
      return { bg: "rgba(125, 206, 160, 0.2)", text: "#7DCEA0" };
    case "HOLD":
      return { bg: "rgba(201, 169, 110, 0.2)", text: "#C9A96E" };
    case "SELL":
      return { bg: "rgba(212, 132, 94, 0.2)", text: "#D4845E" };
    case "STRONG SELL":
      return { bg: "rgba(239, 68, 68, 0.2)", text: "#EF4444" };
    default:
      return { bg: "rgba(255, 255, 255, 0.1)", text: "#ffffff" };
  }
}

function SearchBar({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <motion.div
      className="relative"
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Search for a stock to see its FinScore..."
        className="w-full pl-12 pr-4 py-4 rounded-xl bg-white border border-slate-200 text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-slate-300 focus:ring-2 focus:ring-spike-primary/20 transition shadow-sm"
      />
    </motion.div>
  );
}

function ScoreCircle({ score, size = "large" }: { score: number; size?: "large" | "small" }) {
  const color = getScoreColor(score);
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (score / 100) * circumference;
  const dimensions = size === "large" ? { width: 180, height: 180 } : { width: 60, height: 60 };
  const radius = size === "large" ? 45 : 22;
  const strokeWidth = size === "large" ? 8 : 4;
  const viewBox = size === "large" ? "0 0 112 112" : "0 0 56 56";
  const center = size === "large" ? 56 : 28;

  return (
    <div className="relative" style={dimensions}>
      <svg className="w-full h-full -rotate-90" viewBox={viewBox}>
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="rgba(0,0,0,0.08)"
          strokeWidth={strokeWidth}
        />
        <motion.circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={2 * Math.PI * radius}
          initial={{ strokeDashoffset: 2 * Math.PI * radius }}
          animate={{ strokeDashoffset: (2 * Math.PI * radius) - (score / 100) * (2 * Math.PI * radius) }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          style={{ filter: `drop-shadow(0 0 10px ${color})` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <motion.span
          className="font-bold text-slate-900"
          style={{ fontSize: size === "large" ? "3rem" : "1rem" }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.8 }}
        >
          {score}
        </motion.span>
        {size === "large" && (
          <span className="text-xs text-slate-500">FinScore</span>
        )}
      </div>
    </div>
  );
}

function RankingsList({ stocks, selectedSymbol, onSelect }: {
  stocks: typeof topRankedStocks;
  selectedSymbol: string;
  onSelect: (symbol: string) => void;
}) {
  return (
    <motion.div
      className="rounded-2xl bg-white border border-slate-200 shadow-sm overflow-hidden"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="p-4 border-b border-slate-200 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
          <Star className="w-5 h-5" style={{ color: "#C9A96E" }} />
          Top Rated Stocks
        </h2>
      </div>

      <div className="divide-y divide-slate-100">
        {stocks.map((stock, index) => {
          const isSelected = stock.symbol === selectedSymbol;
          const signalColors = getSignalColor(stock.signal);

          return (
            <motion.div
              key={stock.symbol}
              className={`p-4 cursor-pointer transition ${isSelected ? "bg-slate-100" : "hover:bg-slate-50"}`}
              onClick={() => onSelect(stock.symbol)}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span className="text-sm text-slate-400 w-6">{index + 1}.</span>
                  <div>
                    <p className="font-semibold text-slate-900">{stock.symbol}</p>
                    <p className="text-xs text-slate-500 truncate max-w-[120px]">{stock.name}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="text-right">
                    <p className="text-lg font-bold" style={{ color: getScoreColor(stock.score) }}>
                      {stock.score}
                    </p>
                  </div>
                  <ChevronRight className={`w-4 h-4 text-slate-300 transition ${isSelected ? "text-slate-600" : ""}`} />
                </div>
              </div>
              {/* Score bar */}
              <div className="mt-2 ml-9 h-1.5 bg-slate-100 rounded-full overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ backgroundColor: getScoreColor(stock.score) }}
                  initial={{ width: 0 }}
                  animate={{ width: `${stock.score}%` }}
                  transition={{ duration: 0.8, delay: 0.2 + index * 0.05 }}
                />
              </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}

function FinScoreDetail({ data }: { data: typeof mockFinScoreDetail }) {
  const signalColors = getSignalColor(data.signal);

  return (
    <motion.div
      className="rounded-2xl bg-white border border-slate-200 shadow-sm p-6"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">{data.symbol}</h2>
          <p className="text-slate-500">{data.name}</p>
          <p className="text-xs text-slate-400 mt-1">Updated {data.lastUpdated}</p>
        </div>
        <span
          className="px-3 py-1.5 rounded-lg text-sm font-semibold"
          style={{ backgroundColor: signalColors.bg, color: signalColors.text }}
        >
          {data.signal}
        </span>
      </div>

      {/* Score Circle */}
      <div className="flex justify-center mb-6">
        <ScoreCircle score={data.score} size="large" />
      </div>

      {/* Confidence */}
      <div className="text-center mb-8">
        <p className="text-slate-500 text-sm">Confidence Level</p>
        <p className="text-xl font-semibold text-slate-900">{data.confidence}%</p>
      </div>

      {/* Dimensions */}
      <div>
        <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wider mb-4">
          9-Dimension Breakdown
        </h3>
        <div className="space-y-3">
          {data.dimensions.map((dim, index) => {
            const Icon = dim.icon;
            return (
              <motion.div
                key={dim.name}
                className="flex items-center gap-3"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: 0.5 + index * 0.05 }}
              >
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center"
                  style={{ backgroundColor: "rgba(0,0,0,0.05)" }}
                >
                  <Icon className="w-4 h-4" style={{ color: getScoreColor(dim.score) }} />
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-slate-900">{dim.name}</span>
                    <span className="text-sm font-semibold" style={{ color: getScoreColor(dim.score) }}>
                      {dim.score}
                    </span>
                  </div>
                  <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ backgroundColor: getScoreColor(dim.score) }}
                      initial={{ width: 0 }}
                      animate={{ width: `${dim.score}%` }}
                      transition={{ duration: 0.6, delay: 0.6 + index * 0.05 }}
                    />
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </motion.div>
  );
}

export default function FinScorePage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedStock, setSelectedStock] = useState("HDFCBANK");

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">FinScore</h1>
        <p className="text-slate-500 text-sm mt-1">
          AI-powered stock ratings combining 9 dimensions of analysis
        </p>
      </div>

      {/* Search Bar */}
      <SearchBar value={searchQuery} onChange={setSearchQuery} />

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Rankings List */}
        <RankingsList
          stocks={topRankedStocks}
          selectedSymbol={selectedStock}
          onSelect={setSelectedStock}
        />

        {/* Detail View */}
        <FinScoreDetail data={mockFinScoreDetail} />
      </div>

      {/* Info Banner */}
      <motion.div
        className="rounded-xl bg-slate-50 border border-slate-200 p-4 flex items-start gap-3"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.8 }}
      >
        <Info className="w-5 h-5 text-slate-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="text-sm text-slate-600">
            FinScore is a proprietary 0-100 rating that combines fundamental analysis, technical indicators,
            market sentiment, and AI-driven insights to provide actionable investment signals.
          </p>
          <button className="text-sm mt-2 font-medium" style={{ color: "#16a34a" }}>
            Learn more about FinScore methodology →
          </button>
        </div>
      </motion.div>
    </div>
  );
}
