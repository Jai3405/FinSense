"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  Search,
  TrendingUp,
  TrendingDown,
  Flame,
  ArrowUpRight,
  ArrowDownRight,
  Filter,
  Star,
  Plus,
  ChevronRight,
  Sparkles,
  BarChart3,
} from "lucide-react";

// Mock trending stocks
const trendingStocks = [
  { symbol: "TATAMOTORS", name: "Tata Motors Ltd", price: 742.35, change: 8.45, changePercent: 1.15, volume: "12.5M", finScore: 78 },
  { symbol: "ZOMATO", name: "Zomato Ltd", price: 185.60, change: 12.30, changePercent: 7.10, volume: "45.2M", finScore: 72 },
  { symbol: "ADANIGREEN", name: "Adani Green Energy", price: 1245.80, change: -32.50, changePercent: -2.54, volume: "8.9M", finScore: 65 },
  { symbol: "IRFC", name: "Indian Railway Finance", price: 142.25, change: 5.80, changePercent: 4.25, volume: "32.1M", finScore: 70 },
];

// Mock top gainers
const topGainers = [
  { symbol: "ZOMATO", name: "Zomato Ltd", price: 185.60, change: 7.10 },
  { symbol: "IRFC", name: "Indian Railway Finance", price: 142.25, change: 4.25 },
  { symbol: "IDEA", name: "Vodafone Idea Ltd", price: 14.85, change: 3.92 },
  { symbol: "YESBANK", name: "Yes Bank Ltd", price: 24.50, change: 3.15 },
  { symbol: "SUZLON", name: "Suzlon Energy Ltd", price: 42.80, change: 2.85 },
];

// Mock top losers
const topLosers = [
  { symbol: "ADANIGREEN", name: "Adani Green Energy", price: 1245.80, change: -2.54 },
  { symbol: "ADANIENT", name: "Adani Enterprises", price: 2450.60, change: -1.85 },
  { symbol: "PAYTM", name: "One97 Communications", price: 425.30, change: -1.62 },
  { symbol: "NYKAA", name: "FSN E-Commerce", price: 165.40, change: -1.28 },
  { symbol: "POLICYBZR", name: "PB Fintech Ltd", price: 892.15, change: -0.95 },
];

// Mock sectors
const sectors = [
  { name: "Information Technology", change: 2.4, topStock: "TCS", color: "#3B82F6" },
  { name: "Banking", change: -0.8, topStock: "HDFCBANK", color: "#22C55E" },
  { name: "Pharmaceuticals", change: 1.2, topStock: "SUNPHARMA", color: "#EC4899" },
  { name: "Metals", change: 3.1, topStock: "TATASTEEL", color: "#F59E0B" },
  { name: "Auto", change: 1.8, topStock: "TATAMOTORS", color: "#8B5CF6" },
  { name: "Energy", change: -1.5, topStock: "RELIANCE", color: "#EF4444" },
  { name: "FMCG", change: 0.6, topStock: "HINDUNILVR", color: "#06B6D4" },
  { name: "Realty", change: 2.2, topStock: "DLF", color: "#84CC16" },
];

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(value);
}

function SearchBar() {
  const [query, setQuery] = useState("");

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
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search stocks by name, symbol, or sector..."
        className="w-full pl-12 pr-32 py-4 rounded-xl bg-white border border-slate-200 text-slate-900 placeholder:text-slate-400 focus:outline-none focus:border-slate-300 focus:ring-2 focus:ring-spike-primary/20 transition shadow-sm"
      />
      <button className="absolute right-2 top-1/2 -translate-y-1/2 px-4 py-2 rounded-lg bg-slate-100 text-slate-600 text-sm flex items-center gap-2 hover:bg-slate-200 transition">
        <Filter className="w-4 h-4" />
        Filters
      </button>
    </motion.div>
  );
}

function TrendingSection() {
  return (
    <motion.div
      className="rounded-2xl bg-white border border-slate-200 shadow-sm overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <div className="p-4 border-b border-slate-200 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
          <Flame className="w-5 h-5" style={{ color: "#F59E0B" }} />
          Trending Now
        </h2>
        <button className="text-sm text-slate-500 hover:text-slate-900 transition flex items-center gap-1">
          View all <ChevronRight className="w-4 h-4" />
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 divide-y md:divide-y-0 md:divide-x divide-slate-100">
        {trendingStocks.map((stock, index) => {
          const isPositive = stock.change >= 0;
          return (
            <motion.div
              key={stock.symbol}
              className="p-4 hover:bg-slate-50 transition cursor-pointer"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.2 + index * 0.1 }}
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <p className="font-semibold text-slate-900">{stock.symbol}</p>
                  <p className="text-xs text-slate-500 truncate max-w-[120px]">{stock.name}</p>
                </div>
                <button className="p-1.5 rounded-lg hover:bg-slate-100 transition">
                  <Plus className="w-4 h-4 text-slate-400" />
                </button>
              </div>

              <p className="text-xl font-bold text-slate-900 mb-1">{formatCurrency(stock.price)}</p>

              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1">
                  {isPositive ? (
                    <ArrowUpRight className="w-4 h-4" style={{ color: "#22C55E" }} />
                  ) : (
                    <ArrowDownRight className="w-4 h-4" style={{ color: "#EF4444" }} />
                  )}
                  <span
                    className="text-sm font-medium"
                    style={{ color: isPositive ? "#22C55E" : "#EF4444" }}
                  >
                    {isPositive ? "+" : ""}{stock.changePercent.toFixed(2)}%
                  </span>
                </div>
                <span className="text-xs text-slate-400">Vol: {stock.volume}</span>
              </div>

              {/* FinScore badge */}
              <div className="mt-3 flex items-center gap-2">
                <div
                  className="px-2 py-0.5 rounded text-xs font-medium"
                  style={{
                    backgroundColor: stock.finScore >= 70 ? "rgba(34, 197, 94, 0.15)" : "rgba(201, 169, 110, 0.15)",
                    color: stock.finScore >= 70 ? "#16a34a" : "#a16207",
                  }}
                >
                  FinScore: {stock.finScore}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}

function MoversList({ title, stocks, isGainers }: {
  title: string;
  stocks: typeof topGainers;
  isGainers: boolean;
}) {
  const Icon = isGainers ? TrendingUp : TrendingDown;
  const iconColor = isGainers ? "#22C55E" : "#EF4444";

  return (
    <motion.div
      className="rounded-2xl bg-white border border-slate-200 shadow-sm overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: isGainers ? 0.2 : 0.3 }}
    >
      <div className="p-4 border-b border-slate-200 flex items-center gap-2">
        <Icon className="w-5 h-5" style={{ color: iconColor }} />
        <h2 className="text-lg font-semibold text-slate-900">{title}</h2>
      </div>

      <div className="divide-y divide-slate-100">
        {stocks.map((stock, index) => (
          <motion.div
            key={stock.symbol}
            className="p-4 flex items-center justify-between hover:bg-slate-50 transition cursor-pointer"
            initial={{ opacity: 0, x: isGainers ? -20 : 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.3 + index * 0.05 }}
          >
            <div className="flex items-center gap-3">
              <span className="text-sm text-slate-400 w-5">{index + 1}</span>
              <div>
                <p className="font-medium text-slate-900">{stock.symbol}</p>
                <p className="text-xs text-slate-500 truncate max-w-[100px]">{stock.name}</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm font-medium text-slate-900">{formatCurrency(stock.price)}</p>
              <p
                className="text-sm font-medium"
                style={{ color: stock.change >= 0 ? "#22C55E" : "#EF4444" }}
              >
                {stock.change >= 0 ? "+" : ""}{stock.change.toFixed(2)}%
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

function SectorGrid() {
  return (
    <motion.div
      className="rounded-2xl bg-white border border-slate-200 shadow-sm overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.4 }}
    >
      <div className="p-4 border-b border-slate-200 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-slate-400" />
          Sector Performance
        </h2>
        <span className="text-xs text-slate-400">Today</span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-slate-100">
        {sectors.map((sector, index) => {
          const isPositive = sector.change >= 0;
          return (
            <motion.div
              key={sector.name}
              className="p-4 bg-white hover:bg-slate-50 transition cursor-pointer"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: 0.5 + index * 0.05 }}
            >
              <div className="flex items-center gap-2 mb-2">
                <div
                  className="w-2 h-2 rounded-full"
                  style={{ backgroundColor: sector.color }}
                />
                <p className="text-sm font-medium text-slate-900 truncate">{sector.name}</p>
              </div>
              <p
                className="text-lg font-bold"
                style={{ color: isPositive ? "#22C55E" : "#EF4444" }}
              >
                {isPositive ? "+" : ""}{sector.change.toFixed(1)}%
              </p>
              <p className="text-xs text-slate-400 mt-1">Top: {sector.topStock}</p>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}

function QuickActions() {
  const actions = [
    { label: "AI Stock Picks", icon: Sparkles, description: "Get personalized recommendations" },
    { label: "My Watchlist", icon: Star, description: "Track your favorite stocks" },
  ];

  return (
    <motion.div
      className="grid grid-cols-1 md:grid-cols-2 gap-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.5 }}
    >
      {actions.map((action, index) => {
        const Icon = action.icon;
        return (
          <div
            key={action.label}
            className="rounded-2xl bg-white border border-slate-200 p-6 shadow-sm hover:shadow-md hover:border-slate-300 transition cursor-pointer group"
          >
            <div className="flex items-start gap-4">
              <div
                className="w-12 h-12 rounded-xl flex items-center justify-center"
                style={{ backgroundColor: "rgba(34, 197, 94, 0.1)" }}
              >
                <Icon className="w-6 h-6" style={{ color: "#16a34a" }} />
              </div>
              <div className="flex-1">
                <p className="font-semibold text-slate-900 group-hover:text-green-600 transition">
                  {action.label}
                </p>
                <p className="text-sm text-slate-500 mt-1">{action.description}</p>
              </div>
              <ChevronRight className="w-5 h-5 text-slate-300 group-hover:text-slate-500 transition" />
            </div>
          </div>
        );
      })}
    </motion.div>
  );
}

export default function DiscoverPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Discover</h1>
        <p className="text-slate-500 text-sm mt-1">
          Explore trending stocks, sectors, and investment opportunities
        </p>
      </div>

      {/* Search Bar */}
      <SearchBar />

      {/* Trending Section */}
      <TrendingSection />

      {/* Top Movers */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <MoversList title="Top Gainers" stocks={topGainers} isGainers={true} />
        <MoversList title="Top Losers" stocks={topLosers} isGainers={false} />
      </div>

      {/* Sector Grid */}
      <SectorGrid />

      {/* Quick Actions */}
      <QuickActions />
    </div>
  );
}
