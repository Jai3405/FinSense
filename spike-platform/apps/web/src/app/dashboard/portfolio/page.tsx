"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  TrendingUp,
  TrendingDown,
  ArrowUpRight,
  ArrowDownRight,
  PieChart,
  BarChart3,
  Plus,
  Download,
  Filter,
  Activity,
} from "lucide-react";

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
  { symbol: "HDFCBANK", name: "HDFC Bank Ltd", qty: 50, avgPrice: 1520.45, currentPrice: 1685.30, sector: "Banking" },
  { symbol: "TCS", name: "Tata Consultancy Services", qty: 25, avgPrice: 3450.00, currentPrice: 3892.50, sector: "IT" },
  { symbol: "RELIANCE", name: "Reliance Industries", qty: 40, avgPrice: 2380.25, currentPrice: 2545.80, sector: "Oil & Gas" },
  { symbol: "INFY", name: "Infosys Ltd", qty: 60, avgPrice: 1420.00, currentPrice: 1565.25, sector: "IT" },
  { symbol: "ICICIBANK", name: "ICICI Bank Ltd", qty: 80, avgPrice: 925.50, currentPrice: 1012.40, sector: "Banking" },
  { symbol: "TATAMOTORS", name: "Tata Motors Ltd", qty: 100, avgPrice: 625.00, currentPrice: 742.35, sector: "Auto" },
  { symbol: "SUNPHARMA", name: "Sun Pharma Industries", qty: 45, avgPrice: 1180.00, currentPrice: 1245.60, sector: "Pharma" },
  { symbol: "BHARTIARTL", name: "Bharti Airtel Ltd", qty: 35, avgPrice: 1050.00, currentPrice: 1185.90, sector: "Telecom" },
];

const sectorAllocation = [
  { sector: "Banking", percentage: 32, value: 399312, color: "#22C55E" },
  { sector: "IT", percentage: 28, value: 349398, color: "#3B82F6" },
  { sector: "Oil & Gas", percentage: 16, value: 199656, color: "#F59E0B" },
  { sector: "Auto", percentage: 12, value: 149742, color: "#8B5CF6" },
  { sector: "Pharma", percentage: 7, value: 87349, color: "#EC4899" },
  { sector: "Telecom", percentage: 5, value: 62392, color: "#06B6D4" },
];

// Mock performance data (last 30 days)
const performanceData = [
  { date: "Jan 10", value: 1050000 },
  { date: "Jan 12", value: 1065000 },
  { date: "Jan 14", value: 1058000 },
  { date: "Jan 16", value: 1082000 },
  { date: "Jan 18", value: 1095000 },
  { date: "Jan 20", value: 1078000 },
  { date: "Jan 22", value: 1112000 },
  { date: "Jan 24", value: 1145000 },
  { date: "Jan 26", value: 1132000 },
  { date: "Jan 28", value: 1168000 },
  { date: "Jan 30", value: 1185000 },
  { date: "Feb 01", value: 1198000 },
  { date: "Feb 03", value: 1175000 },
  { date: "Feb 05", value: 1215000 },
  { date: "Feb 07", value: 1247850 },
];

function formatCurrency(value: number): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(value);
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 }).format(value);
}

function PortfolioHeader() {
  const isPositiveDay = portfolioSummary.dayChangePercent >= 0;
  const isPositiveTotal = portfolioSummary.totalReturnsPercent >= 0;

  return (
    <motion.div
      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Total Value */}
      <div className="rounded-2xl bg-white border border-slate-200 p-6 shadow-sm col-span-1 lg:col-span-2">
        <p className="text-sm text-slate-500 mb-1">Total Portfolio Value</p>
        <p className="text-3xl font-bold text-slate-900">{formatCurrency(portfolioSummary.totalValue)}</p>
        <div className="flex items-center gap-4 mt-3">
          <div className="flex items-center gap-1">
            {isPositiveDay ? (
              <ArrowUpRight className="w-4 h-4" style={{ color: "#22C55E" }} />
            ) : (
              <ArrowDownRight className="w-4 h-4" style={{ color: "#EF4444" }} />
            )}
            <span style={{ color: isPositiveDay ? "#22C55E" : "#EF4444" }} className="text-sm font-medium">
              {formatCurrency(portfolioSummary.dayChange)} ({isPositiveDay ? "+" : ""}{portfolioSummary.dayChangePercent.toFixed(2)}%)
            </span>
            <span className="text-slate-400 text-sm">today</span>
          </div>
        </div>
      </div>

      {/* Invested Value */}
      <div className="rounded-2xl bg-white border border-slate-200 p-6 shadow-sm">
        <p className="text-sm text-slate-500 mb-1">Invested Amount</p>
        <p className="text-2xl font-bold text-slate-900">{formatCurrency(portfolioSummary.investedValue)}</p>
      </div>

      {/* Total Returns */}
      <div className="rounded-2xl bg-white border border-slate-200 p-6 shadow-sm">
        <p className="text-sm text-slate-500 mb-1">Total Returns</p>
        <p className="text-2xl font-bold" style={{ color: isPositiveTotal ? "#22C55E" : "#EF4444" }}>
          {isPositiveTotal ? "+" : ""}{formatCurrency(portfolioSummary.totalReturns)}
        </p>
        <p className="text-sm mt-1" style={{ color: isPositiveTotal ? "#22C55E" : "#EF4444" }}>
          {isPositiveTotal ? "+" : ""}{portfolioSummary.totalReturnsPercent.toFixed(2)}%
        </p>
      </div>
    </motion.div>
  );
}

function PerformanceChart() {
  const [timeRange, setTimeRange] = useState<"1W" | "1M" | "3M" | "1Y" | "ALL">("1M");

  // Calculate chart dimensions
  const maxValue = Math.max(...performanceData.map(d => d.value));
  const minValue = Math.min(...performanceData.map(d => d.value));
  const range = maxValue - minValue;

  // Generate SVG path for the line
  const points = performanceData.map((d, i) => {
    const x = (i / (performanceData.length - 1)) * 100;
    const y = 100 - ((d.value - minValue) / range) * 80 - 10; // 80% height with 10% padding
    return `${x},${y}`;
  }).join(" ");

  const linePath = `M ${points.split(" ").join(" L ")}`;

  // Create area path (for gradient fill)
  const areaPath = `${linePath} L 100,100 L 0,100 Z`;

  return (
    <motion.div
      className="rounded-2xl bg-white border border-slate-200 p-6 shadow-sm"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.15 }}
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-slate-400" />
          <h2 className="text-lg font-semibold text-slate-900">Performance</h2>
        </div>
        <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-1">
          {(["1W", "1M", "3M", "1Y", "ALL"] as const).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1 rounded-md text-xs font-medium transition ${
                timeRange === range
                  ? "bg-white text-slate-900 shadow-sm"
                  : "text-slate-500 hover:text-slate-700"
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Chart */}
      <div className="relative h-48">
        <svg
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          className="w-full h-full"
        >
          {/* Gradient definition */}
          <defs>
            <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#22C55E" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#22C55E" stopOpacity="0" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[20, 40, 60, 80].map((y) => (
            <line
              key={y}
              x1="0"
              y1={y}
              x2="100"
              y2={y}
              stroke="rgba(0,0,0,0.05)"
              strokeWidth="0.5"
            />
          ))}

          {/* Area fill */}
          <motion.path
            d={areaPath}
            fill="url(#chartGradient)"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.5 }}
          />

          {/* Line */}
          <motion.path
            d={linePath}
            fill="none"
            stroke="#22C55E"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 1.5, ease: "easeOut" }}
            style={{ filter: "drop-shadow(0 0 6px rgba(34, 197, 94, 0.5))" }}
          />

          {/* End point dot */}
          <motion.circle
            cx="100"
            cy={100 - ((performanceData[performanceData.length - 1].value - minValue) / range) * 80 - 10}
            r="2"
            fill="#22C55E"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ duration: 0.3, delay: 1.5 }}
            style={{ filter: "drop-shadow(0 0 4px rgba(34, 197, 94, 0.8))" }}
          />
        </svg>

        {/* Y-axis labels */}
        <div className="absolute left-0 top-0 bottom-0 flex flex-col justify-between text-xs text-slate-400 pr-2">
          <span>₹{(maxValue / 100000).toFixed(1)}L</span>
          <span>₹{(minValue / 100000).toFixed(1)}L</span>
        </div>
      </div>

      {/* X-axis labels */}
      <div className="flex justify-between mt-2 text-xs text-slate-400">
        <span>{performanceData[0].date}</span>
        <span>{performanceData[Math.floor(performanceData.length / 2)].date}</span>
        <span>{performanceData[performanceData.length - 1].date}</span>
      </div>

      {/* Summary */}
      <div className="flex items-center justify-between mt-4 pt-4 border-t border-slate-100">
        <div>
          <p className="text-xs text-slate-500">Period Return</p>
          <p className="text-lg font-semibold" style={{ color: "#22C55E" }}>+₹1,97,850</p>
        </div>
        <div className="text-right">
          <p className="text-xs text-slate-500">Growth</p>
          <p className="text-lg font-semibold" style={{ color: "#22C55E" }}>+18.84%</p>
        </div>
      </div>
    </motion.div>
  );
}

function HoldingsTable() {
  const [sortBy, setSortBy] = useState<string>("value");

  const sortedHoldings = [...holdings].sort((a, b) => {
    const aValue = a.qty * a.currentPrice;
    const bValue = b.qty * b.currentPrice;
    return bValue - aValue;
  });

  return (
    <motion.div
      className="rounded-2xl bg-white border border-slate-200 shadow-sm overflow-hidden"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <div className="p-6 border-b border-slate-200 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-900">Holdings</h2>
        <div className="flex items-center gap-2">
          <button className="px-3 py-1.5 rounded-lg bg-slate-100 border border-slate-200 text-slate-600 text-sm hover:bg-slate-200 transition flex items-center gap-1">
            <Filter className="w-4 h-4" />
            Filter
          </button>
          <button className="px-3 py-1.5 rounded-lg bg-slate-100 border border-slate-200 text-slate-600 text-sm hover:bg-slate-200 transition flex items-center gap-1">
            <Download className="w-4 h-4" />
            Export
          </button>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50">
              <th className="text-left text-xs font-medium text-slate-500 uppercase tracking-wider px-6 py-4">Stock</th>
              <th className="text-right text-xs font-medium text-slate-500 uppercase tracking-wider px-6 py-4">Qty</th>
              <th className="text-right text-xs font-medium text-slate-500 uppercase tracking-wider px-6 py-4">Avg Price</th>
              <th className="text-right text-xs font-medium text-slate-500 uppercase tracking-wider px-6 py-4">LTP</th>
              <th className="text-right text-xs font-medium text-slate-500 uppercase tracking-wider px-6 py-4">Current Value</th>
              <th className="text-right text-xs font-medium text-slate-500 uppercase tracking-wider px-6 py-4">P&L</th>
              <th className="text-right text-xs font-medium text-slate-500 uppercase tracking-wider px-6 py-4">Returns</th>
            </tr>
          </thead>
          <tbody>
            {sortedHoldings.map((holding, index) => {
              const currentValue = holding.qty * holding.currentPrice;
              const investedValue = holding.qty * holding.avgPrice;
              const pnl = currentValue - investedValue;
              const pnlPercent = ((currentValue - investedValue) / investedValue) * 100;
              const isPositive = pnl >= 0;

              return (
                <motion.tr
                  key={holding.symbol}
                  className="border-b border-slate-100 hover:bg-slate-50 transition cursor-pointer"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.05 }}
                >
                  <td className="px-6 py-4">
                    <div>
                      <p className="font-semibold text-slate-900">{holding.symbol}</p>
                      <p className="text-xs text-slate-500">{holding.name}</p>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right text-slate-900">{holding.qty}</td>
                  <td className="px-6 py-4 text-right text-slate-600">₹{formatNumber(holding.avgPrice)}</td>
                  <td className="px-6 py-4 text-right text-slate-900 font-medium">₹{formatNumber(holding.currentPrice)}</td>
                  <td className="px-6 py-4 text-right text-slate-900 font-medium">₹{formatNumber(currentValue)}</td>
                  <td className="px-6 py-4 text-right">
                    <span style={{ color: isPositive ? "#22C55E" : "#EF4444" }} className="font-medium">
                      {isPositive ? "+" : ""}₹{formatNumber(pnl)}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <span
                      className="px-2 py-1 rounded text-xs font-medium"
                      style={{
                        backgroundColor: isPositive ? "rgba(34, 197, 94, 0.2)" : "rgba(239, 68, 68, 0.2)",
                        color: isPositive ? "#22C55E" : "#EF4444",
                      }}
                    >
                      {isPositive ? "+" : ""}{pnlPercent.toFixed(2)}%
                    </span>
                  </td>
                </motion.tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
}

function SectorAllocation() {
  return (
    <motion.div
      className="rounded-2xl bg-white border border-slate-200 p-6 shadow-sm"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
    >
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-slate-900">Sector Allocation</h2>
        <PieChart className="w-5 h-5 text-slate-400" />
      </div>

      {/* Simple bar visualization */}
      <div className="space-y-4">
        {sectorAllocation.map((sector, index) => (
          <motion.div
            key={sector.sector}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.3 + index * 0.05 }}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-sm text-slate-900">{sector.sector}</span>
              <span className="text-sm text-slate-600">{sector.percentage}%</span>
            </div>
            <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                style={{ backgroundColor: sector.color }}
                initial={{ width: 0 }}
                animate={{ width: `${sector.percentage}%` }}
                transition={{ duration: 0.8, delay: 0.4 + index * 0.05 }}
              />
            </div>
            <p className="text-xs text-slate-400 mt-1">{formatCurrency(sector.value)}</p>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

function QuickStats() {
  const stats = [
    { label: "Total Stocks", value: holdings.length.toString(), icon: BarChart3 },
    { label: "Best Performer", value: "TATAMOTORS", subValue: "+18.8%", isPositive: true },
    { label: "Worst Performer", value: "SUNPHARMA", subValue: "+5.6%", isPositive: true },
  ];

  return (
    <motion.div
      className="rounded-2xl bg-white border border-slate-200 p-6 shadow-sm"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
    >
      <h2 className="text-lg font-semibold text-slate-900 mb-6">Quick Stats</h2>
      <div className="space-y-4">
        {stats.map((stat, index) => (
          <div key={stat.label} className="flex items-center justify-between py-3 border-b border-slate-100 last:border-0">
            <span className="text-sm text-slate-500">{stat.label}</span>
            <div className="text-right">
              <p className="text-sm font-medium text-slate-900">{stat.value}</p>
              {stat.subValue && (
                <p className="text-xs" style={{ color: stat.isPositive ? "#22C55E" : "#EF4444" }}>
                  {stat.subValue}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

export default function PortfolioPage() {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Portfolio</h1>
          <p className="text-slate-500 text-sm mt-1">Track your investments and performance</p>
        </div>
        <button
          className="px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 transition"
          style={{ backgroundColor: "#4ade80", color: "#0D1B17" }}
        >
          <Plus className="w-4 h-4" />
          Add Stock
        </button>
      </div>

      {/* Portfolio Header Cards */}
      <PortfolioHeader />

      {/* Performance Chart */}
      <PerformanceChart />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Holdings Table - Spans 2 columns */}
        <div className="lg:col-span-2">
          <HoldingsTable />
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          <SectorAllocation />
          <QuickStats />
        </div>
      </div>
    </div>
  );
}
