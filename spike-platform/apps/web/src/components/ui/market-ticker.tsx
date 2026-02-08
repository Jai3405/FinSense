"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";

interface MarketIndex {
  symbol: string;
  name: string;
  value: number;
  change: number;
  change_percent: number;
  high: number;
  low: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchMarketIndices(): Promise<MarketIndex[]> {
  const response = await fetch(`${API_URL}/api/v1/market/indices`);
  if (!response.ok) {
    throw new Error("Failed to fetch market indices");
  }
  return response.json();
}

// Fallback data when API is unavailable
const fallbackData: MarketIndex[] = [
  { symbol: "NIFTY50", name: "NIFTY 50", value: 22456.80, change: 185.40, change_percent: 0.83, high: 22510, low: 22280 },
  { symbol: "SENSEX", name: "SENSEX", value: 73890.25, change: 478.50, change_percent: 0.65, high: 74050, low: 73450 },
  { symbol: "BANKNIFTY", name: "BANK NIFTY", value: 48250.00, change: -185.75, change_percent: -0.38, high: 48600, low: 48100 },
  { symbol: "NIFTYIT", name: "NIFTY IT", value: 38950.00, change: 425.50, change_percent: 1.10, high: 39050, low: 38500 },
  { symbol: "DJI", name: "DOW", value: 38654.42, change: 125.69, change_percent: 0.33, high: 38720, low: 38520 },
  { symbol: "IXIC", name: "NASDAQ", value: 15628.95, change: 183.02, change_percent: 1.19, high: 15680, low: 15450 },
  { symbol: "GOLD", name: "GOLD", value: 2035.80, change: -8.40, change_percent: -0.41, high: 2048, low: 2028 },
  { symbol: "CRUDE", name: "CRUDE", value: 76.85, change: 1.24, change_percent: 1.64, high: 77.5, low: 75.8 },
];

function formatValue(value: number, symbol: string): string {
  if (symbol === "GOLD" || symbol === "CRUDE") {
    return value.toFixed(2);
  }
  return value.toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

function formatChange(change: number, symbol: string): string {
  const absChange = Math.abs(change);
  if (symbol === "GOLD" || symbol === "CRUDE") {
    return absChange.toFixed(2);
  }
  return absChange.toLocaleString("en-IN", { maximumFractionDigits: 2 });
}

function TickerItem({ index }: { index: MarketIndex }) {
  const isPositive = index.change >= 0;
  const color = isPositive ? "#22c55e" : "#ef4444";
  const bgColor = isPositive ? "rgba(34, 197, 94, 0.2)" : "rgba(239, 68, 68, 0.2)";

  return (
    <div className="flex items-center gap-2 px-5">
      {/* Index Name */}
      <span className="text-xs font-semibold uppercase tracking-wider whitespace-nowrap" style={{ color: "rgba(255,255,255,0.7)" }}>
        {index.name}
      </span>

      {/* Value */}
      <span className="text-sm font-bold whitespace-nowrap" style={{ color: "#ffffff" }}>
        {formatValue(index.value, index.symbol)}
      </span>

      {/* Change indicator with arrow */}
      <div
        className="flex items-center gap-1 px-2 py-0.5 rounded"
        style={{
          backgroundColor: bgColor,
          boxShadow: `0 0 10px ${bgColor}`,
        }}
      >
        {/* Arrow */}
        <svg
          className="w-3 h-3"
          fill={color}
          viewBox="0 0 20 20"
          style={{ transform: isPositive ? "none" : "rotate(180deg)" }}
        >
          <path
            fillRule="evenodd"
            d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z"
            clipRule="evenodd"
          />
        </svg>

        {/* Change value and percentage */}
        <span className="text-xs font-bold whitespace-nowrap" style={{ color }}>
          {formatChange(index.change, index.symbol)}
        </span>
        <span className="text-xs font-semibold whitespace-nowrap" style={{ color }}>
          ({isPositive ? "+" : ""}{index.change_percent.toFixed(2)}%)
        </span>
      </div>

      {/* Divider */}
      <div className="w-px h-5 ml-3" style={{ backgroundColor: "rgba(255,255,255,0.1)" }} />
    </div>
  );
}

export function MarketTicker() {
  const { data, isLoading } = useQuery({
    queryKey: ["market-indices"],
    queryFn: fetchMarketIndices,
    refetchInterval: 30000, // Refresh every 30 seconds
    retry: 2,
  });

  const indices = data || fallbackData;

  if (isLoading) {
    return (
      <motion.div
        className="px-6 py-3 rounded-xl border border-white/10"
        style={{ background: "rgba(10, 18, 16, 0.8)", backdropFilter: "blur(24px)" }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.7 }}
      >
        <div className="flex items-center gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="flex items-center gap-2">
              <div className="h-4 w-16 bg-white/10 rounded animate-pulse" />
              <div className="h-4 w-20 bg-white/10 rounded animate-pulse" />
              <div className="h-4 w-16 bg-white/10 rounded animate-pulse" />
            </div>
          ))}
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      className="relative overflow-hidden rounded-xl border border-white/10"
      style={{
        background: "rgba(10, 18, 16, 0.85)",
        backdropFilter: "blur(24px)",
        maxWidth: "min(900px, 90vw)"
      }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.7 }}
    >
      {/* Live indicator */}
      <div className="absolute left-3 top-1/2 -translate-y-1/2 z-20 flex items-center gap-1.5 bg-[rgba(10,18,16,0.95)] px-2 py-1 rounded">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#22c55e] opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-[#22c55e]"></span>
        </span>
        <span className="text-[10px] font-semibold text-white/60 uppercase tracking-wider">Live</span>
      </div>

      {/* Gradient fade edges */}
      <div className="absolute left-0 top-0 bottom-0 w-16 bg-gradient-to-r from-[rgba(10,18,16,1)] via-[rgba(10,18,16,0.8)] to-transparent z-10 pointer-events-none" />
      <div className="absolute right-0 top-0 bottom-0 w-12 bg-gradient-to-l from-[rgba(10,18,16,1)] to-transparent z-10 pointer-events-none" />

      {/* Marquee container */}
      <div className="py-2.5 overflow-hidden group pl-16">
        <div className="flex animate-marquee group-hover:[animation-play-state:paused]">
          {/* First set of items */}
          {indices.map((index) => (
            <TickerItem key={`first-${index.symbol}`} index={index} />
          ))}
          {/* Duplicate for seamless loop */}
          {indices.map((index) => (
            <TickerItem key={`second-${index.symbol}`} index={index} />
          ))}
        </div>
      </div>

      {/* Inline styles for animation */}
      <style jsx>{`
        @keyframes marquee {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(-50%);
          }
        }
        .animate-marquee {
          animation: marquee 40s linear infinite;
        }
      `}</style>
    </motion.div>
  );
}

export default MarketTicker;
