"use client";

import { Plus, MoreVertical, Target } from "lucide-react";
import { cn, formatCurrency, getFinScoreColor } from "@/lib/utils";

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
  loss: "#F45B69",
};

interface WatchlistStock {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  finScore: number;
}

export function WatchlistWidget() {
  const watchlist: WatchlistStock[] = [
    {
      symbol: "RELIANCE",
      name: "Reliance Industries",
      price: 2456.75,
      change: 28.5,
      changePercent: 1.17,
      finScore: 8.4,
    },
    {
      symbol: "TCS",
      name: "Tata Consultancy",
      price: 3845.2,
      change: -12.3,
      changePercent: -0.32,
      finScore: 7.9,
    },
    {
      symbol: "HDFCBANK",
      name: "HDFC Bank",
      price: 1678.9,
      change: 15.8,
      changePercent: 0.95,
      finScore: 8.1,
    },
    {
      symbol: "INFY",
      name: "Infosys",
      price: 1523.45,
      change: 8.9,
      changePercent: 0.59,
      finScore: 7.6,
    },
    {
      symbol: "WIPRO",
      name: "Wipro Limited",
      price: 456.8,
      change: -5.2,
      changePercent: -1.13,
      finScore: 6.4,
    },
    {
      symbol: "ICICIBANK",
      name: "ICICI Bank",
      price: 1045.6,
      change: 22.4,
      changePercent: 2.19,
      finScore: 7.8,
    },
  ];

  return (
    <div
      className="rounded-2xl p-6"
      style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
    >
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold" style={{ color: colors.textPrimary }}>
          Watchlist
        </h2>
        <button
          className="p-2 rounded-lg transition-colors"
          style={{ color: colors.textMuted }}
          onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
          onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
        >
          <Plus className="w-5 h-5" />
        </button>
      </div>

      <div className="space-y-2">
        {watchlist.map((stock) => (
          <div
            key={stock.symbol}
            className="flex items-center justify-between p-3 rounded-xl transition cursor-pointer group"
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgHover)}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
          >
            <div className="flex items-center gap-3">
              <div
                className="w-10 h-10 rounded-lg flex items-center justify-center"
                style={{ backgroundColor: colors.bgMint, border: `1px solid ${colors.border}` }}
              >
                <span className="text-xs font-bold" style={{ color: colors.textPrimary }}>
                  {stock.symbol.slice(0, 2)}
                </span>
              </div>
              <div>
                <p className="font-medium text-sm" style={{ color: colors.textPrimary }}>
                  {stock.symbol}
                </p>
                <div className="flex items-center gap-1.5">
                  <Target
                    className={cn("w-3 h-3", getFinScoreColor(stock.finScore))}
                  />
                  <span
                    className={cn("text-xs", getFinScoreColor(stock.finScore))}
                  >
                    {stock.finScore.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="text-right">
                <p className="text-sm font-medium" style={{ color: colors.textPrimary }}>
                  {formatCurrency(stock.price)}
                </p>
                <p
                  className="text-xs"
                  style={{ color: stock.change >= 0 ? colors.gain : colors.loss }}
                >
                  {stock.change >= 0 ? "+" : ""}
                  {stock.changePercent.toFixed(2)}%
                </p>
              </div>
              <button
                className="p-1 rounded-lg opacity-0 group-hover:opacity-100 transition"
                style={{ color: colors.textMuted }}
                onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.bgMint)}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
              >
                <MoreVertical className="w-4 h-4" />
              </button>
            </div>
          </div>
        ))}
      </div>

      <button
        className="w-full mt-4 py-3 rounded-xl border border-dashed text-sm transition-colors"
        style={{ borderColor: colors.border, color: colors.textMuted }}
        onMouseEnter={(e) => {
          e.currentTarget.style.borderColor = colors.accent;
          e.currentTarget.style.color = colors.accent;
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.borderColor = colors.border;
          e.currentTarget.style.color = colors.textMuted;
        }}
      >
        Add more stocks to watchlist
      </button>
    </div>
  );
}
