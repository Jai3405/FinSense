"use client";

import { useState } from "react";
import { ArrowUpRight, ArrowDownRight, Target } from "lucide-react";
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

  const gainers: Stock[] = [
    {
      symbol: "TATAELXSI",
      name: "Tata Elxsi",
      price: 7245.5,
      change: 542.3,
      changePercent: 8.09,
      volume: "2.3L",
      finScore: 8.2,
    },
    {
      symbol: "IRCTC",
      name: "IRCTC",
      price: 892.75,
      change: 54.25,
      changePercent: 6.47,
      volume: "5.1L",
      finScore: 7.4,
    },
    {
      symbol: "ZOMATO",
      name: "Zomato",
      price: 178.9,
      change: 9.8,
      changePercent: 5.79,
      volume: "12.8L",
      finScore: 6.8,
    },
    {
      symbol: "PAYTM",
      name: "One97 Comm.",
      price: 845.6,
      change: 42.3,
      changePercent: 5.27,
      volume: "8.4L",
      finScore: 5.2,
    },
    {
      symbol: "ADANIENT",
      name: "Adani Ent.",
      price: 2890.45,
      change: 125.8,
      changePercent: 4.55,
      volume: "3.2L",
      finScore: 6.1,
    },
  ];

  const losers: Stock[] = [
    {
      symbol: "TATASTEEL",
      name: "Tata Steel",
      price: 142.35,
      change: -8.45,
      changePercent: -5.6,
      volume: "15.2L",
      finScore: 5.8,
    },
    {
      symbol: "HINDALCO",
      name: "Hindalco",
      price: 485.2,
      change: -24.8,
      changePercent: -4.86,
      volume: "6.8L",
      finScore: 6.2,
    },
    {
      symbol: "JSWSTEEL",
      name: "JSW Steel",
      price: 845.6,
      change: -38.5,
      changePercent: -4.35,
      volume: "4.5L",
      finScore: 5.5,
    },
    {
      symbol: "COALINDIA",
      name: "Coal India",
      price: 428.9,
      change: -18.2,
      changePercent: -4.07,
      volume: "7.2L",
      finScore: 6.8,
    },
    {
      symbol: "VEDL",
      name: "Vedanta",
      price: 298.45,
      change: -11.55,
      changePercent: -3.73,
      volume: "9.1L",
      finScore: 4.9,
    },
  ];

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
    </div>
  );
}
