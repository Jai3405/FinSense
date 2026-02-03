"use client";

import { Plus, MoreVertical, Target } from "lucide-react";
import { cn, formatCurrency, getFinScoreColor } from "@/lib/utils";

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
    <div className="rounded-2xl bg-white/5 border border-white/10 p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-white">Watchlist</h2>
        <button className="p-2 rounded-lg hover:bg-white/5 transition">
          <Plus className="w-5 h-5 text-slate-400" />
        </button>
      </div>

      <div className="space-y-2">
        {watchlist.map((stock) => (
          <div
            key={stock.symbol}
            className="flex items-center justify-between p-3 rounded-xl hover:bg-white/5 transition cursor-pointer group"
          >
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
                <span className="text-xs font-bold text-white">
                  {stock.symbol.slice(0, 2)}
                </span>
              </div>
              <div>
                <p className="font-medium text-white text-sm">{stock.symbol}</p>
                <div className="flex items-center gap-1.5">
                  <Target
                    className={cn(
                      "w-3 h-3",
                      getFinScoreColor(stock.finScore)
                    )}
                  />
                  <span
                    className={cn(
                      "text-xs",
                      getFinScoreColor(stock.finScore)
                    )}
                  >
                    {stock.finScore.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="text-right">
                <p className="text-sm font-medium text-white">
                  {formatCurrency(stock.price)}
                </p>
                <p
                  className={cn(
                    "text-xs",
                    stock.change >= 0 ? "text-spike-bull" : "text-spike-bear"
                  )}
                >
                  {stock.change >= 0 ? "+" : ""}
                  {stock.changePercent.toFixed(2)}%
                </p>
              </div>
              <button className="p-1 rounded-lg hover:bg-white/10 opacity-0 group-hover:opacity-100 transition">
                <MoreVertical className="w-4 h-4 text-slate-400" />
              </button>
            </div>
          </div>
        ))}
      </div>

      <button className="w-full mt-4 py-3 rounded-xl border border-dashed border-white/20 text-slate-400 text-sm hover:border-white/40 hover:text-white transition">
        Add more stocks to watchlist
      </button>
    </div>
  );
}
