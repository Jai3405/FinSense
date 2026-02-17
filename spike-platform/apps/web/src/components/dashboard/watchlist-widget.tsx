"use client";

import { Plus, MoreVertical, Target } from "lucide-react";
import { cn, formatCurrency, getFinScoreColor } from "@/lib/utils";
import { useWatchlist } from "@/lib/api/hooks";

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

export function WatchlistWidget() {
  const { data: watchlistData, isLoading, error } = useWatchlist();

  if (isLoading) {
    return (
      <div
        className="rounded-2xl p-6"
        style={{ backgroundColor: colors.bg, border: `1px solid ${colors.border}` }}
      >
        <div className="animate-pulse space-y-4">
          <div className="h-5 w-40 rounded" style={{ backgroundColor: colors.bgMint }} />
          <div className="h-4 w-full rounded" style={{ backgroundColor: colors.bgMint }} />
          <div className="h-4 w-3/4 rounded" style={{ backgroundColor: colors.bgMint }} />
        </div>
      </div>
    );
  }

  const watchlist = watchlistData
    ? watchlistData.map((s) => ({
        symbol: s.symbol,
        name: s.name,
        price: s.price,
        change: s.change,
        changePercent: s.change_percent,
        finScore: s.finscore,
      }))
    : [];

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

      {error || watchlist.length === 0 ? (
        <div className="text-center py-8">
          <p className="text-sm mb-2" style={{ color: colors.textMuted }}>
            No stocks in your watchlist
          </p>
          <p className="text-sm" style={{ color: colors.accent }}>
            Add stocks to track them here
          </p>
        </div>
      ) : (
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
      )}

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
