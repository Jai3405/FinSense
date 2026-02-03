"use client";

import { ArrowDownRight, ArrowUpRight, TrendingUp } from "lucide-react";
import { formatCurrency, formatPercent } from "@/lib/utils";

interface PortfolioData {
  totalValue: number;
  invested: number;
  returns: number;
  returnsPercent: number;
  todayChange: number;
  todayChangePercent: number;
}

export function PortfolioSummary() {
  // TODO: Fetch from API
  const portfolio: PortfolioData = {
    totalValue: 1245678,
    invested: 1000000,
    returns: 245678,
    returnsPercent: 24.57,
    todayChange: 29456,
    todayChangePercent: 2.42,
  };

  const isPositive = portfolio.todayChange >= 0;

  return (
    <div className="rounded-2xl bg-white/5 border border-white/10 p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-white">Portfolio Summary</h2>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-spike-primary/20">
          <TrendingUp className="w-4 h-4 text-spike-primary" />
          <span className="text-sm font-medium text-spike-primary">
            Outperforming NIFTY by 8.2%
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {/* Total Value */}
        <div>
          <p className="text-sm text-slate-400 mb-1">Total Value</p>
          <p className="text-2xl font-bold text-white">
            {formatCurrency(portfolio.totalValue)}
          </p>
        </div>

        {/* Invested */}
        <div>
          <p className="text-sm text-slate-400 mb-1">Invested</p>
          <p className="text-2xl font-bold text-white">
            {formatCurrency(portfolio.invested)}
          </p>
        </div>

        {/* Total Returns */}
        <div>
          <p className="text-sm text-slate-400 mb-1">Total Returns</p>
          <div className="flex items-baseline gap-2">
            <p className="text-2xl font-bold text-spike-bull">
              {formatCurrency(portfolio.returns)}
            </p>
            <span className="text-sm text-spike-bull">
              {formatPercent(portfolio.returnsPercent)}
            </span>
          </div>
        </div>

        {/* Today's Change */}
        <div>
          <p className="text-sm text-slate-400 mb-1">Today's Change</p>
          <div className="flex items-center gap-2">
            <div
              className={`p-1 rounded-lg ${
                isPositive ? "bg-spike-bull/20" : "bg-spike-bear/20"
              }`}
            >
              {isPositive ? (
                <ArrowUpRight className="w-4 h-4 text-spike-bull" />
              ) : (
                <ArrowDownRight className="w-4 h-4 text-spike-bear" />
              )}
            </div>
            <div>
              <p
                className={`text-xl font-bold ${
                  isPositive ? "text-spike-bull" : "text-spike-bear"
                }`}
              >
                {formatCurrency(Math.abs(portfolio.todayChange))}
              </p>
              <span
                className={`text-sm ${
                  isPositive ? "text-spike-bull" : "text-spike-bear"
                }`}
              >
                {formatPercent(portfolio.todayChangePercent)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Mini Chart Placeholder */}
      <div className="mt-6 h-24 rounded-xl bg-gradient-to-r from-spike-bull/10 to-spike-bull/5 flex items-end p-4">
        <div className="flex items-end gap-1 w-full">
          {[40, 45, 38, 52, 48, 60, 55, 65, 70, 68, 75, 80, 78, 85, 90].map(
            (h, i) => (
              <div
                key={i}
                className="flex-1 bg-spike-bull/60 rounded-t"
                style={{ height: `${h}%` }}
              />
            )
          )}
        </div>
      </div>
    </div>
  );
}
