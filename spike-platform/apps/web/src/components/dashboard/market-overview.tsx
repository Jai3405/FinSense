"use client";

import { TrendingUp, TrendingDown, Activity } from "lucide-react";

interface SectorData {
  name: string;
  change: number;
}

export function MarketOverview() {
  const sectors: SectorData[] = [
    { name: "IT", change: 2.4 },
    { name: "Banks", change: -0.8 },
    { name: "Pharma", change: 1.2 },
    { name: "Auto", change: 0.5 },
    { name: "FMCG", change: -0.3 },
    { name: "Metal", change: 3.1 },
    { name: "Energy", change: 1.8 },
    { name: "Realty", change: -1.5 },
  ];

  const marketBreadth = {
    advances: 1245,
    declines: 892,
    unchanged: 156,
  };

  const regimeStatus = {
    current: "Bullish",
    confidence: 78,
    trend: "Uptrend",
  };

  return (
    <div className="rounded-2xl bg-white/5 border border-white/10 p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-lg font-semibold text-white">Market Overview</h2>
        <div className="flex items-center gap-2 text-sm">
          <Activity className="w-4 h-4 text-spike-accent" />
          <span className="text-slate-400">Live</span>
        </div>
      </div>

      {/* Regime Indicator */}
      <div className="flex items-center gap-4 p-4 rounded-xl bg-spike-bull/10 border border-spike-bull/20 mb-6">
        <div className="w-12 h-12 rounded-xl bg-spike-bull/20 flex items-center justify-center">
          <TrendingUp className="w-6 h-6 text-spike-bull" />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="text-lg font-semibold text-white">
              {regimeStatus.current} Regime
            </span>
            <span className="px-2 py-0.5 text-xs rounded-full bg-spike-bull/20 text-spike-bull">
              {regimeStatus.confidence}% confidence
            </span>
          </div>
          <p className="text-sm text-slate-400">
            Market showing sustained {regimeStatus.trend.toLowerCase()} momentum
          </p>
        </div>
      </div>

      {/* Market Breadth */}
      <div className="mb-6">
        <p className="text-sm text-slate-400 mb-3">Market Breadth</p>
        <div className="flex items-center gap-2 h-3 rounded-full overflow-hidden bg-slate-800">
          <div
            className="h-full bg-spike-bull"
            style={{
              width: `${
                (marketBreadth.advances /
                  (marketBreadth.advances +
                    marketBreadth.declines +
                    marketBreadth.unchanged)) *
                100
              }%`,
            }}
          />
          <div
            className="h-full bg-slate-600"
            style={{
              width: `${
                (marketBreadth.unchanged /
                  (marketBreadth.advances +
                    marketBreadth.declines +
                    marketBreadth.unchanged)) *
                100
              }%`,
            }}
          />
          <div
            className="h-full bg-spike-bear"
            style={{
              width: `${
                (marketBreadth.declines /
                  (marketBreadth.advances +
                    marketBreadth.declines +
                    marketBreadth.unchanged)) *
                100
              }%`,
            }}
          />
        </div>
        <div className="flex justify-between mt-2 text-xs">
          <span className="text-spike-bull">
            {marketBreadth.advances} Advances
          </span>
          <span className="text-slate-400">
            {marketBreadth.unchanged} Unchanged
          </span>
          <span className="text-spike-bear">
            {marketBreadth.declines} Declines
          </span>
        </div>
      </div>

      {/* Sector Performance */}
      <div>
        <p className="text-sm text-slate-400 mb-3">Sector Performance</p>
        <div className="grid grid-cols-4 gap-3">
          {sectors.map((sector) => (
            <div
              key={sector.name}
              className="p-3 rounded-xl bg-white/5 border border-white/10 text-center"
            >
              <p className="text-sm text-slate-400 mb-1">{sector.name}</p>
              <p
                className={`text-sm font-semibold ${
                  sector.change >= 0 ? "text-spike-bull" : "text-spike-bear"
                }`}
              >
                {sector.change >= 0 ? "+" : ""}
                {sector.change.toFixed(1)}%
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
