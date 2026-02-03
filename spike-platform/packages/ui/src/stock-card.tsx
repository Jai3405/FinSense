import * as React from "react";
import { FinScoreBadge } from "./finscore-badge";

export interface StockCardProps {
  symbol: string;
  name: string;
  price: number;
  change: number;
  changePercent: number;
  finScore?: number;
  onClick?: () => void;
}

export function StockCard({
  symbol,
  name,
  price,
  change,
  changePercent,
  finScore,
  onClick,
}: StockCardProps) {
  const isPositive = change >= 0;

  return (
    <div
      onClick={onClick}
      className="flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/10 transition cursor-pointer border border-white/10"
    >
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-white/10 flex items-center justify-center">
          <span className="text-xs font-bold text-white">
            {symbol.slice(0, 2)}
          </span>
        </div>
        <div>
          <p className="font-medium text-white">{symbol}</p>
          <p className="text-sm text-slate-400 truncate max-w-[120px]">{name}</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {finScore !== undefined && <FinScoreBadge score={finScore} size="sm" />}

        <div className="text-right">
          <p className="font-medium text-white">
            ₹{price.toLocaleString("en-IN")}
          </p>
          <p
            className={`text-sm ${
              isPositive ? "text-green-500" : "text-red-500"
            }`}
          >
            {isPositive ? "+" : ""}
            {changePercent.toFixed(2)}%
          </p>
        </div>
      </div>
    </div>
  );
}
