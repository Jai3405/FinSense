import * as React from "react";

export interface PriceChangeProps {
  value: number;
  percent?: number;
  showIcon?: boolean;
  size?: "sm" | "md" | "lg";
}

const sizeMap = {
  sm: "text-xs",
  md: "text-sm",
  lg: "text-base",
};

export function PriceChange({
  value,
  percent,
  showIcon = true,
  size = "md",
}: PriceChangeProps) {
  const isPositive = value >= 0;
  const color = isPositive ? "text-green-500" : "text-red-500";

  return (
    <span className={`inline-flex items-center gap-1 ${color} ${sizeMap[size]}`}>
      {showIcon && (
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`w-4 h-4 ${isPositive ? "" : "rotate-180"}`}
        >
          <path d="M12 19V5M5 12l7-7 7 7" />
        </svg>
      )}
      {isPositive ? "+" : ""}
      {value.toFixed(2)}
      {percent !== undefined && (
        <span className="text-slate-400 ml-1">
          ({isPositive ? "+" : ""}{percent.toFixed(2)}%)
        </span>
      )}
    </span>
  );
}
