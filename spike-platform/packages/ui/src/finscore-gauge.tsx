import * as React from "react";

export interface FinScoreGaugeProps {
  score: number;
  size?: "sm" | "md" | "lg";
  showLabel?: boolean;
  symbol?: string;
}

function getScoreColor(score: number): string {
  if (score >= 8) return "#22C55E"; // Excellent - Green
  if (score >= 6) return "#84CC16"; // Good - Lime
  if (score >= 4) return "#EAB308"; // Neutral - Yellow
  if (score >= 2) return "#F97316"; // Poor - Orange
  return "#EF4444"; // Bad - Red
}

function getSignal(score: number): string {
  if (score >= 8) return "Strong Buy";
  if (score >= 6) return "Buy";
  if (score >= 4) return "Hold";
  if (score >= 2) return "Sell";
  return "Strong Sell";
}

const sizeMap = {
  sm: { outer: 80, inner: 60, font: "text-xl", label: "text-xs" },
  md: { outer: 120, inner: 90, font: "text-3xl", label: "text-sm" },
  lg: { outer: 180, inner: 140, font: "text-5xl", label: "text-base" },
};

export function FinScoreGauge({
  score,
  size = "md",
  showLabel = true,
  symbol,
}: FinScoreGaugeProps) {
  const color = getScoreColor(score);
  const signal = getSignal(score);
  const { outer, inner, font, label } = sizeMap[size];

  // Calculate rotation for the gauge indicator
  const rotation = (score / 10) * 180 - 90; // -90 to 90 degrees

  return (
    <div className="relative inline-flex flex-col items-center">
      {/* Gauge */}
      <div
        className="relative rounded-full"
        style={{
          width: outer,
          height: outer,
          background: `conic-gradient(
            from 180deg,
            #EF4444 0deg,
            #F97316 36deg,
            #EAB308 72deg,
            #84CC16 108deg,
            #22C55E 144deg,
            #22C55E 180deg,
            transparent 180deg
          )`,
        }}
      >
        {/* Inner circle */}
        <div
          className="absolute bg-slate-900 rounded-full flex items-center justify-center flex-col"
          style={{
            width: inner,
            height: inner,
            top: (outer - inner) / 2,
            left: (outer - inner) / 2,
          }}
        >
          <span className={`${font} font-bold`} style={{ color }}>
            {score.toFixed(1)}
          </span>
          {symbol && (
            <span className={`${label} text-slate-400 mt-1`}>{symbol}</span>
          )}
        </div>

        {/* Indicator needle */}
        <div
          className="absolute w-1 bg-white rounded-full origin-bottom"
          style={{
            height: (outer - inner) / 2 - 4,
            bottom: outer / 2,
            left: outer / 2 - 2,
            transform: `rotate(${rotation}deg)`,
            boxShadow: "0 0 10px rgba(255,255,255,0.5)",
          }}
        />
      </div>

      {/* Label */}
      {showLabel && (
        <div className={`mt-2 font-medium ${label}`} style={{ color }}>
          {signal}
        </div>
      )}
    </div>
  );
}
