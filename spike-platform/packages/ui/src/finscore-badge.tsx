import * as React from "react";

export interface FinScoreBadgeProps {
  score: number;
  size?: "sm" | "md" | "lg";
  showIcon?: boolean;
}

function getScoreColor(score: number): { bg: string; text: string } {
  if (score >= 8) return { bg: "bg-green-500/20", text: "text-green-500" };
  if (score >= 6) return { bg: "bg-lime-500/20", text: "text-lime-500" };
  if (score >= 4) return { bg: "bg-yellow-500/20", text: "text-yellow-500" };
  if (score >= 2) return { bg: "bg-orange-500/20", text: "text-orange-500" };
  return { bg: "bg-red-500/20", text: "text-red-500" };
}

const sizeMap = {
  sm: "px-1.5 py-0.5 text-xs",
  md: "px-2 py-1 text-sm",
  lg: "px-3 py-1.5 text-base",
};

export function FinScoreBadge({
  score,
  size = "md",
  showIcon = true,
}: FinScoreBadgeProps) {
  const { bg, text } = getScoreColor(score);

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-lg font-medium ${bg} ${text} ${sizeMap[size]}`}
    >
      {showIcon && (
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className="w-3.5 h-3.5"
        >
          <circle cx="12" cy="12" r="10" />
          <path d="M12 6v6l4 2" />
        </svg>
      )}
      {score.toFixed(1)}
    </span>
  );
}
