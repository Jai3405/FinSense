"use client";

import { Plus, Search, Sparkles } from "lucide-react";

// Color constants
const colors = {
  bg: "#FFFFFF",
  bgHover: "#F0FBF9",
  border: "#B8DDD7",
  textPrimary: "#0D3331",
  textSecondary: "#3D6B66",
  accent: "#00897B",
  gain: "#00B386",
};

export function QuickActions() {
  const actions = [
    {
      label: "Search",
      icon: Search,
      shortcut: "⌘K",
      onClick: () => {},
    },
    {
      label: "Add Stock",
      icon: Plus,
      onClick: () => {},
    },
    {
      label: "Ask AI",
      icon: Sparkles,
      onClick: () => {},
      highlight: true,
    },
  ];

  return (
    <div className="flex items-center gap-2">
      {actions.map((action) => (
        <button
          key={action.label}
          onClick={action.onClick}
          className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all"
          style={
            action.highlight
              ? {
                  background: `linear-gradient(135deg, ${colors.accent}, ${colors.gain})`,
                  color: "#FFFFFF",
                }
              : {
                  backgroundColor: colors.bg,
                  color: colors.textSecondary,
                  border: `1px solid ${colors.border}`,
                }
          }
          onMouseEnter={(e) => {
            if (action.highlight) {
              e.currentTarget.style.opacity = "0.9";
            } else {
              e.currentTarget.style.backgroundColor = colors.bgHover;
              e.currentTarget.style.color = colors.textPrimary;
            }
          }}
          onMouseLeave={(e) => {
            if (action.highlight) {
              e.currentTarget.style.opacity = "1";
            } else {
              e.currentTarget.style.backgroundColor = colors.bg;
              e.currentTarget.style.color = colors.textSecondary;
            }
          }}
        >
          <action.icon className="w-4 h-4" />
          {action.label}
          {action.shortcut && (
            <kbd
              className="ml-1 px-1.5 py-0.5 text-xs rounded"
              style={{ backgroundColor: colors.border, color: colors.textSecondary }}
            >
              {action.shortcut}
            </kbd>
          )}
        </button>
      ))}
    </div>
  );
}
