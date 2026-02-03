"use client";

import { Plus, Search, Sparkles, Briefcase } from "lucide-react";

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
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition ${
            action.highlight
              ? "bg-spike-gradient text-white hover:opacity-90"
              : "bg-white/5 text-slate-300 hover:bg-white/10 hover:text-white border border-white/10"
          }`}
        >
          <action.icon className="w-4 h-4" />
          {action.label}
          {action.shortcut && (
            <kbd className="ml-1 px-1.5 py-0.5 text-xs bg-white/10 rounded">
              {action.shortcut}
            </kbd>
          )}
        </button>
      ))}
    </div>
  );
}
