"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  BarChart3,
  Bot,
  Briefcase,
  ChevronDown,
  Compass,
  Home,
  Layers,
  LineChart,
  Settings,
  Sparkles,
  Target,
  Wallet,
  Zap,
} from "lucide-react";

const navigation = [
  { name: "Dashboard", href: "/dashboard", icon: Home },
  { name: "Discover", href: "/dashboard/discover", icon: Compass },
  { name: "FinScore", href: "/dashboard/finscore", icon: Target },
  { name: "Portfolio", href: "/dashboard/portfolio", icon: Briefcase },
  { name: "Themes", href: "/dashboard/themes", icon: Layers },
  { name: "Screener", href: "/dashboard/screener", icon: BarChart3 },
  { name: "Charts", href: "/dashboard/charts", icon: LineChart },
];

const aiTools = [
  { name: "Strategy-GPT", href: "/dashboard/strategy", icon: Sparkles },
  { name: "Legend Agents", href: "/dashboard/legends", icon: Bot },
  { name: "Autopilot", href: "/dashboard/autopilot", icon: Zap, badge: "Pro" },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed inset-y-0 left-0 z-50 hidden w-72 flex-col border-r border-white/10 bg-slate-950 lg:flex">
      {/* Logo */}
      <div className="flex h-16 items-center gap-2 border-b border-white/10 px-6">
        <div className="w-10 h-10 rounded-xl bg-spike-gradient flex items-center justify-center">
          <Zap className="w-6 h-6 text-white" />
        </div>
        <span className="text-2xl font-bold text-white">SPIKE</span>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 p-4">
        <div className="space-y-1">
          {navigation.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all",
                  isActive
                    ? "bg-spike-primary/20 text-white"
                    : "text-slate-400 hover:bg-white/5 hover:text-white"
                )}
              >
                <item.icon
                  className={cn("h-5 w-5", isActive && "text-spike-primary")}
                />
                {item.name}
              </Link>
            );
          })}
        </div>

        {/* AI Tools Section */}
        <div className="pt-6">
          <div className="flex items-center justify-between px-4 py-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
              AI Tools
            </span>
          </div>
          <div className="space-y-1">
            {aiTools.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "flex items-center justify-between gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all",
                    isActive
                      ? "bg-spike-primary/20 text-white"
                      : "text-slate-400 hover:bg-white/5 hover:text-white"
                  )}
                >
                  <div className="flex items-center gap-3">
                    <item.icon
                      className={cn(
                        "h-5 w-5",
                        isActive && "text-spike-primary"
                      )}
                    />
                    {item.name}
                  </div>
                  {item.badge && (
                    <span className="px-2 py-0.5 text-xs rounded-full bg-spike-gradient text-white">
                      {item.badge}
                    </span>
                  )}
                </Link>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Bottom Section */}
      <div className="border-t border-white/10 p-4">
        <Link
          href="/dashboard/settings"
          className={cn(
            "flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all",
            pathname === "/dashboard/settings"
              ? "bg-spike-primary/20 text-white"
              : "text-slate-400 hover:bg-white/5 hover:text-white"
          )}
        >
          <Settings className="h-5 w-5" />
          Settings
        </Link>

        {/* Portfolio Summary */}
        <div className="mt-4 rounded-xl bg-white/5 p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-slate-400">Portfolio Value</span>
            <Wallet className="w-4 h-4 text-slate-400" />
          </div>
          <div className="text-2xl font-bold text-white">₹12,45,678</div>
          <div className="flex items-center gap-1 text-sm text-spike-bull">
            <span>+2.4%</span>
            <span className="text-slate-400">today</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
