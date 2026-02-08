"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  BarChart3,
  Bot,
  Briefcase,
  Compass,
  Home,
  Layers,
  LineChart,
  Settings,
  Sparkles,
  Target,
  Zap,
} from "lucide-react";

// Color constants
const colors = {
  bg: "#FFFFFF",
  bgHover: "#F0FBF9",
  bgActive: "#E5F7F4",
  border: "#B8DDD7",
  textPrimary: "#0D3331",
  textSecondary: "#3D6B66",
  textMuted: "#6B9B94",
  accent: "#00897B",
  accentBg: "#E0F2F1",
};

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
    <aside
      className="fixed inset-y-0 left-0 z-50 hidden w-64 flex-col lg:flex"
      style={{ backgroundColor: colors.bg, borderRight: `1px solid ${colors.border}` }}
    >
      {/* Logo */}
      <div
        className="flex h-14 items-center px-5"
        style={{ borderBottom: `1px solid ${colors.border}` }}
      >
        <Link href="/dashboard">
          <span
            className="font-serif text-2xl font-semibold tracking-tight"
            style={{ color: colors.accent }}
          >
            spike
          </span>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 px-3">
        <div className="space-y-0.5">
          {navigation.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.name}
                href={item.href}
                className="flex items-center gap-3 px-3 py-2.5 text-sm font-medium rounded-lg transition-colors"
                style={{
                  backgroundColor: isActive ? colors.bgActive : "transparent",
                  color: isActive ? colors.accent : colors.textSecondary,
                }}
                onMouseEnter={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = colors.bgHover;
                }}
                onMouseLeave={(e) => {
                  if (!isActive) e.currentTarget.style.backgroundColor = "transparent";
                }}
              >
                <item.icon className="h-[18px] w-[18px]" />
                {item.name}
              </Link>
            );
          })}
        </div>

        {/* AI Tools Section */}
        <div className="mt-8">
          <div className="px-3 mb-2">
            <span
              className="text-xs font-medium uppercase tracking-wider"
              style={{ color: colors.textMuted }}
            >
              AI Tools
            </span>
          </div>
          <div className="space-y-0.5">
            {aiTools.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className="flex items-center justify-between px-3 py-2.5 text-sm font-medium rounded-lg transition-colors"
                  style={{
                    backgroundColor: isActive ? colors.bgActive : "transparent",
                    color: isActive ? colors.accent : colors.textSecondary,
                  }}
                  onMouseEnter={(e) => {
                    if (!isActive) e.currentTarget.style.backgroundColor = colors.bgHover;
                  }}
                  onMouseLeave={(e) => {
                    if (!isActive) e.currentTarget.style.backgroundColor = "transparent";
                  }}
                >
                  <div className="flex items-center gap-3">
                    <item.icon className="h-[18px] w-[18px]" />
                    {item.name}
                  </div>
                  {item.badge && (
                    <span
                      className="px-1.5 py-0.5 text-[10px] font-semibold rounded text-white"
                      style={{ backgroundColor: colors.accent }}
                    >
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
      <div className="p-3" style={{ borderTop: `1px solid ${colors.border}` }}>
        <Link
          href="/dashboard/settings"
          className="flex items-center gap-3 px-3 py-2.5 text-sm font-medium rounded-lg transition-colors"
          style={{
            backgroundColor: pathname === "/dashboard/settings" ? colors.bgActive : "transparent",
            color: pathname === "/dashboard/settings" ? colors.accent : colors.textSecondary,
          }}
        >
          <Settings className="h-[18px] w-[18px]" />
          Settings
        </Link>
      </div>
    </aside>
  );
}
