import { Suspense } from "react";
import { PortfolioSummary } from "@/components/dashboard/portfolio-summary";
import { MarketOverview } from "@/components/dashboard/market-overview";
import { TopMovers } from "@/components/dashboard/top-movers";
import { WatchlistWidget } from "@/components/dashboard/watchlist-widget";
import { AIInsights } from "@/components/dashboard/ai-insights";
import { QuickActions } from "@/components/dashboard/quick-actions";

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* Welcome Section */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Good Morning!</h1>
          <p className="text-slate-400">
            Here's what's happening with your investments today.
          </p>
        </div>
        <QuickActions />
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Portfolio & Market */}
        <div className="lg:col-span-2 space-y-6">
          <Suspense fallback={<WidgetSkeleton className="h-48" />}>
            <PortfolioSummary />
          </Suspense>

          <Suspense fallback={<WidgetSkeleton className="h-64" />}>
            <MarketOverview />
          </Suspense>

          <Suspense fallback={<WidgetSkeleton className="h-80" />}>
            <TopMovers />
          </Suspense>
        </div>

        {/* Right Column - Watchlist & AI */}
        <div className="space-y-6">
          <Suspense fallback={<WidgetSkeleton className="h-96" />}>
            <WatchlistWidget />
          </Suspense>

          <Suspense fallback={<WidgetSkeleton className="h-64" />}>
            <AIInsights />
          </Suspense>
        </div>
      </div>
    </div>
  );
}

function WidgetSkeleton({ className }: { className?: string }) {
  return (
    <div
      className={`rounded-2xl bg-white/5 border border-white/10 animate-pulse ${className}`}
    />
  );
}
