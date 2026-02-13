"use client";

import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { SidebarProvider, useSidebar } from "@/components/providers/sidebar-provider";
import { Sidebar } from "./sidebar";
import { TopBar } from "./topbar";

function DashboardContent({ children }: { children: ReactNode }) {
  const { isCollapsed, isHovering } = useSidebar();
  const showExpanded = !isCollapsed || isHovering;

  return (
    <div
      className={cn(
        "transition-all duration-300 ease-out",
        showExpanded ? "lg:pl-64" : "lg:pl-[72px]"
      )}
    >
      <TopBar />
      <main className="py-5 px-6">{children}</main>
    </div>
  );
}

export function DashboardShell({ children }: { children: ReactNode }) {
  return (
    <SidebarProvider>
      <div className="min-h-screen" style={{ backgroundColor: "#F5FFFC" }}>
        <Sidebar />
        <DashboardContent>{children}</DashboardContent>
      </div>
    </SidebarProvider>
  );
}
