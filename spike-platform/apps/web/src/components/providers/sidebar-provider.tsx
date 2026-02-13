"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

interface SidebarContextType {
  isCollapsed: boolean;
  setIsCollapsed: (collapsed: boolean) => void;
  isHovering: boolean;
  setIsHovering: (hovering: boolean) => void;
}

const SidebarContext = createContext<SidebarContextType | undefined>(undefined);

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [isHovering, setIsHovering] = useState(false);
  const [mounted, setMounted] = useState(false);

  // Mark as mounted after hydration
  useEffect(() => {
    setMounted(true);
  }, []);

  // Persist collapsed state (only after mount to avoid SSR issues)
  useEffect(() => {
    if (mounted) {
      try {
        const saved = localStorage.getItem("sidebar-collapsed");
        if (saved) {
          setIsCollapsed(JSON.parse(saved));
        }
      } catch (e) {
        // Ignore localStorage errors
      }
    }
  }, [mounted]);

  useEffect(() => {
    if (mounted) {
      try {
        localStorage.setItem("sidebar-collapsed", JSON.stringify(isCollapsed));
      } catch (e) {
        // Ignore localStorage errors
      }
    }
  }, [isCollapsed, mounted]);

  // Keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "[" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setIsCollapsed((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <SidebarContext.Provider value={{ isCollapsed, setIsCollapsed, isHovering, setIsHovering }}>
      {children}
    </SidebarContext.Provider>
  );
}

export function useSidebar() {
  const context = useContext(SidebarContext);
  if (context === undefined) {
    throw new Error("useSidebar must be used within a SidebarProvider");
  }
  return context;
}
