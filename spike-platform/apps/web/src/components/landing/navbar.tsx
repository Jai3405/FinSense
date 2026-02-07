"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowRight, Menu, X } from "lucide-react";

const navItems = [
  { label: "Features", href: "#features" },
  { label: "FinScore", href: "#finscore" },
  { label: "How it Works", href: "#how-it-works" },
  { label: "Pricing", href: "#pricing" },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 50);
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  return (
    <>
      <motion.header
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
          scrolled
            ? "py-3"
            : "py-5"
        }`}
      >
        <div className="max-w-7xl mx-auto px-6">
          <div
            className={`flex items-center justify-between rounded-2xl px-6 py-3 transition-all duration-500 ${
              scrolled
                ? "glass glow"
                : "bg-transparent"
            }`}
          >
            {/* Logo */}
            <Link href="/" className="group">
              <span className="text-2xl font-bold italic tracking-tight bg-gradient-to-r from-spike-sage via-spike-accent to-spike-mint bg-clip-text text-transparent">
                spike
              </span>
            </Link>

            {/* Desktop Nav */}
            <nav className="hidden md:flex items-center gap-1">
              {navItems.map((item) => (
                <Link
                  key={item.label}
                  href={item.href}
                  className="relative px-4 py-2 text-sm text-spike-muted hover:text-spike-cream transition-colors duration-300 group"
                >
                  {item.label}
                  <span className="absolute bottom-1 left-4 right-4 h-px bg-spike-sage scale-x-0 group-hover:scale-x-100 transition-transform duration-300 origin-left" />
                </Link>
              ))}
            </nav>

            {/* CTA */}
            <div className="hidden md:flex items-center gap-3">
              <Link
                href="/sign-in"
                className="px-4 py-2 text-sm text-spike-muted hover:text-spike-cream transition-colors duration-300"
              >
                Sign in
              </Link>
              <Link
                href="/sign-up"
                className="group relative px-5 py-2.5 rounded-xl overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-spike-sage to-spike-accent opacity-90 group-hover:opacity-100 transition-opacity" />
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
                <span className="relative flex items-center gap-2 text-sm font-medium text-spike-dark">
                  Get Started
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
                </span>
              </Link>
            </div>

            {/* Mobile toggle */}
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="md:hidden p-2 text-spike-muted"
            >
              {mobileOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </motion.header>

      {/* Mobile menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="fixed inset-0 z-40 bg-spike-dark/95 backdrop-blur-xl pt-28 px-6"
          >
            <nav className="space-y-2">
              {navItems.map((item, i) => (
                <motion.div
                  key={item.label}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.1 }}
                >
                  <Link
                    href={item.href}
                    onClick={() => setMobileOpen(false)}
                    className="block px-4 py-4 text-lg text-spike-cream border-b border-white/5"
                  >
                    {item.label}
                  </Link>
                </motion.div>
              ))}
              <div className="pt-6 space-y-3">
                <Link
                  href="/sign-in"
                  className="block text-center py-3 text-spike-muted"
                >
                  Sign in
                </Link>
                <Link
                  href="/sign-up"
                  className="block text-center py-3 rounded-xl bg-gradient-to-r from-spike-sage to-spike-accent text-spike-dark font-medium"
                >
                  Get Started
                </Link>
              </div>
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
