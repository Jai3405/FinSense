"use client";

import Link from "next/link";
import { motion } from "framer-motion";

const footerLinks = {
  Product: [
    { label: "FinScore", href: "#finscore" },
    { label: "Legend Agents", href: "#" },
    { label: "Strategy-GPT", href: "#" },
    { label: "Portfolio Autopilot", href: "#" },
    { label: "Smart Themes", href: "#" },
  ],
  Company: [
    { label: "About", href: "#" },
    { label: "Careers", href: "#" },
    { label: "Blog", href: "#" },
    { label: "Press", href: "#" },
  ],
  Legal: [
    { label: "Privacy Policy", href: "#" },
    { label: "Terms of Service", href: "#" },
    { label: "Risk Disclosure", href: "#" },
    { label: "SEBI Compliance", href: "#" },
  ],
  Support: [
    { label: "Help Center", href: "#" },
    { label: "Contact Us", href: "#" },
    { label: "API Docs", href: "#" },
    { label: "Status", href: "#" },
  ],
};

export function Footer() {
  return (
    <footer className="relative border-t border-white/5">
      <div className="max-w-7xl mx-auto px-6 pt-20 pb-12">
        {/* Top section */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-10 mb-16">
          {/* Brand */}
          <div className="col-span-2 md:col-span-1">
            <Link href="/" className="flex items-center gap-3 mb-6">
              <div className="relative w-10 h-10">
                <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-spike-sage to-spike-accent opacity-80" />
                <div className="absolute inset-[2px] rounded-[10px] bg-spike-dark flex items-center justify-center">
                  <span className="text-spike-mint font-bold text-lg">S</span>
                </div>
              </div>
              <span className="text-xl font-semibold text-spike-cream tracking-tight">
                spike
              </span>
            </Link>
            <p className="text-sm text-spike-muted leading-relaxed max-w-xs">
              India's first AI wealth intelligence platform. Institutional-grade
              analysis for everyone.
            </p>
          </div>

          {/* Links */}
          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h4 className="text-sm font-medium text-spike-cream mb-4">
                {category}
              </h4>
              <ul className="space-y-3">
                {links.map((link) => (
                  <li key={link.label}>
                    <Link
                      href={link.href}
                      className="text-sm text-spike-muted hover:text-spike-cream transition-colors duration-300"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        {/* Divider */}
        <div className="h-px bg-gradient-to-r from-transparent via-spike-sage/20 to-transparent mb-8" />

        {/* Bottom section */}
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-xs text-spike-muted/60 text-center md:text-left">
            <p>SEBI Registered Research Analyst</p>
            <p className="mt-1">
              Investment in securities market are subject to market risks. Read
              all related documents carefully before investing. Past performance
              is not indicative of future returns.
            </p>
          </div>

          <div className="text-xs text-spike-muted/40">
            &copy; {new Date().getFullYear()} Spike Technologies Pvt. Ltd. All
            rights reserved.
          </div>
        </div>
      </div>
    </footer>
  );
}
