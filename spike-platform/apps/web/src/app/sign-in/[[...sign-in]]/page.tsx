"use client";

import { SignIn } from "@clerk/nextjs";
import Link from "next/link";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";

// Animated floating orb component
function FloatingOrb({
  size,
  color,
  initialX,
  initialY,
  duration,
}: {
  size: number;
  color: string;
  initialX: number;
  initialY: number;
  duration: number;
}) {
  return (
    <motion.div
      className="absolute rounded-full blur-3xl opacity-30"
      style={{
        width: size,
        height: size,
        background: color,
        left: `${initialX}%`,
        top: `${initialY}%`,
      }}
      animate={{
        x: [0, 100, -50, 0],
        y: [0, -80, 60, 0],
        scale: [1, 1.2, 0.9, 1],
      }}
      transition={{
        duration,
        repeat: Infinity,
        ease: "easeInOut",
      }}
    />
  );
}

// Live ticker simulation
function LiveTicker() {
  const stocks = [
    { symbol: "RELIANCE", change: "+2.4%", positive: true },
    { symbol: "TCS", change: "+1.8%", positive: true },
    { symbol: "HDFCBANK", change: "-0.5%", positive: false },
    { symbol: "INFY", change: "+3.2%", positive: true },
    { symbol: "WIPRO", change: "+0.9%", positive: true },
  ];

  return (
    <div className="flex gap-6 animate-marquee">
      {[...stocks, ...stocks].map((stock, i) => (
        <div key={i} className="flex items-center gap-2 text-sm whitespace-nowrap">
          <span className="text-spike-cream/60">{stock.symbol}</span>
          <span className={stock.positive ? "text-spike-bull" : "text-spike-bear"}>
            {stock.change}
          </span>
        </div>
      ))}
    </div>
  );
}

// Animated FinScore gauge
function MiniFinScore() {
  const [score, setScore] = useState(0);
  const targetScore = 78;

  useEffect(() => {
    const timer = setTimeout(() => {
      if (score < targetScore) {
        setScore((s) => Math.min(s + 2, targetScore));
      }
    }, 30);
    return () => clearTimeout(timer);
  }, [score]);

  const circumference = 2 * Math.PI * 36;
  const progress = (score / 100) * circumference;

  return (
    <div className="relative w-24 h-24">
      <svg className="w-full h-full -rotate-90">
        <circle
          cx="48"
          cy="48"
          r="36"
          fill="none"
          stroke="rgba(255,255,255,0.05)"
          strokeWidth="6"
        />
        <motion.circle
          cx="48"
          cy="48"
          r="36"
          fill="none"
          stroke="url(#scoreGradient)"
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference - progress}
        />
        <defs>
          <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#638C82" />
            <stop offset="100%" stopColor="#7DCEA0" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-spike-cream">{score}</span>
        <span className="text-[10px] text-spike-muted uppercase tracking-wider">FinScore</span>
      </div>
    </div>
  );
}

export default function SignInPage() {
  return (
    <div className="min-h-screen relative overflow-hidden bg-spike-dark">
      {/* Animated background orbs */}
      <div className="absolute inset-0 overflow-hidden">
        <FloatingOrb size={400} color="#638C82" initialX={-10} initialY={20} duration={25} />
        <FloatingOrb size={300} color="#7DCEA0" initialX={70} initialY={60} duration={30} />
        <FloatingOrb size={250} color="#C9A96E" initialX={50} initialY={-10} duration={20} />
      </div>

      {/* Mesh gradient overlay */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(99,140,130,0.15)_0%,_transparent_50%)]" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_right,_rgba(125,206,160,0.1)_0%,_transparent_50%)]" />

      {/* Grid pattern */}
      <div
        className="absolute inset-0 opacity-[0.02]"
        style={{
          backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                           linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
          backgroundSize: "64px 64px",
        }}
      />

      {/* Content */}
      <div className="relative z-10 min-h-screen flex">
        {/* Left Panel - Branding */}
        <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-12">
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <Link href="/" className="inline-block">
              <span
                className="text-3xl font-semibold text-spike-accent"
                style={{
                  textShadow: "0 0 30px rgba(125, 206, 160, 0.5)",
                }}
              >
                spike
              </span>
            </Link>
          </motion.div>

          {/* Main content */}
          <div className="space-y-8">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <h1 className="text-5xl font-bold text-spike-cream leading-tight">
                Welcome back.
                <br />
                <span className="text-spike-sage">Your portfolio awaits.</span>
              </h1>
              <p className="mt-6 text-lg text-spike-muted max-w-md leading-relaxed">
                Pick up right where you left off. Your AI wealth partner has been
                watching the markets while you were away.
              </p>
            </motion.div>

            {/* Live preview card */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              className="p-6 rounded-2xl bg-white/[0.03] border border-white/[0.06] backdrop-blur-xl max-w-md"
            >
              <div className="flex items-center justify-between mb-4">
                <div>
                  <p className="text-spike-muted text-sm">Today's top pick</p>
                  <p className="text-spike-cream font-semibold text-lg">HDFCBANK</p>
                </div>
                <MiniFinScore />
              </div>
              <div className="h-px bg-white/[0.06] my-4" />
              <div className="overflow-hidden">
                <LiveTicker />
              </div>
            </motion.div>

            {/* Stats */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.6 }}
              className="flex items-center gap-12"
            >
              {[
                { value: "5,000+", label: "Stocks Analyzed" },
                { value: "₹500Cr+", label: "AUM Tracked" },
                { value: "99.9%", label: "Uptime" },
              ].map((stat) => (
                <div key={stat.label}>
                  <div className="text-2xl font-bold text-spike-cream">{stat.value}</div>
                  <div className="text-spike-muted text-sm">{stat.label}</div>
                </div>
              ))}
            </motion.div>
          </div>

          {/* Footer */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="text-spike-muted/60 text-sm"
          >
            SEBI Registered Research Analyst • 100% Data Localized in India
          </motion.div>
        </div>

        {/* Right Panel - Sign In Form */}
        <div className="flex-1 flex items-center justify-center p-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            className="w-full max-w-md"
          >
            {/* Mobile logo */}
            <div className="lg:hidden mb-8 text-center">
              <Link href="/" className="inline-block">
                <span
                  className="text-3xl font-semibold text-spike-accent"
                  style={{
                    textShadow: "0 0 30px rgba(125, 206, 160, 0.5)",
                  }}
                >
                  spike
                </span>
              </Link>
              <p className="mt-2 text-spike-muted">Welcome back</p>
            </div>

            {/* Glassmorphism card wrapper */}
            <div className="p-8 rounded-3xl bg-white/[0.02] border border-white/[0.06] backdrop-blur-2xl shadow-2xl">
              <SignIn
                appearance={{
                  elements: {
                    rootBox: "w-full",
                    card: "bg-transparent shadow-none p-0",
                    headerTitle: "text-spike-cream text-2xl font-semibold",
                    headerSubtitle: "text-spike-muted",
                    socialButtonsBlockButton:
                      "bg-white/[0.03] border border-white/[0.08] text-spike-cream hover:bg-white/[0.06] hover:border-spike-sage/30 transition-all duration-300 rounded-xl",
                    socialButtonsBlockButtonText: "text-spike-cream font-medium",
                    dividerLine: "bg-white/[0.06]",
                    dividerText: "text-spike-muted",
                    formFieldLabel: "text-spike-muted text-sm",
                    formFieldInput:
                      "bg-white/[0.03] border-white/[0.08] text-spike-cream placeholder:text-spike-muted/50 rounded-xl focus:border-spike-sage focus:ring-spike-sage/20",
                    formButtonPrimary:
                      "bg-gradient-to-r from-spike-sage to-spike-accent hover:opacity-90 text-spike-dark font-semibold rounded-xl transition-all duration-300",
                    footerActionLink: "text-spike-accent hover:text-spike-mint",
                    identityPreviewText: "text-spike-cream",
                    identityPreviewEditButton: "text-spike-accent hover:text-spike-mint",
                    formFieldInputShowPasswordButton: "text-spike-muted hover:text-spike-cream",
                    otpCodeFieldInput: "bg-white/[0.03] border-white/[0.08] text-spike-cream",
                    footer: "hidden",
                  },
                }}
              />
            </div>

            {/* Bottom link */}
            <p className="mt-6 text-center text-sm text-spike-muted">
              New to spike?{" "}
              <Link href="/sign-up" className="text-spike-accent hover:text-spike-mint transition-colors">
                Create an account
              </Link>
            </p>
          </motion.div>
        </div>
      </div>

      {/* Marquee animation styles */}
      <style jsx global>{`
        @keyframes marquee {
          0% {
            transform: translateX(0);
          }
          100% {
            transform: translateX(-50%);
          }
        }
        .animate-marquee {
          animation: marquee 20s linear infinite;
        }
      `}</style>
    </div>
  );
}
