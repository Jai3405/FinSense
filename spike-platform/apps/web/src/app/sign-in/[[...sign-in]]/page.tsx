"use client";

import { SignIn } from "@clerk/nextjs";
import Link from "next/link";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { Activity, TrendingUp, TrendingDown } from "lucide-react";

// FinScore Ring
function FinScoreRing({ score }: { score: number }) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => value < score && setValue((v) => Math.min(v + 2, score)), 25);
    return () => clearTimeout(t);
  }, [value, score]);

  const r = 22, circ = 2 * Math.PI * r, prog = (value / 100) * circ;

  return (
    <div className="relative w-14 h-14">
      <svg className="w-full h-full -rotate-90" viewBox="0 0 52 52">
        <circle cx="26" cy="26" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="3" />
        <circle
          cx="26" cy="26" r={r} fill="none" stroke="#22C55E" strokeWidth="3" strokeLinecap="round"
          strokeDasharray={circ} strokeDashoffset={circ - prog}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-base font-bold text-white">{value}</span>
      </div>
    </div>
  );
}

export default function SignInPage() {
  const indices = [
    { name: "NIFTY", value: "22,456", change: "+0.82%", up: true },
    { name: "SENSEX", value: "73,890", change: "+0.65%", up: true },
    { name: "BANKNIFTY", value: "48,234", change: "-0.23%", up: false },
  ];

  const news = [
    { symbol: "TATAMOTORS", text: "Q3 profit up 47%", up: true },
    { symbol: "RELIANCE", text: "Jio adds 8.5M users", up: true },
    { symbol: "ZOMATO", text: "Regulatory scrutiny", up: false },
  ];


  return (
    <>
      <style jsx global>{`
        body {
          background: #050a08 !important;
        }

        /* Animated flowing gradients - like market waves */
        @keyframes flow1 {
          0%, 100% { transform: translate(0%, 0%) scale(1); }
          25% { transform: translate(5%, 10%) scale(1.1); }
          50% { transform: translate(-5%, 5%) scale(0.95); }
          75% { transform: translate(10%, -5%) scale(1.05); }
        }

        @keyframes flow2 {
          0%, 100% { transform: translate(0%, 0%) scale(1); }
          25% { transform: translate(-10%, -5%) scale(1.05); }
          50% { transform: translate(5%, -10%) scale(1.1); }
          75% { transform: translate(-5%, 5%) scale(0.95); }
        }

        @keyframes flow3 {
          0%, 100% { transform: translate(0%, 0%) scale(1.05); }
          33% { transform: translate(8%, 8%) scale(0.95); }
          66% { transform: translate(-8%, 3%) scale(1.1); }
        }

        @keyframes pulse {
          0%, 100% { opacity: 0.5; }
          50% { opacity: 0.7; }
        }

        .gradient-orb-1 {
          animation: flow1 20s ease-in-out infinite, pulse 8s ease-in-out infinite;
        }

        .gradient-orb-2 {
          animation: flow2 25s ease-in-out infinite, pulse 10s ease-in-out infinite;
        }

        .gradient-orb-3 {
          animation: flow3 18s ease-in-out infinite, pulse 6s ease-in-out infinite;
        }

        /* Animated grid lines - like trading charts */
        @keyframes gridPulse {
          0%, 100% { opacity: 0.03; }
          50% { opacity: 0.06; }
        }

        .chart-grid {
          background-image:
            linear-gradient(rgba(125,206,160,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(125,206,160,0.1) 1px, transparent 1px);
          background-size: 60px 60px;
          animation: gridPulse 4s ease-in-out infinite;
        }
      `}</style>

      <div className="min-h-screen relative overflow-hidden" style={{ background: "#050a08" }}>
        {/* Dynamic Gradient Orbs - SPIKE colors */}
        <div className="fixed inset-0 overflow-hidden">
          {/* Primary sage orb - large, top-left */}
          <div
            className="gradient-orb-1 absolute -top-[10%] -left-[5%] w-[60vw] h-[60vw] rounded-full"
            style={{
              background: "radial-gradient(circle at 30% 30%, rgba(99,140,130,0.9) 0%, rgba(99,140,130,0.5) 30%, rgba(99,140,130,0.2) 50%, transparent 70%)",
              filter: "blur(40px)",
            }}
          />

          {/* Gold accent orb - top-center */}
          <div
            className="gradient-orb-2 absolute -top-[5%] left-[30%] w-[45vw] h-[45vw] rounded-full"
            style={{
              background: "radial-gradient(circle at 50% 50%, rgba(201,169,110,0.8) 0%, rgba(201,169,110,0.4) 30%, rgba(201,169,110,0.15) 50%, transparent 70%)",
              filter: "blur(40px)",
            }}
          />

          {/* Mint orb - right side */}
          <div
            className="gradient-orb-3 absolute top-[30%] right-[-10%] w-[40vw] h-[40vw] rounded-full"
            style={{
              background: "radial-gradient(circle at 50% 50%, rgba(125,206,160,0.7) 0%, rgba(125,206,160,0.35) 30%, rgba(125,206,160,0.1) 50%, transparent 70%)",
              filter: "blur(40px)",
            }}
          />
        </div>

        {/* Subtle chart grid overlay */}
        <div className="fixed inset-0 chart-grid pointer-events-none" />

        {/* Vignette */}
        <div className="fixed inset-0 pointer-events-none" style={{
          background: "radial-gradient(ellipse at center, transparent 0%, #050a08 80%)"
        }} />

        {/* Content */}
        <div className="relative z-10 min-h-screen flex">
          {/* Left Panel */}
          <div className="hidden lg:flex lg:w-1/2 p-12 flex-col">
            {/* Logo + Live indicator */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center justify-between"
            >
              <Link href="/">
                <span className="text-2xl font-bold text-spike-accent" style={{ textShadow: "0 0 20px rgba(125,206,160,0.4)" }}>
                  spike
                </span>
              </Link>
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-emerald-400 animate-pulse" />
                <span className="text-white/40 text-sm">Markets Live</span>
              </div>
            </motion.div>

            {/* Main content */}
            <div className="flex-1 flex flex-col justify-center">
              {/* Headline */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.8 }}
              >
                <h1 className="text-7xl lg:text-8xl font-black leading-[0.9] tracking-tight">
                  <motion.span
                    className="block text-white"
                    initial={{ x: -50, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ duration: 0.6, delay: 0.2 }}
                  >
                    Welcome
                  </motion.span>
                  <motion.span
                    className="block bg-gradient-to-r from-spike-sage via-spike-accent to-spike-mint bg-clip-text text-transparent"
                    initial={{ x: -50, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    transition={{ duration: 0.6, delay: 0.35 }}
                  >
                    back.
                  </motion.span>
                </h1>

                <motion.div
                  className="h-1 w-32 mt-6 rounded-full overflow-hidden bg-white/10"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.6 }}
                >
                  <motion.div
                    className="h-full bg-gradient-to-r from-spike-sage to-spike-accent"
                    initial={{ x: "-100%" }}
                    animate={{ x: "0%" }}
                    transition={{ duration: 0.8, delay: 0.7 }}
                  />
                </motion.div>

                <motion.p
                  className="mt-6 text-white/40 text-lg max-w-md"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.9 }}
                >
                  Markets are moving. Your portfolio awaits.
                </motion.p>
              </motion.div>

              {/* Live Indices */}
              <motion.div
                className="mt-16"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
              >
                <div className="text-white/30 text-xs uppercase tracking-widest mb-5">Live Indices</div>
                <div className="flex gap-10">
                  {indices.map((idx, i) => (
                    <motion.div
                      key={idx.name}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.6 + i * 0.1 }}
                    >
                      <div className="text-white/40 text-xs mb-1">{idx.name}</div>
                      <div className="text-white font-semibold text-lg">{idx.value}</div>
                      <div className={`text-sm flex items-center gap-1 mt-1 ${idx.up ? "text-emerald-400" : "text-red-400"}`}>
                        {idx.up ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                        {idx.change}
                      </div>
                    </motion.div>
                  ))}
                </div>
              </motion.div>

              {/* Today's Pick & News - Side by side */}
              <motion.div
                className="mt-14 flex gap-16"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
              >
                {/* Today's Pick */}
                <div>
                  <div className="text-amber-400/80 text-xs uppercase tracking-widest mb-5">Today&apos;s Pick</div>
                  <div className="flex items-center gap-5">
                    <FinScoreRing score={82} />
                    <div>
                      <div className="text-white font-semibold text-lg">HDFCBANK</div>
                      <div className="text-white/30 text-sm">Banking</div>
                      <div className="text-emerald-400 text-sm flex items-center gap-1 mt-1">
                        <TrendingUp className="w-3 h-3" /> +2.4%
                      </div>
                    </div>
                  </div>
                </div>

                {/* In the News */}
                <div>
                  <div className="text-white/30 text-xs uppercase tracking-widest mb-5">In the News</div>
                  <div className="space-y-3">
                    {news.map((item, i) => (
                      <motion.div
                        key={item.symbol}
                        className="flex items-center gap-3"
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.8 + i * 0.1 }}
                      >
                        <span className={`w-0.5 h-5 rounded-full ${item.up ? "bg-emerald-500" : "bg-red-500"}`} />
                        <span className="text-white text-sm">{item.symbol}</span>
                        <span className="text-white/30 text-sm">{item.text}</span>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Footer */}
            <motion.div
              className="text-white/20 text-xs"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.2 }}
            >
              SEBI Registered Research Analyst
            </motion.div>
          </div>

          {/* Right Panel - Form */}
          <div className="flex-1 flex items-center justify-center p-8 lg:p-12">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="w-full max-w-md"
            >
              {/* Mobile header */}
              <div className="lg:hidden mb-8 text-center">
                <span className="text-2xl font-bold text-spike-accent">spike</span>
                <p className="mt-2 text-white/40">Welcome back</p>
              </div>

              {/* Form card */}
              <div className="p-8 rounded-3xl bg-black/50 border border-white/[0.08] backdrop-blur-xl">
                <SignIn
                  appearance={{
                    variables: {
                      colorBackground: "transparent",
                      colorInputBackground: "transparent",
                      colorNeutral: "white",
                      colorText: "white",
                      colorTextSecondary: "rgba(255,255,255,0.5)",
                      colorPrimary: "#7DCEA0",
                    },
                    elements: {
                      rootBox: "w-full",
                      card: "!bg-transparent !shadow-none !border-none p-0 gap-5",
                      cardBox: "!border-none !shadow-none !bg-transparent",
                      headerTitle: "text-white text-xl font-semibold",
                      headerSubtitle: "text-white/50 text-sm",
                      socialButtonsBlockButton:
                        "bg-white/[0.04] border border-white/[0.08] text-white hover:bg-white/[0.08] transition-all duration-200 rounded-xl h-11",
                      socialButtonsBlockButtonText: "text-white/70 font-medium text-sm",
                      dividerLine: "bg-white/[0.08]",
                      dividerText: "text-white/30 text-xs",
                      formFieldLabel: "text-white/50 text-sm font-medium",
                      formFieldInput:
                        "!bg-white/[0.04] border border-white/[0.08] text-white placeholder:text-white/30 rounded-xl h-11 focus:border-spike-accent/50 focus:ring-0 transition-all",
                      formButtonPrimary:
                        "bg-gradient-to-r from-spike-sage to-spike-accent text-spike-dark font-semibold rounded-xl h-11 shadow-lg shadow-spike-sage/20 hover:shadow-spike-sage/30 transition-all",
                      footer: "hidden",
                      footerAction: "hidden",
                      footerActionLink: "hidden",
                      identityPreviewText: "text-white",
                      identityPreviewEditButton: "text-spike-accent",
                      formFieldInputShowPasswordButton: "text-white/40 hover:text-white",
                      otpCodeFieldInput: "!bg-white/[0.04] border border-white/[0.08] text-white rounded-xl",
                      formFieldErrorText: "text-red-400 text-xs",
                      alert: "bg-red-500/10 border border-red-500/20 text-red-400 rounded-xl",
                    },
                    layout: {
                      socialButtonsPlacement: "top",
                      showOptionalFields: false,
                    },
                  }}
                />
              </div>

              <motion.p
                className="mt-6 text-center text-white/40 text-sm"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6 }}
              >
                New to Spike?{" "}
                <Link href="/sign-up" className="text-spike-accent hover:text-spike-mint transition-colors">
                  Create account
                </Link>
              </motion.p>
            </motion.div>
          </div>
        </div>
      </div>
    </>
  );
}
