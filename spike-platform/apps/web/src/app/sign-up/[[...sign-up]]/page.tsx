"use client";

import { SignUp } from "@clerk/nextjs";
import Link from "next/link";
import { motion } from "framer-motion";
import { Sparkles, Shield, Brain, BarChart3, Bell, Zap } from "lucide-react";

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
        x: [0, 80, -60, 0],
        y: [0, -60, 80, 0],
        scale: [1, 1.1, 0.95, 1],
      }}
      transition={{
        duration,
        repeat: Infinity,
        ease: "easeInOut",
      }}
    />
  );
}

// Feature item component
function FeatureItem({
  icon: Icon,
  text,
  delay,
}: {
  icon: React.ComponentType<{ className?: string }>;
  text: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.5, delay }}
      className="flex items-center gap-4 group"
    >
      <div className="w-10 h-10 rounded-xl bg-spike-sage/15 border border-spike-sage/25 flex items-center justify-center group-hover:bg-spike-sage/25 group-hover:border-spike-sage/40 transition-all duration-300">
        <Icon className="w-5 h-5 text-spike-sage" />
      </div>
      <span className="text-white/90">{text}</span>
    </motion.div>
  );
}

// Animated benefit cards
function BenefitCard({
  title,
  value,
  delay,
}: {
  title: string;
  value: string;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay }}
      className="p-4 rounded-2xl bg-black/40 border border-white/10 backdrop-blur-xl"
    >
      <div className="text-2xl font-bold text-spike-accent">{value}</div>
      <div className="text-spike-cream/50 text-sm mt-1">{title}</div>
    </motion.div>
  );
}

export default function SignUpPage() {
  const features = [
    { icon: BarChart3, text: "Deep research on 5,000+ Indian stocks" },
    { icon: Brain, text: "AI-powered personalized strategies" },
    { icon: Shield, text: "Real-time risk monitoring & alerts" },
    { icon: Sparkles, text: "Legend Agents (Buffett, Lynch, Dalio)" },
    { icon: Bell, text: "Proactive alerts before the market moves" },
    { icon: Zap, text: "One FinScore per stock — your single source of truth" },
  ];

  return (
    <div className="min-h-screen relative overflow-hidden bg-spike-dark">
      {/* Animated background orbs */}
      <div className="absolute inset-0 overflow-hidden">
        <FloatingOrb size={450} color="#638C82" initialX={60} initialY={10} duration={28} />
        <FloatingOrb size={350} color="#7DCEA0" initialX={-5} initialY={50} duration={32} />
        <FloatingOrb size={280} color="#D3E9D7" initialX={40} initialY={70} duration={24} />
      </div>

      {/* Mesh gradient overlay */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_left,_rgba(99,140,130,0.15)_0%,_transparent_50%)]" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_rgba(125,206,160,0.1)_0%,_transparent_50%)]" />

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
          <div className="space-y-10">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
            >
              <h1 className="text-5xl font-bold text-white leading-tight">
                Your wealth journey
                <br />
                <span className="text-spike-sage">starts here.</span>
              </h1>
              <p className="mt-6 text-lg text-spike-cream/70 max-w-md leading-relaxed">
                Join thousands of investors using AI-powered intelligence to make
                smarter decisions. Free forever for basic features.
              </p>
            </motion.div>

            {/* Features list */}
            <div className="space-y-4">
              {features.map((feature, i) => (
                <FeatureItem
                  key={feature.text}
                  icon={feature.icon}
                  text={feature.text}
                  delay={0.4 + i * 0.1}
                />
              ))}
            </div>

            {/* Benefit cards */}
            <div className="grid grid-cols-3 gap-4 max-w-lg">
              <BenefitCard title="Active Users" value="50K+" delay={1} />
              <BenefitCard title="Avg Return" value="+24%" delay={1.1} />
              <BenefitCard title="Uptime" value="99.9%" delay={1.2} />
            </div>
          </div>

          {/* Footer */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 1.3 }}
            className="text-spike-cream/40 text-sm"
          >
            SEBI Registered Research Analyst • 100% Data Localized in India
          </motion.div>
        </div>

        {/* Right Panel - Sign Up Form */}
        <div className="flex-1 flex items-center justify-center p-8">
          {/* Dark backdrop for form readability */}
          <div className="absolute inset-0 lg:left-1/2 bg-black/30 backdrop-blur-sm" />

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
            className="relative z-10 w-full max-w-md"
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
              <p className="mt-2 text-spike-cream/60">Start your journey</p>
            </div>

            {/* Glassmorphism card wrapper */}
            <div className="p-8 rounded-3xl bg-black/60 border border-white/10 backdrop-blur-2xl shadow-2xl">
              <SignUp
                appearance={{
                  elements: {
                    rootBox: "w-full",
                    card: "bg-transparent shadow-none p-0",
                    headerTitle: "text-white text-2xl font-semibold",
                    headerSubtitle: "text-spike-cream/60",
                    socialButtonsBlockButton:
                      "bg-white/10 border border-white/15 text-white hover:bg-white/15 hover:border-spike-sage/40 transition-all duration-300 rounded-xl",
                    socialButtonsBlockButtonText: "text-white font-medium",
                    dividerLine: "bg-white/15",
                    dividerText: "text-spike-cream/50",
                    formFieldLabel: "text-spike-cream/80 text-sm font-medium",
                    formFieldInput:
                      "bg-white/10 border-white/15 text-white placeholder:text-spike-cream/40 rounded-xl focus:border-spike-sage focus:ring-spike-sage/30 focus:bg-white/15",
                    formButtonPrimary:
                      "bg-gradient-to-r from-spike-sage to-spike-accent hover:opacity-90 text-spike-dark font-semibold rounded-xl transition-all duration-300 shadow-lg shadow-spike-sage/25",
                    footerActionLink: "text-spike-accent hover:text-spike-mint",
                    identityPreviewText: "text-white",
                    identityPreviewEditButton: "text-spike-accent hover:text-spike-mint",
                    formFieldInputShowPasswordButton: "text-spike-cream/60 hover:text-white",
                    otpCodeFieldInput: "bg-white/10 border-white/15 text-white",
                    formFieldErrorText: "text-red-400",
                    alert: "bg-red-500/20 border-red-500/30 text-red-300",
                  },
                }}
              />
            </div>

            {/* Terms */}
            <p className="mt-6 text-center text-xs text-spike-cream/50 leading-relaxed">
              By signing up, you agree to our{" "}
              <Link href="/terms" className="text-spike-accent hover:text-spike-mint transition-colors">
                Terms
              </Link>{" "}
              and{" "}
              <Link href="/privacy" className="text-spike-accent hover:text-spike-mint transition-colors">
                Privacy Policy
              </Link>
              .
              <br />
              Investment in securities market are subject to market risks.
            </p>

            {/* Bottom link */}
            <p className="mt-4 text-center text-sm text-spike-cream/60">
              Already have an account?{" "}
              <Link href="/sign-in" className="text-spike-accent hover:text-spike-mint transition-colors font-medium">
                Sign in
              </Link>
            </p>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
