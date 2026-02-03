"use client";

import { motion, useInView } from "framer-motion";
import { useRef } from "react";
import {
  UserPlus,
  Search,
  Brain,
  Shield,
  TrendingUp,
  RefreshCcw,
} from "lucide-react";
import { Reveal } from "./reveal";

const steps = [
  {
    number: "01",
    icon: UserPlus,
    title: "You tell us who you are",
    description:
      "2-minute risk profile. Your goals, your horizon, your comfort with volatility. SPIKE shapes everything around you from here.",
  },
  {
    number: "02",
    icon: Search,
    title: "SPIKE researches everything",
    description:
      "5,000+ stocks analyzed across fundamentals, technicals, sentiment, flows, and regime fit. Deep data — not surface-level metrics.",
  },
  {
    number: "03",
    icon: Brain,
    title: "You get personalized intelligence",
    description:
      "Strategies tailored to you. Legend Agent opinions on your picks. Smart Themes matched to your profile. Commentary that makes sense of the noise.",
  },
  {
    number: "04",
    icon: Shield,
    title: "Your risks are watched",
    description:
      "Concentration alerts, drawdown monitoring, VaR calculations, regime misalignment warnings. Your downside is never unattended.",
  },
  {
    number: "05",
    icon: TrendingUp,
    title: "Your portfolio grows",
    description:
      "Auto-rebalancing, tax-loss harvesting suggestions, and regime-adaptive allocation. SPIKE optimizes while you live your life.",
  },
  {
    number: "06",
    icon: RefreshCcw,
    title: "You get smarter every day",
    description:
      "Every recommendation comes with a 'why'. Every alert teaches you something. SPIKE is a partner that makes you a better investor over time.",
  },
];

export function HowItWorks() {
  const containerRef = useRef<HTMLDivElement>(null);
  const isInView = useInView(containerRef, { once: true, margin: "-100px" });

  return (
    <section id="how-it-works" className="relative py-32 px-6">
      <div className="max-w-6xl mx-auto" ref={containerRef}>
        {/* Header */}
        <div className="max-w-3xl mx-auto text-center mb-20">
          <Reveal>
            <p className="text-sm uppercase tracking-[0.2em] text-spike-sage mb-4">
              Your journey
            </p>
          </Reveal>
          <Reveal delay={0.1}>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold text-spike-cream leading-tight">
              End to end.
              <span className="text-gradient"> Every step.</span>
            </h2>
          </Reveal>
          <Reveal delay={0.2}>
            <p className="mt-6 text-lg text-spike-muted leading-relaxed">
              From your very first question to your hundredth rebalance — SPIKE
              is the partner that never leaves your side.
            </p>
          </Reveal>
        </div>

        {/* Steps — alternating layout */}
        <div className="relative">
          {/* Connecting line */}
          <div className="absolute top-0 bottom-0 left-[27px] md:left-1/2 w-px">
            <motion.div
              className="w-full bg-gradient-to-b from-spike-sage/40 via-spike-sage/20 to-transparent"
              initial={{ height: "0%" }}
              animate={isInView ? { height: "100%" } : { height: "0%" }}
              transition={{ duration: 2.5, ease: [0.16, 1, 0.3, 1] }}
            />
          </div>

          <div className="space-y-12 md:space-y-20">
            {steps.map((step, i) => (
              <Reveal key={step.number} delay={i * 0.1}>
                <div
                  className={`relative flex items-start gap-8 md:gap-16 ${
                    i % 2 === 0 ? "md:flex-row" : "md:flex-row-reverse"
                  }`}
                >
                  {/* Dot on line */}
                  <div className="absolute left-[20px] md:left-1/2 md:-translate-x-1/2 z-10">
                    <motion.div
                      className="w-[16px] h-[16px] rounded-full border-2 border-spike-sage bg-spike-dark"
                      initial={{ scale: 0 }}
                      animate={isInView ? { scale: 1 } : { scale: 0 }}
                      transition={{ delay: i * 0.2 + 0.5, duration: 0.4 }}
                    />
                  </div>

                  {/* Content */}
                  <div
                    className={`flex-1 pl-16 md:pl-0 ${
                      i % 2 === 0 ? "md:pr-20 md:text-right" : "md:pl-20"
                    }`}
                  >
                    <div
                      className={`inline-block ${
                        i % 2 === 0 ? "md:ml-auto" : ""
                      }`}
                    >
                      <div
                        className={`glass-subtle rounded-3xl p-7 max-w-md group hover:glow transition-all duration-500 ${
                          i % 2 === 0 ? "md:ml-auto" : ""
                        }`}
                      >
                        <div
                          className={`flex items-center gap-4 mb-4 ${
                            i % 2 === 0 ? "md:flex-row-reverse" : ""
                          }`}
                        >
                          <div className="w-11 h-11 rounded-2xl bg-spike-sage/10 flex items-center justify-center group-hover:bg-spike-sage/20 transition-colors">
                            <step.icon className="w-5 h-5 text-spike-sage" />
                          </div>
                          <span className="text-2xl font-bold text-spike-sage/20">
                            {step.number}
                          </span>
                        </div>
                        <h3 className="text-lg font-semibold text-spike-cream mb-2">
                          {step.title}
                        </h3>
                        <p className="text-spike-muted text-sm leading-relaxed">
                          {step.description}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Spacer */}
                  <div className="hidden md:block flex-1" />
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
