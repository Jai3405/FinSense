"use client";

import { useRef } from "react";
import Link from "next/link";
import { motion, useScroll, useTransform } from "framer-motion";
import { ArrowRight, Play } from "lucide-react";
import { Reveal } from "./reveal";

export function Hero() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end start"],
  });

  const y = useTransform(scrollYProgress, [0, 1], [0, 200]);
  const opacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);
  const scale = useTransform(scrollYProgress, [0, 0.8], [1, 0.95]);

  return (
    <section
      ref={containerRef}
      className="relative min-h-screen flex items-center justify-center pt-24 pb-20 px-6"
    >
      <motion.div
        style={{ y, opacity, scale }}
        className="max-w-5xl mx-auto text-center"
      >
        {/* Badge */}
        <Reveal delay={0.1}>
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-subtle mb-8">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-spike-accent opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-spike-sage" />
            </span>
            <span className="text-sm text-spike-muted tracking-wide">
              SEBI Registered &middot; Your wealth, intelligently managed
            </span>
          </div>
        </Reveal>

        {/* Headline */}
        <div className="mb-8">
          <h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-bold leading-[0.95] tracking-tight">
            <Reveal delay={0.2}>
              <span className="text-spike-cream">The wealth</span>
            </Reveal>
            <Reveal delay={0.35}>
              <span className="text-spike-cream"> partner</span>
            </Reveal>
            <Reveal delay={0.5}>
              <span className="text-gradient"> you've always</span>
            </Reveal>
            <Reveal delay={0.65}>
              <span className="text-gradient"> needed.</span>
            </Reveal>
          </h1>
        </div>

        {/* Subhead */}
        <Reveal delay={0.75}>
          <p className="text-lg md:text-xl text-spike-muted max-w-2xl mx-auto mb-12 leading-relaxed">
            SPIKE sits beside you at every financial decision. It researches,
            strategizes, monitors, warns, and optimizes — so your money is never
            left unattended. Not a tool. A partner.
          </p>
        </Reveal>

        {/* CTAs */}
        <Reveal delay={0.9}>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link href="/sign-up">
              <motion.div
                className="group relative"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="absolute -inset-1 rounded-2xl bg-gradient-to-r from-spike-sage via-spike-accent to-spike-mint opacity-50 blur-lg group-hover:opacity-70 transition-opacity" />
                <div className="relative btn-premium text-spike-dark text-lg">
                  <span className="flex items-center gap-2">
                    Start your journey
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform duration-300" />
                  </span>
                </div>
              </motion.div>
            </Link>

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="flex items-center gap-3 px-8 py-4 rounded-2xl glass-subtle text-spike-cream hover:bg-white/5 transition-all duration-300 group"
            >
              <div className="w-10 h-10 rounded-full glass flex items-center justify-center group-hover:glow transition-all">
                <Play className="w-4 h-4 text-spike-accent ml-0.5" />
              </div>
              <span>See it in action</span>
            </motion.button>
          </div>
        </Reveal>

        {/* Value props — not stats, but what SPIKE does */}
        <Reveal delay={1.05}>
          <div className="mt-20 grid grid-cols-2 md:grid-cols-4 gap-6 max-w-3xl mx-auto">
            {[
              { label: "Researches", desc: "Every stock, every day" },
              { label: "Strategizes", desc: "Tailored to you" },
              { label: "Monitors", desc: "24/7, never sleeps" },
              { label: "Protects", desc: "Your downside, always" },
            ].map((item, i) => (
              <div key={i} className="text-center">
                <div className="text-lg font-semibold text-gradient mb-1">
                  {item.label}
                </div>
                <div className="text-xs text-spike-muted">{item.desc}</div>
              </div>
            ))}
          </div>
        </Reveal>

        {/* Scroll indicator */}
        <motion.div
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        >
          <div className="w-6 h-10 rounded-full border-2 border-spike-sage/30 flex justify-center pt-2">
            <motion.div
              className="w-1 h-2 rounded-full bg-spike-sage"
              animate={{ opacity: [1, 0.3, 1], y: [0, 8, 0] }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            />
          </div>
        </motion.div>
      </motion.div>
    </section>
  );
}
