"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import { Reveal } from "./reveal";

export function CTA() {
  return (
    <section id="pricing" className="relative py-32 px-6">
      <div className="max-w-4xl mx-auto relative">
        <Reveal>
          <div className="relative rounded-[2.5rem] overflow-hidden">
            {/* Background */}
            <div className="absolute inset-0 bg-gradient-to-br from-spike-sage/20 via-spike-deep to-spike-dark" />
            <div className="absolute inset-0 dot-grid" />

            {/* Glow orbs */}
            <div className="absolute -top-20 -right-20 w-60 h-60 rounded-full bg-spike-sage/10 blur-[80px]" />
            <div className="absolute -bottom-20 -left-20 w-60 h-60 rounded-full bg-spike-accent/10 blur-[80px]" />

            {/* Border */}
            <div className="absolute inset-0 rounded-[2.5rem] gradient-border" />

            {/* Content */}
            <div className="relative px-8 py-20 md:px-16 md:py-24 text-center">
              <motion.div
                className="w-16 h-16 mx-auto mb-8 rounded-2xl bg-gradient-to-br from-spike-sage to-spike-accent p-[1px]"
                animate={{
                  boxShadow: [
                    "0 0 20px rgba(99, 140, 130, 0.3)",
                    "0 0 40px rgba(99, 140, 130, 0.5)",
                    "0 0 20px rgba(99, 140, 130, 0.3)",
                  ],
                }}
                transition={{ duration: 3, repeat: Infinity }}
              >
                <div className="w-full h-full rounded-2xl bg-spike-dark flex items-center justify-center">
                  <span className="text-2xl font-bold text-gradient">S</span>
                </div>
              </motion.div>

              <h2 className="text-4xl md:text-5xl font-bold text-spike-cream mb-6 leading-tight">
                Your wealth deserves
                <br />
                <span className="text-gradient">a real partner.</span>
              </h2>

              <p className="text-lg text-spike-muted max-w-xl mx-auto mb-10 leading-relaxed">
                Research, strategies, risk monitoring, portfolio intelligence,
                live commentary, and continuous learning — all in one place,
                all personalized to you. Stop juggling tools. Start growing.
              </p>

              {/* Pricing preview */}
              <div className="flex items-center justify-center gap-6 mb-10 flex-wrap">
                <div className="glass-subtle rounded-2xl px-6 py-4 text-center">
                  <div className="text-sm text-spike-muted mb-1">Free</div>
                  <div className="text-2xl font-bold text-spike-cream">₹0</div>
                  <div className="text-xs text-spike-muted mt-1">forever</div>
                </div>
                <div className="relative glass rounded-2xl px-6 py-4 text-center glow">
                  <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-0.5 rounded-full bg-gradient-to-r from-spike-sage to-spike-accent text-xs font-medium text-spike-dark">
                    Popular
                  </div>
                  <div className="text-sm text-spike-sage mb-1">Pro</div>
                  <div className="text-2xl font-bold text-spike-cream">₹299</div>
                  <div className="text-xs text-spike-muted mt-1">/month</div>
                </div>
                <div className="glass-subtle rounded-2xl px-6 py-4 text-center">
                  <div className="text-sm text-spike-muted mb-1">Premium</div>
                  <div className="text-2xl font-bold text-spike-cream">₹999</div>
                  <div className="text-xs text-spike-muted mt-1">/month</div>
                </div>
              </div>

              {/* CTA Button */}
              <Link href="/sign-up">
                <motion.div
                  className="inline-block group"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <div className="btn-premium text-spike-dark text-lg">
                    <span className="flex items-center gap-2">
                      Start free, upgrade anytime
                      <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform duration-300" />
                    </span>
                  </div>
                </motion.div>
              </Link>

              <p className="text-spike-muted/50 text-sm mt-6">
                No credit card required &middot; Cancel anytime &middot; SEBI compliant
              </p>
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
