"use client";

import { motion } from "framer-motion";
import { Reveal } from "./reveal";

const partners = [
  "NSE", "BSE", "SEBI", "CDSL", "NSDL", "RBI",
];

export function LogoCloud() {
  return (
    <section className="relative py-16 px-6">
      <div className="max-w-5xl mx-auto">
        <Reveal>
          <div className="text-center mb-10">
            <p className="text-sm uppercase tracking-[0.2em] text-spike-muted/60">
              Built for the Indian market ecosystem
            </p>
          </div>
        </Reveal>

        <div className="flex items-center justify-center gap-8 md:gap-16 flex-wrap">
          {partners.map((name, i) => (
            <Reveal key={name} delay={i * 0.08}>
              <div className="px-6 py-3 rounded-xl glass-subtle">
                <span className="text-sm font-medium text-spike-muted/50 tracking-wider">
                  {name}
                </span>
              </div>
            </Reveal>
          ))}
        </div>

        {/* Divider */}
        <div className="mt-20 flex items-center gap-4">
          <div className="flex-1 h-px bg-gradient-to-r from-transparent to-spike-sage/20" />
          <div className="w-1.5 h-1.5 rounded-full bg-spike-sage/40" />
          <div className="flex-1 h-px bg-gradient-to-l from-transparent to-spike-sage/20" />
        </div>
      </div>
    </section>
  );
}
