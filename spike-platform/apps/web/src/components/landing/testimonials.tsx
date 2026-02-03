"use client";

import { motion } from "framer-motion";
import { Reveal } from "./reveal";
import { Star } from "lucide-react";

const testimonials = [
  {
    quote:
      "SPIKE doesn't just give me data — it sits beside me at every decision. The portfolio monitoring caught a concentration risk I'd been blind to for months. It genuinely feels like having a wealth partner.",
    name: "Priya Sharma",
    role: "DIY Investor",
    location: "Mumbai",
  },
  {
    quote:
      "The depth is unreal. Research, risk alerts, regime analysis, rebalancing suggestions — all personalized. I used to juggle 6 tools. Now I open SPIKE and everything's there.",
    name: "Arjun Mehta",
    role: "Portfolio Manager",
    location: "Bangalore",
  },
  {
    quote:
      "I told SPIKE what I wanted in plain English and it built a backtested strategy around my risk profile. Then it kept monitoring, alerting, optimizing. It never stops working for you.",
    name: "Kavya Reddy",
    role: "Algo Trader",
    location: "Hyderabad",
  },
];

export function Testimonials() {
  return (
    <section className="relative py-32 px-6">
      {/* Background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-0 left-1/4 w-[400px] h-[400px] rounded-full bg-spike-sage/3 blur-[100px]" />
        <div className="absolute bottom-0 right-1/4 w-[300px] h-[300px] rounded-full bg-spike-accent/3 blur-[100px]" />
      </div>

      <div className="max-w-7xl mx-auto relative">
        {/* Header */}
        <div className="max-w-3xl mx-auto text-center mb-16">
          <Reveal>
            <p className="text-sm uppercase tracking-[0.2em] text-spike-sage mb-4">
              Trusted by investors
            </p>
          </Reveal>
          <Reveal delay={0.1}>
            <h2 className="text-4xl md:text-5xl font-bold text-spike-cream leading-tight">
              Built for people who
              <span className="text-gradient"> take wealth seriously</span>
            </h2>
          </Reveal>
        </div>

        {/* Testimonial cards */}
        <div className="grid md:grid-cols-3 gap-6">
          {testimonials.map((testimonial, i) => (
            <Reveal key={i} delay={i * 0.12}>
              <motion.div
                className="glass-subtle rounded-3xl p-8 h-full flex flex-col justify-between group"
                whileHover={{ y: -4 }}
                transition={{ duration: 0.3 }}
              >
                {/* Stars */}
                <div>
                  <div className="flex items-center gap-1 mb-6">
                    {[...Array(5)].map((_, j) => (
                      <Star
                        key={j}
                        className="w-4 h-4 fill-spike-gold text-spike-gold"
                      />
                    ))}
                  </div>

                  {/* Quote */}
                  <p className="text-spike-cream/90 leading-relaxed text-[15px]">
                    "{testimonial.quote}"
                  </p>
                </div>

                {/* Author */}
                <div className="flex items-center gap-3 mt-8 pt-6 border-t border-white/5">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-spike-sage to-spike-accent flex items-center justify-center">
                    <span className="text-sm font-semibold text-spike-dark">
                      {testimonial.name[0]}
                    </span>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-spike-cream">
                      {testimonial.name}
                    </p>
                    <p className="text-xs text-spike-muted">
                      {testimonial.role} &middot; {testimonial.location}
                    </p>
                  </div>
                </div>
              </motion.div>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}
