"use client";

import { useEffect, useState } from "react";
import { motion, useScroll, useTransform, useSpring } from "framer-motion";
import { Navbar } from "./navbar";
import { Hero } from "./hero";
import { LogoCloud } from "./logo-cloud";
import { Features } from "./features";
import { FinScoreSection } from "./finscore-section";
import { HowItWorks } from "./how-it-works";
import { Testimonials } from "./testimonials";
import { CTA } from "./cta";
import { Footer } from "./footer";
import { ThreeScene } from "@/components/three/scene";

export function LandingPage() {
  const { scrollYProgress } = useScroll();
  const smoothProgress = useSpring(scrollYProgress, {
    stiffness: 100,
    damping: 30,
    restDelta: 0.001,
  });

  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <div className="relative min-h-screen bg-spike-dark overflow-x-hidden">
      {/* 3D Background */}
      <ThreeScene />

      {/* Mesh gradient overlay */}
      <div className="fixed inset-0 -z-[5] bg-mesh-gradient opacity-60" />

      {/* Grain texture */}
      <div
        className="fixed inset-0 -z-[4] pointer-events-none opacity-[0.015]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        }}
      />

      {/* Scroll progress indicator */}
      <motion.div
        className="fixed top-0 left-0 right-0 h-[2px] z-[100] origin-left"
        style={{
          scaleX: smoothProgress,
          background: "linear-gradient(90deg, #638C82, #7DCEA0, #D3E9D7)",
        }}
      />

      {/* Content */}
      <Navbar />
      <Hero />
      <LogoCloud />
      <Features />
      <FinScoreSection />
      <HowItWorks />
      <Testimonials />
      <CTA />
      <Footer />
    </div>
  );
}
