"use client";

import { SignIn } from "@clerk/nextjs";
import Link from "next/link";
import { motion } from "framer-motion";
import { MarketTicker } from "@/components/ui/market-ticker";
export default function SignInPage() {
  return (
    <>
      <style jsx global>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

        body {
          background: #0a1210 !important;
        }

        .heading-serif {
          font-family: 'Playfair Display', serif;
        }

        @keyframes aurora {
          0%, 100% {
            opacity: 0.4;
            transform: translateY(0) rotate(0deg) scale(1);
          }
          25% {
            opacity: 0.6;
            transform: translateY(-20px) rotate(2deg) scale(1.05);
          }
          50% {
            opacity: 0.5;
            transform: translateY(-10px) rotate(-1deg) scale(1.02);
          }
          75% {
            opacity: 0.7;
            transform: translateY(-30px) rotate(1deg) scale(1.08);
          }
        }

        .aurora-blob {
          animation: aurora 15s ease-in-out infinite;
        }

        .aurora-blob-delayed {
          animation: aurora 18s ease-in-out infinite;
          animation-delay: -5s;
        }

        .aurora-blob-slow {
          animation: aurora 22s ease-in-out infinite;
          animation-delay: -10s;
        }
      `}</style>

      <div className="min-h-screen relative" style={{ background: "#0a1210" }}>
        {/* AURORA BACKGROUND - Fixed, behind everything */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none" style={{ zIndex: 0 }}>
          {/* Base gradient */}
          <div
            className="absolute inset-0"
            style={{ background: "linear-gradient(to bottom right, #0a1210, #0a1210, rgba(26, 48, 40, 0.2))" }}
          />

          {/* Aurora blob 1 */}
          <div
            className="aurora-blob absolute rounded-full"
            style={{
              top: "25%",
              right: "25%",
              width: 600,
              height: 600,
              background: "linear-gradient(to bottom right, rgba(45, 90, 71, 0.3), rgba(31, 64, 53, 0.2), transparent)",
              filter: "blur(60px)",
            }}
          />

          {/* Aurora blob 2 */}
          <div
            className="aurora-blob-delayed absolute rounded-full"
            style={{
              top: "33%",
              right: "33%",
              width: 800,
              height: 400,
              background: "linear-gradient(to right, rgba(74, 222, 128, 0.15), rgba(45, 90, 71, 0.2), transparent)",
              filter: "blur(60px)",
              transform: "rotate(45deg)",
            }}
          />

          {/* Aurora blob 3 */}
          <div
            className="aurora-blob-slow absolute rounded-full"
            style={{
              bottom: "25%",
              left: "33%",
              width: 500,
              height: 500,
              background: "linear-gradient(to top left, rgba(34, 84, 61, 0.25), rgba(26, 48, 40, 0.15), transparent)",
              filter: "blur(60px)",
            }}
          />

          {/* Noise texture */}
          <div
            className="absolute inset-0"
            style={{
              opacity: 0.015,
              backgroundImage: `url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIj48ZmlsdGVyIGlkPSJhIiB4PSIwIiB5PSIwIj48ZmVUdXJidWxlbmNlIGJhc2VGcmVxdWVuY3k9Ii43NSIgc3RpdGNoVGlsZXM9InN0aXRjaCIgdHlwZT0iZnJhY3RhbE5vaXNlIi8+PC9maWx0ZXI+PHJlY3Qgd2lkdGg9IjMwMCIgaGVpZ2h0PSIzMDAiIGZpbHRlcj0idXJsKCNhKSIgb3BhY2l0eT0iMC4wNSIvPjwvc3ZnPg==")`,
            }}
          />
        </div>

        {/* HEADER - Fixed top */}
        <motion.header
          className="fixed top-0 left-0 right-0 p-6"
          style={{ zIndex: 50 }}
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Link href="/">
            <span className="text-2xl font-semibold tracking-tight text-white">
              spike
            </span>
          </Link>
        </motion.header>

        {/* MAIN CONTENT - Centered */}
        <main
          className="relative min-h-screen flex flex-col items-center justify-center px-6"
          style={{ zIndex: 10 }}
        >
          {/* Heading */}
          <motion.h1
            className="heading-serif text-6xl md:text-7xl lg:text-8xl text-center text-white mb-12"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            Welcome back.
          </motion.h1>

          {/* Login form */}
          <motion.div
            className="w-full max-w-md"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <SignIn
              appearance={{
                elements: {
                  rootBox: "w-full",
                  card: "bg-transparent shadow-none p-0",
                  cardBox: "shadow-none bg-transparent",
                  header: "hidden",
                  socialButtonsBlockButton:
                    "flex-1 h-12 rounded-lg border border-white/10 bg-white/5 backdrop-blur-sm hover:bg-white/10 transition-all duration-200",
                  socialButtonsBlockButtonText: "text-white",
                  dividerLine: "bg-white/10",
                  dividerText: "text-white/40",
                  formFieldLabel: "text-sm font-medium text-white/90 mb-1.5",
                  formFieldInput:
                    "bg-white/5 border border-white/10 text-white placeholder:text-white/30 rounded-lg h-12 px-4 backdrop-blur-sm focus:border-[#4ade80]/50 focus:ring-0 transition-colors",
                  formButtonPrimary:
                    "w-full h-12 text-base font-medium bg-[#4ade80] hover:bg-[#4ade80]/90 text-[#0a1210] rounded-lg transition-all shadow-[0_0_20px_rgba(74,222,128,0.25)]",
                  footerActionLink: "text-[#4ade80] hover:text-[#4ade80]/80",
                  identityPreviewEditButton: "text-[#4ade80]",
                  formFieldInputShowPasswordButton: "text-white/40 hover:text-white/60",
                  formFieldErrorText: "text-red-400 text-sm",
                  alert: "bg-red-500/10 border border-red-500/20 text-red-400 rounded-lg",
                  footer: "hidden",
                },
                layout: {
                  socialButtonsPlacement: "top",
                  showOptionalFields: false,
                },
              }}
            />

            <p className="text-center text-sm text-white/50 mt-5">
              Don't have an account?{" "}
              <Link
                href="/sign-up"
                className="text-[#4ade80] hover:text-[#4ade80]/80 underline underline-offset-4 transition-colors"
              >
                Sign up
              </Link>
            </p>
          </motion.div>
        </main>

        {/* MARKET TICKER - Fixed bottom center */}
        <div className="fixed bottom-8 left-1/2 -translate-x-1/2" style={{ zIndex: 100 }}>
          <MarketTicker />
        </div>
      </div>
    </>
  );
}
