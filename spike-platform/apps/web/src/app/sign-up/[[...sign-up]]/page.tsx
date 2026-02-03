import { SignUp } from "@clerk/nextjs";
import { Zap, Check } from "lucide-react";
import Link from "next/link";

export default function SignUpPage() {
  const benefits = [
    "FinScore ratings for 5000+ stocks",
    "AI-powered portfolio analysis",
    "Real-time market insights",
    "Strategy-GPT natural language queries",
    "Legend Agents (Buffett, Lynch, Dalio)",
    "Smart theme portfolios",
  ];

  return (
    <div className="min-h-screen flex bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950">
      {/* Left Panel - Branding */}
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-between p-12 bg-gradient-to-br from-indigo-600 to-violet-600">
        <div>
          <Link href="/" className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl bg-white/20 backdrop-blur flex items-center justify-center">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <span className="text-2xl font-bold text-white">SPIKE</span>
          </Link>
        </div>

        <div className="space-y-6">
          <h1 className="text-4xl font-bold text-white leading-tight">
            Start Your
            <br />
            Wealth Intelligence
            <br />
            Journey
          </h1>
          <p className="text-lg text-white/80 max-w-md">
            Join 50,000+ investors using AI-powered intelligence to make smarter
            investment decisions.
          </p>

          <div className="space-y-3 pt-4">
            {benefits.map((benefit, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className="w-5 h-5 rounded-full bg-white/20 flex items-center justify-center">
                  <Check className="w-3 h-3 text-white" />
                </div>
                <span className="text-white/90">{benefit}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="text-white/60 text-sm">
          SEBI Registered Research Analyst | Data Localized in India
        </div>
      </div>

      {/* Right Panel - Sign Up Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          <div className="lg:hidden mb-8 text-center">
            <Link href="/" className="inline-flex items-center gap-2">
              <div className="w-10 h-10 rounded-xl bg-spike-gradient flex items-center justify-center">
                <Zap className="w-6 h-6 text-white" />
              </div>
              <span className="text-2xl font-bold text-white">SPIKE</span>
            </Link>
          </div>

          <SignUp
            appearance={{
              elements: {
                rootBox: "mx-auto",
                card: "bg-white/5 border border-white/10 shadow-2xl",
                headerTitle: "text-white",
                headerSubtitle: "text-slate-400",
                socialButtonsBlockButton:
                  "bg-white/5 border border-white/10 text-white hover:bg-white/10",
                socialButtonsBlockButtonText: "text-white",
                dividerLine: "bg-white/10",
                dividerText: "text-slate-400",
                formFieldLabel: "text-slate-300",
                formFieldInput:
                  "bg-white/5 border-white/10 text-white placeholder:text-slate-500",
                formButtonPrimary:
                  "bg-spike-gradient hover:opacity-90 text-white",
                footerActionLink: "text-spike-primary hover:text-spike-primary/80",
                identityPreviewText: "text-white",
                identityPreviewEditButton: "text-spike-primary",
              },
            }}
          />

          <p className="mt-6 text-center text-xs text-slate-400">
            By signing up, you agree to our{" "}
            <Link href="/terms" className="text-spike-primary hover:underline">
              Terms of Service
            </Link>{" "}
            and{" "}
            <Link href="/privacy" className="text-spike-primary hover:underline">
              Privacy Policy
            </Link>
            . Investment in securities market are subject to market risks.
          </p>
        </div>
      </div>
    </div>
  );
}
