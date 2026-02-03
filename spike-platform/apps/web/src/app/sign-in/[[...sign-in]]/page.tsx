import { SignIn } from "@clerk/nextjs";
import { Zap } from "lucide-react";
import Link from "next/link";

export default function SignInPage() {
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
            India's First
            <br />
            AI Wealth Intelligence
            <br />
            Platform
          </h1>
          <p className="text-lg text-white/80 max-w-md">
            Get institutional-grade insights, personalized strategies, and
            autonomous portfolio management powered by 12 AI agents.
          </p>

          <div className="flex items-center gap-8 pt-4">
            <div>
              <div className="text-3xl font-bold text-white">50K+</div>
              <div className="text-white/60 text-sm">Active Users</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-white">₹500Cr+</div>
              <div className="text-white/60 text-sm">AUM Analyzed</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-white">98.5%</div>
              <div className="text-white/60 text-sm">Uptime</div>
            </div>
          </div>
        </div>

        <div className="text-white/60 text-sm">
          SEBI Registered Research Analyst | 100% Compliant
        </div>
      </div>

      {/* Right Panel - Sign In Form */}
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

          <SignIn
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
        </div>
      </div>
    </div>
  );
}
