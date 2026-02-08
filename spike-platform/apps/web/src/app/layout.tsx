import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Playfair_Display } from "next/font/google";
import { ClerkProvider } from "@clerk/nextjs";
import { ThemeProvider } from "@/components/providers/theme-provider";
import { QueryProvider } from "@/components/providers/query-provider";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
});

const playfairDisplay = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-playfair",
});

export const metadata: Metadata = {
  title: {
    default: "SPIKE - AI Wealth Intelligence Platform",
    template: "%s | SPIKE",
  },
  description:
    "India's first AI-powered wealth intelligence platform. Get institutional-grade insights, personalized strategies, and autonomous portfolio management.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} ${playfairDisplay.variable} font-sans antialiased`}
      >
        <ClerkProvider
          appearance={{
            variables: {
              colorPrimary: "#638C82",
              colorTextOnPrimaryBackground: "#0D1B17",
            },
            elements: {
              formButtonPrimary:
                "bg-gradient-to-r from-spike-sage to-spike-accent hover:opacity-90 text-spike-dark",
              card: "bg-spike-deep border border-white/10 shadow-2xl",
              headerTitle: "text-spike-cream",
              headerSubtitle: "text-spike-muted",
              socialButtonsBlockButton:
                "bg-white/5 border border-white/10 text-spike-cream hover:bg-white/10",
              formFieldLabel: "text-spike-muted",
              formFieldInput:
                "bg-white/5 border-white/10 text-spike-cream",
              footerActionLink: "text-spike-sage hover:text-spike-accent",
            },
          }}
        >
          <ThemeProvider
            attribute="class"
            defaultTheme="dark"
            enableSystem
            disableTransitionOnChange
          >
            <QueryProvider>{children}</QueryProvider>
          </ThemeProvider>
        </ClerkProvider>
      </body>
    </html>
  );
}
