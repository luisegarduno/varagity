import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Instrument_Serif } from "next/font/google";

import "katex/dist/katex.min.css";
import "./globals.css";

import { AppearanceApplier } from "@/components/AppearanceApplier";
import { CommandPalette } from "@/components/palette/CommandPalette";
import { QueryBusBridge } from "@/components/query-bus-bridge";
import { QueryProvider } from "@/components/query-provider";
import { MobileTopBar } from "@/components/shell/MobileTopBar";
import { Sidebar } from "@/components/Sidebar";
import { ThemeProvider } from "@/components/theme-provider";
import { TooltipProvider } from "@/components/ui/tooltip";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// The editorial serif accent (`font-heading`): wordmark and panel headings.
const instrumentSerif = Instrument_Serif({
  variable: "--font-instrument-serif",
  weight: "400",
  style: ["normal", "italic"],
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Varagity",
  description:
    "Contextual-Retrieval RAG over your own corpus — grounded, cited, transparent.",
};

// Browser-chrome tint matching `--background` in each theme (globals.css).
export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fbfbfd" },
    { media: "(prefers-color-scheme: dark)", color: "#090a0d" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${geistSans.variable} ${geistMono.variable} ${instrumentSerif.variable} h-full antialiased`}
    >
      <body className="h-dvh overflow-hidden">
        <a href="#main" className="skip-link">
          Skip to content
        </a>
        <QueryProvider>
          <QueryBusBridge />
          <ThemeProvider>
            <AppearanceApplier />
            {/* Mounted once at the root: ⌘K works everywhere and navigation
                can't unmount the palette mid-command. */}
            <CommandPalette />
            <TooltipProvider>
              <div className="flex h-full flex-col">
                <MobileTopBar />
                <div className="flex min-h-0 flex-1">
                  <Sidebar />
                  <main id="main" className="flex min-w-0 flex-1 flex-col">
                    {children}
                  </main>
                </div>
              </div>
            </TooltipProvider>
          </ThemeProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
