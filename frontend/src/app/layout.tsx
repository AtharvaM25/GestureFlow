import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";

import { SiteHeader } from "@/components/site-header";
import { AuthProvider } from "@/lib/auth";

import "./globals.css";

const sans = Geist({ variable: "--font-sans", subsets: ["latin"] });
const mono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "GestureFlow", template: "%s · GestureFlow" },
  description: "Real-time hand-sign recognition. Camera frames stay on your device; only hand landmarks are sent.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`dark ${sans.variable} ${mono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col bg-background font-sans text-foreground">
        <AuthProvider>
          <SiteHeader />
          <main className="flex flex-1 flex-col">{children}</main>
          <footer className="border-t border-border/60">
            <div className="mx-auto flex h-12 w-full max-w-6xl items-center justify-between px-4 text-xs text-muted-foreground">
              <span>GestureFlow</span>
              <Link href="/privacy" className="hover:text-foreground">Privacy</Link>
            </div>
          </footer>
        </AuthProvider>
      </body>
    </html>
  );
}
