import type { Metadata } from "next";
import { Fraunces, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/ui/Nav";
import { Footer } from "@/components/ui/Footer";
import { PageFrame } from "@/components/ui/PageFrame";

const display = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  axes: ["opsz"],
});
const body = Inter({ subsets: ["latin"], variable: "--font-body" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "DocMind — Grounded answers from your documents",
  description:
    "Upload documents and ask questions in natural language. Every answer is cited back to its exact source page. Hybrid retrieval, reranking, and grounded generation.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body className="min-h-screen antialiased">
        <Nav />
        {/* Reserve at least a full viewport (minus the nav) for page content so
            short pages keep the footer just below the fold — it then reveals
            cleanly on scroll instead of peeking up from the bottom. */}
        <main className="mx-auto min-h-[calc(100vh-4rem)] w-full">
          <PageFrame>{children}</PageFrame>
        </main>
        <Footer />
      </body>
    </html>
  );
}
