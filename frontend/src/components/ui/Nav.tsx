"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { motion } from "framer-motion";

const LINKS = [
  { href: "/chat", label: "Chat" },
  { href: "/documents", label: "Documents" },
  { href: "/evaluation", label: "Evaluation" },
  { href: "/settings", label: "Settings" },
  { href: "/about", label: "About" },
];

export function Nav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 border-b border-sand bg-paper/95">
      <nav className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-ink-900 bg-ink-900 font-display text-paper">
            D
          </span>
          <span className="font-display text-xl font-semibold text-ink-900">DocMind</span>
        </Link>

        <div className="hidden items-center gap-1 md:flex">
          {LINKS.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`relative rounded-full px-4 py-2 text-sm transition-colors ${
                  active ? "text-ink-900" : "text-ink-700 hover:text-ink-900"
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-0 -z-10 rounded-full bg-cream"
                    transition={{ type: "spring", stiffness: 380, damping: 32 }}
                  />
                )}
                {l.label}
              </Link>
            );
          })}
          <Link href="/chat" className="btn ml-2 px-5 py-2 text-sm">
            Ask a question
          </Link>
        </div>

        <button
          aria-label="Toggle menu"
          className="md:hidden"
          onClick={() => setOpen((v) => !v)}
        >
          <div className="space-y-1.5">
            <span className="block h-0.5 w-6 bg-ink-900" />
            <span className="block h-0.5 w-6 bg-ink-900" />
            <span className="block h-0.5 w-6 bg-ink-900" />
          </div>
        </button>
      </nav>

      {open && (
        <div className="border-t border-sand px-5 py-3 md:hidden">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              onClick={() => setOpen(false)}
              className="block rounded-lg px-3 py-2 text-ink-700 hover:bg-cream"
            >
              {l.label}
            </Link>
          ))}
        </div>
      )}
    </header>
  );
}
