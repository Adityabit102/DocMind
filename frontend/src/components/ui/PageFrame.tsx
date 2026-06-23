"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { LoadingScreen } from "./LoadingScreen";

/**
 * Orchestrates the app's motion shell:
 *  - a one-time boot loader (the document glyph assembling) on first load;
 *  - a paper "curtain" stamped with the DocMind mark that wipes across on every
 *    route change, used as the transition screen between pages;
 *  - a subtle content crossfade so pages never hard-cut.
 * All of it collapses to instant under prefers-reduced-motion.
 */
export function PageFrame({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const reduce = useReducedMotion();
  const [booting, setBooting] = useState(true);
  const [curtain, setCurtain] = useState(0);
  const firstRender = useRef(true);

  // Boot loader — shown once while the first page settles.
  useEffect(() => {
    if (reduce) {
      setBooting(false);
      return;
    }
    const t = setTimeout(() => setBooting(false), 1900);
    return () => clearTimeout(t);
  }, [reduce]);

  // Fire the curtain on each navigation (not the initial mount).
  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    if (!reduce) setCurtain((c) => c + 1);
  }, [pathname, reduce]);

  return (
    <>
      <AnimatePresence>{booting && <LoadingScreen key="boot" />}</AnimatePresence>

      <AnimatePresence mode="wait">
        <motion.div
          key={pathname}
          className="mx-auto w-full"
          initial={{ opacity: 0, y: reduce ? 0 : 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: reduce ? 0 : -8 }}
          transition={{ duration: 0.4, ease: [0.2, 0.7, 0.2, 1] }}
        >
          {children}
        </motion.div>
      </AnimatePresence>

      <RouteCurtain trigger={curtain} />
    </>
  );
}

function RouteCurtain({ trigger }: { trigger: number }) {
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (trigger === 0) return;
    setShow(true);
    const t = setTimeout(() => setShow(false), 900);
    return () => clearTimeout(t);
  }, [trigger]);

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          key={trigger}
          aria-hidden
          className="pointer-events-none fixed inset-0 z-[90] flex items-center justify-center bg-cream"
          initial={{ y: "100%" }}
          animate={{ y: ["100%", "0%", "0%", "-100%"] }}
          transition={{ duration: 0.9, times: [0, 0.42, 0.58, 1], ease: [0.7, 0, 0.3, 1] }}
        >
          <motion.div
            className="flex items-center gap-3"
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: [0, 1, 1, 0], scale: [0.92, 1, 1, 0.96] }}
            transition={{ duration: 0.9, times: [0, 0.4, 0.62, 1] }}
          >
            <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-ink-900 bg-ink-900 font-display text-paper">
              D
            </span>
            <span className="font-display text-xl font-semibold text-ink-900">DocMind</span>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
