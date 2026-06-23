"use client";

import { motion } from "framer-motion";
import { DocMark } from "./DocMark";

/**
 * Full-screen boot loader: the document glyph assembles itself over a paper
 * field while a thin progress rule fills and the wordmark rises. Exits by
 * lifting up like a sheet being pulled away to reveal the page beneath.
 */
export function LoadingScreen() {
  return (
    <motion.div
      className="fixed inset-0 z-[100] flex flex-col items-center justify-center bg-paper"
      style={{
        backgroundImage:
          "radial-gradient(rgba(146,131,108,0.07) 1px, transparent 1px)",
        backgroundSize: "22px 22px",
      }}
      initial={{ opacity: 1 }}
      exit={{ y: "-100%", transition: { duration: 0.7, ease: [0.7, 0, 0.3, 1] } }}
    >
      <DocMark size={104} />

      <motion.div
        className="mt-8 flex items-center gap-2.5"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9, duration: 0.5 }}
      >
        <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-ink-900 bg-ink-900 font-display text-paper">
          D
        </span>
        <span className="font-display text-2xl font-semibold text-ink-900">DocMind</span>
      </motion.div>

      <motion.p
        className="mt-2 font-mono text-xs uppercase tracking-[0.25em] text-clay"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2, duration: 0.5 }}
      >
        Indexing your knowledge
      </motion.p>

      {/* progress rule */}
      <div className="mt-6 h-px w-44 overflow-hidden bg-sand/60">
        <motion.div
          className="h-full bg-ink-900"
          initial={{ x: "-100%" }}
          animate={{ x: "0%" }}
          transition={{ duration: 1.5, ease: [0.4, 0, 0.2, 1] }}
        />
      </div>
    </motion.div>
  );
}
