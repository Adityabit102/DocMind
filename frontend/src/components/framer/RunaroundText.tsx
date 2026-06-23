"use client";

/**
 * RunaroundTextWrap: body text flows around a floating circular shape using CSS
 * `shape-outside`, the classic editorial run-around. The float gently bobs.
 */
import { motion } from "framer-motion";

export function RunaroundText({
  children,
  badge,
}: {
  children: React.ReactNode;
  badge?: React.ReactNode;
}) {
  return (
    <div className="text-lg leading-relaxed text-ink-700">
      <motion.div
        className="float-left mr-6 mb-2 flex h-32 w-32 items-center justify-center rounded-full border border-ink bg-cream text-center font-display text-ink-900 [shape-outside:circle(50%)]"
        animate={{ y: [0, -8, 0] }}
        transition={{ duration: 5, ease: "easeInOut", repeat: Infinity }}
      >
        {badge ?? "RAG"}
      </motion.div>
      {children}
    </div>
  );
}
