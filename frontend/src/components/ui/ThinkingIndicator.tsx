"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";

/**
 * Shown in the assistant bubble between "send" and the first streamed token, so
 * the wait never reads as frozen. Three pulsing dots plus a status label that
 * advances through the pipeline stages (retrieve → read → write). The labels are
 * time-paced rather than wired to real backend events — the chain does its
 * retrieval before the token stream opens — but they honestly describe the work.
 */
const STAGES = [
  "Searching your documents",
  "Reading the most relevant passages",
  "Reranking by relevance",
  "Writing a grounded answer",
];

export function ThinkingIndicator() {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const id = setInterval(() => {
      setStage((s) => Math.min(s + 1, STAGES.length - 1));
    }, 2600);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-1">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className="h-1.5 w-1.5 rounded-full bg-ink"
            animate={{ opacity: [0.25, 1, 0.25], y: [0, -2, 0] }}
            transition={{ duration: 1, repeat: Infinity, delay: i * 0.18, ease: "easeInOut" }}
          />
        ))}
      </div>
      <AnimatePresence mode="wait">
        <motion.span
          key={stage}
          className="font-mono text-xs text-clay"
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -6 }}
          transition={{ duration: 0.3 }}
        >
          {STAGES[stage]}…
        </motion.span>
      </AnimatePresence>
    </div>
  );
}
