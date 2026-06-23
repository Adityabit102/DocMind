"use client";

/**
 * Horizontal auto-scrolling marquee (the "ImageScroller" pattern). Renders a row
 * of pill items that loop seamlessly and pause on hover. Here it scrolls the
 * supported file formats / capabilities.
 */
import { motion } from "framer-motion";

export function ImageScroller({
  items,
  duration = 26,
}: {
  items: string[];
  duration?: number;
}) {
  const loop = [...items, ...items];
  return (
    <div className="group relative overflow-hidden py-2">
      {/* solid edge masks (flat, not gradient) */}
      <div className="pointer-events-none absolute inset-y-0 left-0 z-10 w-16 bg-paper" />
      <div className="pointer-events-none absolute inset-y-0 right-0 z-10 w-16 bg-paper" />
      <motion.div
        className="flex w-max gap-3"
        animate={{ x: ["0%", "-50%"] }}
        transition={{ duration, ease: "linear", repeat: Infinity }}
        style={{ willChange: "transform" }}
      >
        {loop.map((item, i) => (
          <span
            key={i}
            className="flex items-center gap-2 whitespace-nowrap rounded-full border border-sand bg-cream px-5 py-2 font-mono text-sm text-ink-700"
          >
            <span className="h-2 w-2 rounded-full bg-ink" />
            {item}
          </span>
        ))}
      </motion.div>
    </div>
  );
}
