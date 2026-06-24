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
  // Soft edge fade (alpha mask, not a colour box) so the pills dissolve into
  // whatever is behind them at the screen edges — complements any background.
  const fade =
    "linear-gradient(to right, transparent 0, #000 5%, #000 95%, transparent 100%)";
  return (
    <div
      className="group relative w-full overflow-hidden py-2"
      style={{ maskImage: fade, WebkitMaskImage: fade }}
    >
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
