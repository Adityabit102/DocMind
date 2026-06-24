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
  // Build one "set" wide enough to overflow even ultrawide screens by repeating
  // the items, then render it twice and slide by exactly one set width (-50%).
  // This guarantees a seamless loop with no gap: as set A scrolls off the left,
  // the identical set B is already filling the right.
  const REPEAT = 5;
  const set = Array.from({ length: REPEAT }, () => items).flat();
  const loop = [...set, ...set];
  // Soft edge fade (alpha mask, not a colour box) so the pills dissolve into
  // whatever is behind them at the screen edges — complements any background.
  const fade =
    "linear-gradient(to right, transparent 0, #000 6%, #000 94%, transparent 100%)";
  return (
    <div
      className="group relative w-full overflow-hidden py-2"
      style={{ maskImage: fade, WebkitMaskImage: fade }}
    >
      <motion.div
        className="flex w-max gap-3"
        animate={{ x: ["0%", "-50%"] }}
        transition={{ duration: duration * REPEAT, ease: "linear", repeat: Infinity }}
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
