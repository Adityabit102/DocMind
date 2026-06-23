"use client";

import { motion } from "framer-motion";

/**
 * The DocMind document glyph, animated: the sheet outline draws itself, a corner
 * folds, text lines stroke in one by one, and a magnifier settles over the page.
 * Flat palette strokes only — the same editorial language as the rest of the UI.
 * Reused by the boot loader and the route-transition curtain so the motif is
 * consistent everywhere.
 */
const INK = "#4a4031";
const CLAY = "#a8957f";
const SAND = "#c8bba9";

const draw = {
  hidden: { pathLength: 0, opacity: 0 },
  show: (i: number) => ({
    pathLength: 1,
    opacity: 1,
    transition: {
      pathLength: { duration: 0.7, delay: 0.15 * i, ease: [0.4, 0, 0.2, 1] },
      opacity: { duration: 0.2, delay: 0.15 * i },
    },
  }),
};

const lineIn = {
  hidden: { scaleX: 0, opacity: 0 },
  show: (i: number) => ({
    scaleX: 1,
    opacity: 1,
    transition: { duration: 0.4, delay: 0.5 + i * 0.12, ease: [0.2, 0.7, 0.2, 1] },
  }),
};

export function DocMark({ size = 96 }: { size?: number }) {
  const lines = [
    { y: 26, w: 26 },
    { y: 33, w: 30 },
    { y: 40, w: 22 },
    { y: 47, w: 28 },
    { y: 54, w: 18 },
  ];
  return (
    <motion.svg
      width={size}
      height={size * 1.2}
      viewBox="0 0 64 76"
      fill="none"
      initial="hidden"
      animate="show"
    >
      {/* page body */}
      <motion.path
        d="M14 6 H42 L54 18 V70 H14 Z"
        stroke={INK}
        strokeWidth={2}
        strokeLinejoin="round"
        variants={draw}
        custom={0}
      />
      {/* folded corner */}
      <motion.path
        d="M42 6 V18 H54"
        stroke={CLAY}
        strokeWidth={2}
        strokeLinejoin="round"
        variants={draw}
        custom={1}
      />
      {/* text lines */}
      {lines.map((l, i) => (
        <motion.rect
          key={i}
          x={20}
          y={l.y}
          width={l.w}
          height={2.4}
          rx={1.2}
          fill={SAND}
          style={{ originX: 0 }}
          variants={lineIn}
          custom={i}
        />
      ))}
      {/* magnifier sweeping over the page */}
      <motion.g
        initial={{ opacity: 0, x: -6, y: -6 }}
        animate={{ opacity: 1, x: 0, y: 0 }}
        transition={{ delay: 1.15, duration: 0.5, ease: [0.2, 0.7, 0.2, 1] }}
      >
        <circle cx={44} cy={50} r={9} stroke={INK} strokeWidth={2} fill="none" />
        <line x1={51} y1={57} x2={58} y2={64} stroke={INK} strokeWidth={2.4} strokeLinecap="round" />
      </motion.g>
    </motion.svg>
  );
}
