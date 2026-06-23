"use client";

/**
 * Project-hover card: the title row sits flush; on hover the card lifts, a hidden
 * description panel slides up, and a small index marker rotates. Used for the
 * feature/capability grid.
 */
import { motion } from "framer-motion";

export function ProjectHoverCard({
  index,
  title,
  description,
  icon,
}: {
  index: string;
  title: string;
  description: string;
  icon?: React.ReactNode;
}) {
  return (
    <motion.article
      initial="rest"
      whileHover="hover"
      animate="rest"
      variants={{ rest: { y: 0 }, hover: { y: -6 } }}
      transition={{ type: "spring", stiffness: 300, damping: 26 }}
      className="card relative overflow-hidden p-6"
    >
      <div className="flex items-start justify-between">
        <span className="font-mono text-xs text-clay">{index}</span>
        <motion.span
          variants={{ rest: { rotate: 0, color: "#a8957f" }, hover: { rotate: 45, color: "#4a4031" } }}
          className="text-ink"
        >
          {icon ?? (
            <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth={2}>
              <path d="M7 17L17 7M17 7H8M17 7v9" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </motion.span>
      </div>
      <h3 className="mt-6 font-display text-xl text-ink-900">{title}</h3>
      <motion.p
        variants={{ rest: { opacity: 0.7, height: "auto" }, hover: { opacity: 1 } }}
        className="mt-2 text-sm leading-relaxed text-ink-700"
      >
        {description}
      </motion.p>
      <motion.div
        aria-hidden
        variants={{ rest: { scaleX: 0 }, hover: { scaleX: 1 } }}
        transition={{ type: "spring", stiffness: 260, damping: 30 }}
        className="absolute bottom-0 left-0 h-1 w-full origin-left bg-ink-900"
      />
    </motion.article>
  );
}
