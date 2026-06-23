"use client";

/**
 * Circle-expand button: a solid disc grows from the corner to fill the pill on
 * hover, and the label flips to the paper colour. Flat fills only.
 */
import { motion } from "framer-motion";

export function CircleExpandButton({
  children,
  onClick,
  type = "button",
  className = "",
}: {
  children: React.ReactNode;
  onClick?: () => void;
  type?: "button" | "submit";
  className?: string;
}) {
  return (
    <motion.button
      type={type}
      onClick={onClick}
      initial="rest"
      whileHover="hover"
      whileTap={{ scale: 0.97 }}
      animate="rest"
      className={`relative isolate overflow-hidden rounded-full border border-ink-900 px-7 py-3 font-medium ${className}`}
    >
      <motion.span
        aria-hidden
        className="absolute left-4 top-1/2 -z-10 h-4 w-4 -translate-y-1/2 rounded-full bg-ink-900"
        variants={{ rest: { scale: 1 }, hover: { scale: 22 } }}
        transition={{ type: "spring", stiffness: 220, damping: 28 }}
      />
      <motion.span
        className="relative z-10"
        variants={{ rest: { color: "#4a4031" }, hover: { color: "#f7f1e9" } }}
      >
        {children}
      </motion.span>
    </motion.button>
  );
}
