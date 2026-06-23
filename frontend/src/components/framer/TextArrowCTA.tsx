"use client";

/** Text + arrow call-to-action. Arrow slides forward and the label shifts on hover. */
import Link from "next/link";
import { motion } from "framer-motion";

const MotionLink = motion(Link);

export function TextArrowCTA({
  href,
  children,
  className = "",
}: {
  href: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <MotionLink
      href={href}
      initial="rest"
      whileHover="hover"
      animate="rest"
      className={`inline-flex items-center gap-2 ${className}`}
    >
      <motion.span
        className="font-medium text-ink-900"
        variants={{ rest: { x: 0 }, hover: { x: 2 } }}
      >
        {children}
      </motion.span>
      <span className="relative h-5 w-6 overflow-hidden">
        <motion.svg
          viewBox="0 0 24 24"
          className="absolute inset-0 h-5 w-6 text-ink-900"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          variants={{ rest: { x: -2, opacity: 0.85 }, hover: { x: 4, opacity: 1 } }}
          transition={{ type: "spring", stiffness: 400, damping: 22 }}
        >
          <path d="M5 12h14M13 6l6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
        </motion.svg>
      </span>
    </MotionLink>
  );
}
