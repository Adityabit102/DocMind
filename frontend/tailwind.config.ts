import type { Config } from "tailwindcss";

/**
 * DocMind design system.
 * Warm taupe → cream palette. Deliberately NO gradients and NO glassmorphism:
 * flat solid fills, hand-tuned ink-on-paper shadows, editorial type.
 */
const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: "#92836c", // deepest taupe — primary text / accents
        clay: "#a8957f", // secondary surface
        sand: "#c8bba9", // borders / muted
        cream: "#ecdcc4", // cards / highlight
        paper: "#f7f1e9", // page background
        "ink-900": "#4a4031",
        "ink-700": "#6b5f4a",
      },
      fontFamily: {
        display: ["var(--font-display)", "Georgia", "serif"],
        sans: ["var(--font-body)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      boxShadow: {
        // soft, single-direction "object resting on paper" shadows — never glow
        paper: "0 2px 0 0 #d8c9b3, 0 14px 30px -18px rgba(74,64,49,0.45)",
        "paper-lg": "0 3px 0 0 #cdbb9f, 0 28px 50px -24px rgba(74,64,49,0.5)",
        inset: "inset 0 1px 0 0 rgba(255,255,255,0.5)",
      },
      borderRadius: {
        xl2: "1.25rem",
      },
      keyframes: {
        "fade-up": {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "fade-up": "fade-up 0.7s cubic-bezier(0.2,0.7,0.2,1) both",
      },
    },
  },
  plugins: [],
};
export default config;
