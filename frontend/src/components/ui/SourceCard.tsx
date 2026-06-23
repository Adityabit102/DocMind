"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import type { SourceChunk } from "@/lib/types";

const WORD_RE = /[a-z0-9]+/g;
const tokenize = (s: string) => new Set(s.toLowerCase().match(WORD_RE) ?? []);

/** Split chunk into sentences and pick the one most overlapping the answer —
 *  the passage the answer most likely drew from — to highlight on expand. */
function bestSentence(text: string, answer: string): string | null {
  const ans = tokenize(answer);
  if (ans.size === 0) return null;
  const sentences = text.split(/(?<=[.!?])\s+/).filter((s) => s.trim().length > 12);
  let best: string | null = null;
  let bestScore = 0.18; // require a minimum overlap to bother highlighting
  for (const s of sentences) {
    const toks = Array.from(tokenize(s));
    if (toks.length === 0) continue;
    const overlap = toks.filter((t) => ans.has(t)).length / toks.length;
    if (overlap > bestScore) {
      bestScore = overlap;
      best = s.trim();
    }
  }
  return best;
}

/** Collapsible citation: page + file header, expands to the chunk with the
 *  answer's most-relevant sentence highlighted. */
export function SourceCard({
  source,
  index,
  answer = "",
}: {
  source: SourceChunk;
  index: number;
  answer?: string;
}) {
  const [open, setOpen] = useState(false);
  const cited = answer ? bestSentence(source.text, answer) : null;
  return (
    <div id={`source-${index}`} className="card-cream overflow-hidden scroll-mt-24">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
      >
        <span className="flex items-center gap-3">
          <span className="flex h-6 w-6 items-center justify-center rounded-md border border-sand bg-paper font-mono text-xs text-ink-700">
            {index + 1}
          </span>
          <span className="font-mono text-xs text-ink-700">
            {source.filename} · p.{source.page_number}
          </span>
        </span>
        <span className="flex items-center gap-2">
          <span className="tag">{(source.relevance_score * 100).toFixed(0)}%</span>
          <motion.span animate={{ rotate: open ? 180 : 0 }} className="text-ink">
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2}>
              <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </motion.span>
        </span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
          >
            <p className="border-t border-sand px-4 py-3 text-sm leading-relaxed text-ink-900">
              {cited && source.text.includes(cited) ? (
                <>
                  {source.text.slice(0, source.text.indexOf(cited))}
                  <mark className="rounded bg-cream px-0.5 text-ink-900 ring-1 ring-clay/50">
                    {cited}
                  </mark>
                  {source.text.slice(source.text.indexOf(cited) + cited.length)}
                </>
              ) : (
                source.text
              )}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
