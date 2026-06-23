"use client";

import type { SentenceSupport, SourceChunk } from "@/lib/types";

/**
 * Renders an answer with two grounding affordances:
 *  - sentences the backend flagged as unsupported by the retrieved context are
 *    underlined in amber (potential hallucinations);
 *  - inline `[Source: file, Page: N]` citations become clickable chips that
 *    scroll to and flash the matching source card (click-to-source).
 */
const CITATION_RE = /\[Source:\s*([^,\]]+?)\s*,\s*Page:\s*([^\]]+?)\]/g;

function onCiteClick(filename: string, page: string, sources: SourceChunk[]) {
  const idx = sources.findIndex(
    (s) => s.filename === filename.trim() && String(s.page_number) === page.trim(),
  );
  const target = idx >= 0 ? idx : sources.findIndex((s) => s.filename === filename.trim());
  if (target < 0) return;
  const el = document.getElementById(`source-${target}`);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("source-flash");
  setTimeout(() => el.classList.remove("source-flash"), 1400);
}

function renderWithCitations(text: string, sources: SourceChunk[]) {
  const parts: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  CITATION_RE.lastIndex = 0;
  let key = 0;
  while ((m = CITATION_RE.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const [, file, page] = m;
    parts.push(
      <button
        key={`c-${key++}`}
        onClick={() => onCiteClick(file, page, sources)}
        className="mx-0.5 inline-flex items-center rounded border border-sand bg-cream px-1 font-mono text-[10px] text-ink-700 align-baseline hover:bg-paper"
        title={`Jump to ${file.trim()} · p.${page.trim()}`}
      >
        {file.trim()} p.{page.trim()}
      </button>,
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

export function GroundedAnswer({
  text,
  spans,
  sources,
}: {
  text: string;
  spans?: SentenceSupport[];
  sources: SourceChunk[];
}) {
  // No grounding data yet (e.g. still streaming) → plain text + citations.
  const unsupported = (spans ?? []).filter((s) => !s.supported && s.text.trim());
  if (unsupported.length === 0) {
    return (
      <p className="whitespace-pre-wrap leading-relaxed text-ink-900">
        {renderWithCitations(text, sources)}
      </p>
    );
  }
  // Highlight unsupported sentences inline by splitting the answer on them.
  const flagged = new Set(unsupported.map((s) => s.text.trim()));
  const allSentences = (spans ?? []).map((s) => s.text);
  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  allSentences.forEach((sentence, i) => {
    const at = text.indexOf(sentence, cursor);
    if (at < 0) return;
    if (at > cursor) nodes.push(renderWithCitations(text.slice(cursor, at), sources));
    const body = renderWithCitations(sentence, sources);
    if (flagged.has(sentence.trim())) {
      nodes.push(
        <span
          key={`u-${i}`}
          className="decoration-wavy underline decoration-amber-600/70 underline-offset-2"
          title={`Not clearly supported by the retrieved context`}
        >
          {body}
        </span>,
      );
    } else {
      nodes.push(<span key={`s-${i}`}>{body}</span>);
    }
    cursor = at + sentence.length;
  });
  if (cursor < text.length) nodes.push(renderWithCitations(text.slice(cursor), sources));
  return <p className="whitespace-pre-wrap leading-relaxed text-ink-900">{nodes}</p>;
}
