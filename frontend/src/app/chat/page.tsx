"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { api, streamQuery } from "@/lib/api";
import type { Confidence, DocumentRecord, SentenceSupport } from "@/lib/types";
import { useChats, type Conversation, type Turn } from "@/lib/useChats";
import { ConfidenceBadge } from "@/components/ui/ConfidenceBadge";
import { GroundingBadge } from "@/components/ui/GroundingBadge";
import { GroundedAnswer } from "@/components/ui/GroundedAnswer";
import { SourceCard } from "@/components/ui/SourceCard";
import { ThinkingIndicator } from "@/components/ui/ThinkingIndicator";
import { CircleExpandButton } from "@/components/framer/CircleExpandButton";
import { Scene3D } from "@/components/three/Scene3D";

const MODES = ["hybrid", "similarity", "mmr"] as const;

export default function ChatPage() {
  const { conversations, activeId, turns, setTurns, newChat, selectChat, deleteChat } = useChats();
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [docCount, setDocCount] = useState<number | null>(null);

  // retrieval controls
  const [mode, setMode] = useState<(typeof MODES)[number]>("hybrid");
  const [topK, setTopK] = useState(20);
  const [temperature, setTemperature] = useState(0);
  // Off by default: with a local model, LLM-driven query expansion adds several
  // slow round-trips for little gain. Users can switch it on for hard queries.
  const [expansion, setExpansion] = useState(false);
  const [rerank, setRerank] = useState(true);

  // document scope: empty set = ask across all indexed documents
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());

  // suggested follow-ups for the latest answer
  const [followups, setFollowups] = useState<string[]>([]);

  const endRef = useRef<HTMLDivElement>(null);
  const prevTurns = useRef(0);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    api.stats().then((s) => setDocCount(s.document_count)).catch(() => setDocCount(0));
    api
      .listDocuments()
      .then((docs) => setDocuments(docs.filter((d) => d.status === "indexed")))
      .catch(() => setDocuments([]));
  }, []);

  function toggleDoc(id: string) {
    setSelectedDocs((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }
  useEffect(() => {
    // Only follow the conversation once it has started — never yank the page
    // down on first navigation to /chat (when there are no turns yet). Scroll
    // the sentinel into view without hijacking the whole window position.
    if (turns.length === 0) {
      prevTurns.current = 0;
      return;
    }
    const grew = turns.length > prevTurns.current;
    prevTurns.current = turns.length;
    endRef.current?.scrollIntoView({
      behavior: grew ? "smooth" : "auto",
      block: "nearest",
    });
  }, [turns]);

  // Shared streaming core for both new questions and regenerate.
  async function runStream(q: string, history: { role: string; content: string }[], replaceLast: boolean) {
    setError(null);
    setFollowups([]);
    if (replaceLast) {
      // Regenerate: reset the last assistant turn in place.
      setTurns((t) => {
        const next = [...t];
        next[next.length - 1] = { role: "assistant", content: "", streaming: true };
        return next;
      });
    } else {
      setTurns((t) => [
        ...t,
        { role: "user", content: q },
        { role: "assistant", content: "", streaming: true },
      ]);
    }
    setBusy(true);

    const body: Record<string, unknown> = {
      question: q,
      retrieval_mode: mode,
      top_k: topK,
      temperature,
      enable_query_expansion: expansion,
      enable_reranking: rerank,
      chat_history: history,
    };
    if (selectedDocs.size > 0) body.document_ids = Array.from(selectedDocs);

    const controller = new AbortController();
    abortRef.current = controller;
    let finalAnswer = "";

    await streamQuery(
      body,
      (token) =>
        setTurns((t) => {
          // Pure update (no mutation) — React 18 StrictMode invokes updaters
          // twice in dev, so mutating the existing turn would double each token.
          const next = [...t];
          const i = next.length - 1;
          next[i] = { ...next[i], content: next[i].content + token };
          finalAnswer = next[i].content;
          return next;
        }),
      (sources, meta) =>
        setTurns((t) => {
          const next = [...t];
          const i = next.length - 1;
          next[i] = {
            ...next[i],
            streaming: false,
            sources,
            confidence: (meta.confidence as Confidence) ?? "medium",
            grounding: (meta.grounding_score as number | null) ?? null,
            spans: (meta.sentence_support as SentenceSupport[]) ?? [],
            model: (meta.model as string) ?? "",
            tokens:
              ((meta.prompt_tokens as number) ?? 0) + ((meta.completion_tokens as number) ?? 0),
            cost: (meta.cost_usd as number) ?? 0,
          };
          return next;
        }),
      (msg) => {
        setError(msg);
        setTurns((t) => {
          const next = [...t];
          next[next.length - 1] = {
            role: "assistant",
            content:
              msg.includes("409") || msg.toLowerCase().includes("no documents")
                ? "No documents are indexed yet. Upload some on the Documents page first."
                : "Something went wrong reaching the backend.",
          };
          return next;
        });
      },
      controller.signal,
    );
    setBusy(false);
    abortRef.current = null;
    if (finalAnswer.trim()) {
      api
        .followups(q, finalAnswer)
        .then((r) => setFollowups(r.followups ?? []))
        .catch(() => {});
    }
  }

  function ask(question?: string) {
    const q = (question ?? input).trim();
    if (!q || busy) return;
    setInput("");
    // Prior turns become conversation context so follow-ups understand "it"/"that".
    const history = turns
      .filter((t) => t.content && !t.streaming)
      .map((t) => ({ role: t.role, content: t.content }));
    runStream(q, history, false);
  }

  function regenerate() {
    // Re-answer the most recent question, replacing the last answer in place.
    if (busy || turns.length < 2) return;
    const lastUser = turns[turns.length - 2];
    if (lastUser?.role !== "user") return;
    const history = turns
      .slice(0, turns.length - 2)
      .filter((t) => t.content && !t.streaming)
      .map((t) => ({ role: t.role, content: t.content }));
    runStream(lastUser.content, history, true);
  }

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
    setTurns((t) => {
      const next = [...t];
      const i = next.length - 1;
      if (i >= 0 && next[i].streaming) next[i] = { ...next[i], streaming: false };
      return next;
    });
  }

  function exportChat() {
    if (turns.length === 0) return;
    const body = turns
      .map((t) => {
        if (t.role === "user") return `**You:** ${t.content}`;
        const cites = (t.sources ?? [])
          .map((s) => `> [${s.filename} · p.${s.page_number}] ${s.text.slice(0, 160)}`)
          .join("\n");
        return `**DocMind:** ${t.content}${cites ? `\n\n${cites}` : ""}`;
      })
      .join("\n\n");
    const blob = new Blob([`# DocMind chat\n\n${body}\n`], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `docmind-chat-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="mx-auto grid max-w-7xl gap-6 px-5 py-6 lg:h-[calc(100vh-5.5rem)] lg:grid-cols-[280px_1fr]">
      {/* ── Controls sidebar ─────────────────────────────── */}
      <aside className="card h-fit p-5 lg:sticky lg:top-24 lg:max-h-[calc(100vh-7rem)] lg:overflow-y-auto chat-scroll">
        {/* ── Chat history ───────────────────────────────── */}
        <ChatHistory
          conversations={conversations}
          activeId={activeId}
          busy={busy}
          onNew={newChat}
          onSelect={selectChat}
          onDelete={deleteChat}
        />

        <div className="mt-6 grain-divider" />

        <p className="mt-6 eyebrow">Retrieval</p>
        <div className="mt-4 space-y-5">
          <div>
            <label className="mb-2 block text-sm text-ink-700">Mode</label>
            <div className="flex gap-1.5">
              {MODES.map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`flex-1 rounded-lg border px-2 py-1.5 font-mono text-xs capitalize transition-colors ${
                    mode === m ? "border-ink-900 bg-ink-900 text-paper" : "border-sand text-ink-700 hover:bg-cream"
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>

          <SliderRow label="Top-k" value={topK} min={1} max={50} onChange={setTopK} />
          <SliderRow
            label="Temperature"
            value={temperature}
            min={0}
            max={1}
            step={0.1}
            onChange={setTemperature}
          />

          <Toggle label="Query expansion" on={expansion} set={setExpansion} />
          <Toggle label="Reranking" on={rerank} set={setRerank} />
        </div>

        <div className="mt-6 grain-divider" />

        {/* ── Document scope ─────────────────────────────── */}
        <DocScope
          documents={documents}
          selected={selectedDocs}
          onToggle={toggleDoc}
          onClear={() => setSelectedDocs(new Set())}
        />

        <div className="mt-6 grain-divider" />
        <div className="mt-4 flex items-center justify-between">
          <span className="font-mono text-xs text-clay">
            {docCount === null ? "…" : `${docCount} docs indexed`}
          </span>
          <button
            onClick={exportChat}
            disabled={turns.length === 0}
            className="font-mono text-xs text-ink underline disabled:opacity-40"
          >
            Export chat
          </button>
        </div>
      </aside>

      {/* ── Conversation ─────────────────────────────────── */}
      <section className="flex h-[calc(100vh-12rem)] min-h-0 flex-col lg:h-full">
        <div className="chat-scroll flex-1 space-y-6 overflow-y-auto pr-2">
          {turns.length === 0 && (
            <motion.div
              className="card-cream grid items-center gap-4 overflow-hidden p-8 md:grid-cols-[1fr_240px]"
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, ease: [0.2, 0.7, 0.2, 1] }}
            >
              <div>
                <h1 className="font-display text-3xl text-ink-900">Ask your documents</h1>
                <p className="mt-2 max-w-lg text-ink-700">
                  Questions are answered strictly from indexed sources, with a
                  citation on every claim. Try “What are the key findings?” or
                  “Summarise the payment terms.”
                </p>
                <div className="mt-5 flex flex-wrap gap-2">
                  {["What are the key findings?", "Summarise the payment terms.", "List the main risks."].map(
                    (s) => (
                      <button
                        key={s}
                        onClick={() => ask(s)}
                        className="tag transition-colors hover:bg-paper"
                      >
                        {s}
                      </button>
                    ),
                  )}
                </div>
              </div>
              <div className="hidden h-[200px] md:block">
                <Scene3D name="ring" className="h-full w-full" />
              </div>
            </motion.div>
          )}

          {turns.map((turn, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              className={turn.role === "user" ? "flex justify-end" : ""}
            >
              {turn.role === "user" ? (
                <div className="max-w-[80%] rounded-xl2 rounded-br-md border border-ink-900 bg-ink-900 px-5 py-3 text-paper">
                  {turn.content}
                </div>
              ) : (
                <div className="max-w-[88%]">
                  <div className="card px-5 py-4">
                    {turn.streaming && !turn.content ? (
                      <ThinkingIndicator />
                    ) : turn.streaming ? (
                      <p className="whitespace-pre-wrap leading-relaxed text-ink-900">
                        {turn.content}
                        <span className="ml-0.5 inline-block h-4 w-2 animate-pulse bg-ink align-middle" />
                      </p>
                    ) : (
                      <GroundedAnswer
                        text={turn.content}
                        spans={turn.spans}
                        sources={turn.sources ?? []}
                      />
                    )}
                  </div>
                  {!turn.streaming && turn.content && (
                    <AnswerMeta
                      turn={turn}
                      onRegenerate={i === turns.length - 1 && !busy ? regenerate : undefined}
                    />
                  )}
                </div>
              )}
            </motion.div>
          ))}
          <div ref={endRef} />
        </div>

        {/* Composer — fixed below the scrollable conversation */}
        <div className="shrink-0 border-t border-sand pt-4">
          {/* Suggested follow-ups */}
          {!busy && followups.length > 0 && (
            <div className="mb-3 flex flex-wrap gap-2">
              {followups.map((f) => (
                <button
                  key={f}
                  onClick={() => ask(f)}
                  className="tag transition-colors hover:bg-paper"
                >
                  {f}
                </button>
              ))}
            </div>
          )}
          {error && (
            <p className="mb-3 rounded-lg border border-sand bg-cream px-3 py-2 font-mono text-xs text-ink-900">
              {error}
            </p>
          )}
          <div className="flex items-end gap-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  ask();
                }
              }}
              rows={1}
              placeholder="Ask a question about your documents…"
              className="input min-h-[52px] flex-1 resize-none"
            />
            {busy ? (
              <button
                onClick={stop}
                className="flex h-[52px] shrink-0 items-center gap-2 rounded-full border border-ink-900 px-6 font-medium text-ink-900 transition-colors hover:bg-cream"
              >
                <span className="h-3 w-3 rounded-sm bg-ink-900" />
                Stop
              </button>
            ) : (
              <CircleExpandButton onClick={() => ask()} className="h-[52px] shrink-0">
                Send
              </CircleExpandButton>
            )}
          </div>
        </div>
      </section>
    </div>
  );
}

function ChatHistory({
  conversations,
  activeId,
  busy,
  onNew,
  onSelect,
  onDelete,
}: {
  conversations: Conversation[];
  activeId: string;
  busy: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between">
        <p className="eyebrow">Chats</p>
        <button
          onClick={onNew}
          disabled={busy}
          className="flex items-center gap-1 font-mono text-xs text-ink underline disabled:opacity-40"
        >
          + New
        </button>
      </div>

      <div className="chat-scroll mt-3 max-h-44 space-y-1 overflow-y-auto pr-1">
        {conversations.map((c) => {
          const active = c.id === activeId;
          return (
            <div
              key={c.id}
              className={`group flex items-center gap-1.5 rounded-lg border px-2.5 py-2 transition-colors ${
                active ? "border-ink-900 bg-cream" : "border-transparent hover:bg-cream/60"
              }`}
            >
              <button
                onClick={() => onSelect(c.id)}
                disabled={busy}
                className="min-w-0 flex-1 truncate text-left text-sm text-ink-900 disabled:opacity-60"
                title={c.title}
              >
                {c.title}
              </button>
              <button
                onClick={() => onDelete(c.id)}
                disabled={busy}
                aria-label="Delete chat"
                className="shrink-0 text-clay opacity-0 transition-opacity hover:text-ink-900 group-hover:opacity-100 disabled:opacity-0"
              >
                <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="none" stroke="currentColor" strokeWidth={2}>
                  <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" />
                </svg>
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DocScope({
  documents,
  selected,
  onToggle,
  onClear,
}: {
  documents: DocumentRecord[];
  selected: Set<string>;
  onToggle: (id: string) => void;
  onClear: () => void;
}) {
  const all = selected.size === 0;
  return (
    <div className="mt-4">
      <div className="mb-2 flex items-center justify-between">
        <label className="text-sm text-ink-700">Sources</label>
        <span className="font-mono text-xs text-clay">
          {all ? "All documents" : `${selected.size} selected`}
        </span>
      </div>

      {documents.length === 0 ? (
        <p className="rounded-lg border border-sand bg-cream/50 px-3 py-2 font-mono text-xs text-clay">
          No indexed documents yet.
        </p>
      ) : (
        <div className="max-h-52 space-y-1 overflow-y-auto pr-1">
          {/* All documents (clears the selection) */}
          <button
            onClick={onClear}
            className={`flex w-full items-center gap-2.5 rounded-lg border px-2.5 py-2 text-left transition-colors ${
              all ? "border-ink-900 bg-cream" : "border-sand hover:bg-cream/60"
            }`}
          >
            <Check on={all} />
            <span className="truncate text-sm text-ink-900">All documents</span>
          </button>

          {documents.map((d) => {
            const on = selected.has(d.id);
            return (
              <button
                key={d.id}
                onClick={() => onToggle(d.id)}
                title={d.filename}
                className={`flex w-full items-center gap-2.5 rounded-lg border px-2.5 py-2 text-left transition-colors ${
                  on ? "border-ink-900 bg-cream" : "border-sand hover:bg-cream/60"
                }`}
              >
                <Check on={on} />
                <span className="min-w-0 flex-1 truncate text-sm text-ink-900">{d.filename}</span>
                <span className="shrink-0 font-mono text-[10px] text-clay">{d.chunk_count}c</span>
              </button>
            );
          })}
        </div>
      )}

      {!all && (
        <button onClick={onClear} className="mt-2 font-mono text-xs text-ink underline">
          Reset to all
        </button>
      )}
    </div>
  );
}

function Check({ on }: { on: boolean }) {
  return (
    <span
      className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border transition-colors ${
        on ? "border-ink-900 bg-ink-900 text-paper" : "border-sand bg-paper"
      }`}
    >
      {on && (
        <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth={3}>
          <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      )}
    </span>
  );
}

function AnswerMeta({ turn, onRegenerate }: { turn: Turn; onRegenerate?: () => void }) {
  const [copied, setCopied] = useState(false);
  const [vote, setVote] = useState<"up" | "down" | null>(null);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const sources = turn.sources ?? [];

  function sendFeedback(helpful: boolean) {
    setVote(helpful ? "up" : "down");
    // One feedback id per answer instance — enough for the analytics counters.
    api.feedback(`chat-${Date.now()}`, helpful).catch(() => {});
  }

  return (
    <div className="mt-3 space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {turn.confidence && <ConfidenceBadge level={turn.confidence} />}
        {typeof turn.grounding === "number" && <GroundingBadge score={turn.grounding} />}
        <button
          onClick={() => {
            navigator.clipboard.writeText(turn.content);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
          className="tag hover:bg-paper"
        >
          {copied ? "Copied" : "Copy"}
        </button>
        {onRegenerate && (
          <button onClick={onRegenerate} className="tag hover:bg-paper">
            ↻ Regenerate
          </button>
        )}
        <button onClick={() => sendFeedback(true)} className={`tag ${vote === "up" ? "bg-paper" : ""}`}>
          ↑ Helpful
        </button>
        <button onClick={() => sendFeedback(false)} className={`tag ${vote === "down" ? "bg-paper" : ""}`}>
          ↓
        </button>
        {(turn.model || turn.tokens) && (
          <span className="font-mono text-[10px] text-clay">
            {turn.model}
            {turn.tokens ? ` · ~${turn.tokens} tok` : ""}
            {turn.cost ? ` · $${turn.cost.toFixed(5)}` : ""}
          </span>
        )}
      </div>
      {sources.length > 0 && (
        <div>
          {/* Single dropdown for the whole citation list — click to reveal all. */}
          <button
            onClick={() => setSourcesOpen((v) => !v)}
            className="flex w-full items-center justify-between rounded-lg border border-sand bg-cream/60 px-4 py-2.5 text-left transition-colors hover:bg-cream"
          >
            <span className="eyebrow">Sources ({sources.length})</span>
            <motion.span animate={{ rotate: sourcesOpen ? 180 : 0 }} className="text-ink">
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2}>
                <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </motion.span>
          </button>
          <AnimatePresence initial={false}>
            {sourcesOpen && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                <div className="space-y-2 pt-2">
                  {sources.map((s, i) => (
                    <SourceCard key={i} source={s} index={i} answer={turn.content} />
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

function SliderRow({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-sm text-ink-700">
        <span>{label}</span>
        <span className="font-mono text-xs text-ink-900">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-sand accent-ink-900"
      />
    </div>
  );
}

function Toggle({ label, on, set }: { label: string; on: boolean; set: (v: boolean) => void }) {
  return (
    <button onClick={() => set(!on)} className="flex w-full items-center justify-between">
      <span className="text-sm text-ink-700">{label}</span>
      <span className={`relative h-6 w-11 rounded-full border transition-colors ${on ? "border-ink-900 bg-ink-900" : "border-sand bg-paper"}`}>
        <motion.span
          layout
          className="absolute top-0.5 h-4 w-4 rounded-full bg-paper"
          animate={{ left: on ? 24 : 4 }}
          style={{ left: on ? 24 : 4 }}
        />
      </span>
    </button>
  );
}
