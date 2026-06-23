"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api } from "@/lib/api";
import type { DriftStatus, EvaluationResult } from "@/lib/types";
import { Reveal } from "@/components/framer/Reveal";
import { CircleExpandButton } from "@/components/framer/CircleExpandButton";

const METRICS: { key: keyof EvaluationResult; label: string; hint: string }[] = [
  { key: "faithfulness", label: "Faithfulness", hint: "Claims supported by context" },
  { key: "answer_relevancy", label: "Answer relevancy", hint: "Answer addresses the question" },
  { key: "context_precision", label: "Context precision", hint: "Retrieved chunks are relevant" },
  { key: "context_recall", label: "Context recall", hint: "Context covers the ground truth" },
];

type Analytics = Awaited<ReturnType<typeof api.analytics>>;

const ENGINE_LABEL: Record<string, string> = {
  ragas: "Scored by RAGAS",
  lexical: "Lexical heuristics (approximate)",
};

function engineLabel(engine?: string): string {
  if (!engine) return "";
  if (engine.startsWith("llm-judge")) return `LLM judge · ${engine.split(":")[1] ?? ""}`;
  return ENGINE_LABEL[engine] ?? engine;
}

export default function EvaluationPage() {
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [drift, setDrift] = useState<DriftStatus | null>(null);
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [running, setRunning] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    api.evaluationResults().then(setResult).catch(() => {});
    api.drift().then(setDrift).catch(() => {});
    api.analytics().then(setAnalytics).catch(() => {});
  }, []);

  async function run() {
    setRunning(true);
    setNote("Running the full pipeline over a small test set — this can take a minute or two on a local model.");
    try {
      const r = await api.runEvaluation(3);
      setResult(r);
      setNote(null);
    } catch {
      setNote("Evaluation failed — make sure documents are indexed and the backend is running.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="mx-auto max-w-7xl px-5 py-10">
      <Reveal>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow">Quality</p>
            <h1 className="mt-3 font-display text-4xl text-ink-900">Evaluation</h1>
            <p className="mt-2 max-w-xl text-ink-700">
              RAGAS scores the pipeline on a synthetic test set — faithfulness,
              relevancy, and how well retrieval covers the answer.
            </p>
          </div>
          <CircleExpandButton onClick={run}>
            {running ? "Running…" : "Run evaluation"}
          </CircleExpandButton>
        </div>
      </Reveal>

      {note && (
        <p className="mt-4 rounded-lg border border-sand bg-cream px-3 py-2 font-mono text-xs text-ink-900">
          {note}
        </p>
      )}

      {result && (
        <p className="mt-4 font-mono text-xs text-clay">
          {engineLabel(result.engine)} · {result.question_count} questions
        </p>
      )}

      {/* Metric cards */}
      <div className="mt-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {METRICS.map((m, i) => {
          const value = result ? (result[m.key] as number) : null;
          return (
            <Reveal key={m.key} delay={i * 0.05}>
              <div className="card p-6">
                <p className="text-sm text-ink-700">{m.label}</p>
                <div className="mt-3 flex items-end gap-2">
                  <span className="font-display text-4xl text-ink-900">
                    {value === null ? "—" : (value * 100).toFixed(0)}
                  </span>
                  <span className="mb-1 font-mono text-xs text-clay">/ 100</span>
                </div>
                {/* flat bar gauge */}
                <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-sand">
                  <motion.div
                    className="h-full rounded-full bg-ink-900"
                    initial={{ width: 0 }}
                    animate={{ width: value === null ? 0 : `${value * 100}%` }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                  />
                </div>
                <p className="mt-3 text-xs text-ink-700">{m.hint}</p>
              </div>
            </Reveal>
          );
        })}
      </div>

      {/* Retrieval metrics + drift */}
      <div className="mt-6 grid gap-5 md:grid-cols-2">
        <Reveal>
          <div className="card-cream p-6">
            <p className="eyebrow">Retrieval metrics</p>
            <div className="mt-4 grid grid-cols-3 gap-4">
              {[
                { l: "Recall@k", v: result?.recall_at_k },
                { l: "NDCG", v: result?.ndcg },
                { l: "MRR", v: result?.mrr },
              ].map((x) => (
                <div key={x.l}>
                  <p className="font-display text-2xl text-ink-900">
                    {x.v === undefined ? "—" : x.v.toFixed(2)}
                  </p>
                  <p className="font-mono text-xs text-ink-700">{x.l}</p>
                </div>
              ))}
            </div>
          </div>
        </Reveal>
        <Reveal delay={0.05}>
          <div className="card p-6">
            <p className="eyebrow">Embedding drift</p>
            <div className="mt-4 flex items-center gap-3">
              <span
                className={`h-3 w-3 rounded-full ${drift?.drift_detected ? "bg-ink-700" : "bg-ink-900"}`}
              />
              <span className="font-display text-xl text-ink-900">
                {drift ? drift.message : "—"}
              </span>
            </div>
            <p className="mt-3 text-sm text-ink-700">
              {drift
                ? `Score ${drift.drift_score.toFixed(3)} vs threshold ${drift.threshold}`
                : "Run a check once documents are indexed."}
            </p>
          </div>
        </Reveal>
      </div>

      {/* Usage analytics */}
      <Reveal>
        <div className="mt-6 card p-6">
          <p className="eyebrow">Usage analytics</p>
          <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
            {[
              { l: "Queries", v: analytics ? String(analytics.query_count) : "—" },
              { l: "Avg latency", v: analytics ? `${Math.round(analytics.avg_latency_ms)} ms` : "—" },
              { l: "Total tokens", v: analytics ? analytics.total_tokens.toLocaleString() : "—" },
              {
                l: "Helpful",
                v:
                  analytics && analytics.feedback_count > 0
                    ? `${Math.round(analytics.helpful_ratio * 100)}%`
                    : "—",
              },
            ].map((x) => (
              <div key={x.l}>
                <p className="font-display text-2xl text-ink-900">{x.v}</p>
                <p className="font-mono text-xs text-ink-700">{x.l}</p>
              </div>
            ))}
          </div>
          {analytics && analytics.recent.length > 0 && (
            <div className="mt-5 space-y-1.5">
              <p className="font-mono text-xs uppercase tracking-wider text-clay">Recent queries</p>
              {analytics.recent.slice(0, 6).map((r, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between gap-3 border-t border-sand pt-1.5 text-sm"
                >
                  <span className="min-w-0 flex-1 truncate text-ink-900">{r.answer || "—"}</span>
                  <span className="shrink-0 font-mono text-xs text-clay">
                    {r.confidence} · {Math.round(r.latency_ms)}ms
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </Reveal>
    </div>
  );
}
