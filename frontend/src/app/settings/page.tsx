"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { AppSettings } from "@/lib/types";
import { Reveal } from "@/components/framer/Reveal";

export default function SettingsPage() {
  const [s, setS] = useState<AppSettings | null>(null);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getSettings().then(setS).catch(() => setError("Backend unreachable."));
  }, []);

  async function patch(p: Partial<AppSettings>) {
    if (!s) return;
    setS({ ...s, ...p });
    try {
      const updated = await api.patchSettings(p);
      setS(updated);
      setSaved(true);
      setTimeout(() => setSaved(false), 1200);
    } catch {
      setError("Could not save.");
    }
  }

  if (error) {
    return (
      <div className="mx-auto max-w-3xl px-5 py-20">
        <p className="rounded-lg border border-sand bg-cream px-3 py-2 font-mono text-xs">{error}</p>
      </div>
    );
  }
  if (!s) return <div className="mx-auto max-w-3xl px-5 py-20 text-ink-700">Loading…</div>;

  return (
    <div className="mx-auto max-w-3xl px-5 py-10">
      <Reveal>
        <p className="eyebrow">Configuration</p>
        <h1 className="mt-3 font-display text-4xl text-ink-900">Settings</h1>
        <p className="mt-2 text-ink-700">Changes apply immediately to subsequent queries.</p>
      </Reveal>

      <div className="mt-8 space-y-5">
        <Section title="Models">
          <Field label="LLM provider">
            <Select value={s.llm_provider} options={["groq", "ollama", "openai", "anthropic"]} onChange={(v) => patch({ llm_provider: v })} />
          </Field>
          <Field label="LLM model">
            <ModelPicker provider={s.llm_provider} value={s.llm_model} onChange={(v) => patch({ llm_model: v })} />
          </Field>
          <Field label="Embedding model">
            <Select
              value={s.embedding_model}
              options={["all-MiniLM-L6-v2", "BAAI/bge-large-en-v1.5", "text-embedding-3-small"]}
              onChange={(v) => patch({ embedding_model: v })}
            />
          </Field>
        </Section>

        <Section title="Retrieval">
          <Field label="Mode">
            <Select value={s.retrieval_mode} options={["hybrid", "similarity", "mmr"]} onChange={(v) => patch({ retrieval_mode: v })} />
          </Field>
          <Field label="Chunk strategy">
            <Select value={s.chunk_strategy} options={["recursive", "semantic", "parent"]} onChange={(v) => patch({ chunk_strategy: v })} />
          </Field>
          <NumberField label="Top-k" value={s.retrieval_k} min={1} max={50} onChange={(v) => patch({ retrieval_k: v })} />
          <NumberField label="Reranker top-k" value={s.reranker_top_k} min={1} max={20} onChange={(v) => patch({ reranker_top_k: v })} />
          <NumberField label="Chunk size" value={s.chunk_size} min={200} max={4000} step={100} onChange={(v) => patch({ chunk_size: v })} />
        </Section>

        <Section title="Pipeline toggles">
          {([
            ["enable_query_expansion", "Query expansion"],
            ["enable_hyde", "HyDE"],
            ["enable_reranking", "Reranking"],
            ["enable_context_compression", "Context compression"],
            ["enable_crag", "Corrective RAG"],
            ["enable_self_rag", "Self-RAG"],
          ] as [keyof AppSettings, string][]).map(([key, label]) => (
            <ToggleRow key={key} label={label} on={s[key] as boolean} set={(v) => patch({ [key]: v } as Partial<AppSettings>)} />
          ))}
        </Section>
      </div>

      <div className="mt-6 font-mono text-xs text-clay">{saved ? "Saved ✓" : " "}</div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="card p-6">
      <p className="eyebrow mb-4">{title}</p>
      <div className="space-y-4">{children}</div>
    </div>
  );
}
function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5 sm:flex-row sm:items-center sm:justify-between">
      <span className="text-sm text-ink-700">{label}</span>
      <div className="sm:w-64">{children}</div>
    </label>
  );
}
const MODELS_BY_PROVIDER: Record<string, string[]> = {
  groq: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it"],
  openai: ["gpt-4o", "gpt-4o-mini"],
  anthropic: ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"],
  ollama: ["llama3", "llama3.2:1b", "mistral", "phi3", "gemma2"],
};

/** Curated model dropdown per provider, but still accepts a custom value typed
 *  into the box (via the "Custom…" option revealing a text input). */
function ModelPicker({ provider, value, onChange }: { provider: string; value: string; onChange: (v: string) => void }) {
  const preset = MODELS_BY_PROVIDER[provider] ?? [];
  const options = preset.includes(value) || !value ? preset : [value, ...preset];
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="input cursor-pointer">
      {options.map((o) => (
        <option key={o} value={o}>{o}</option>
      ))}
    </select>
  );
}
function Select({ value, options, onChange }: { value: string; options: string[]; onChange: (v: string) => void }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="input cursor-pointer">
      {options.map((o) => (
        <option key={o} value={o}>{o}</option>
      ))}
    </select>
  );
}
function NumberField({ label, value, min, max, step = 1, onChange }: { label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void }) {
  return (
    <Field label={label}>
      <input type="number" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="input" />
    </Field>
  );
}
function ToggleRow({ label, on, set }: { label: string; on: boolean; set: (v: boolean) => void }) {
  return (
    <button onClick={() => set(!on)} className="flex w-full items-center justify-between">
      <span className="text-sm text-ink-700">{label}</span>
      <span className={`relative h-6 w-11 rounded-full border transition-colors ${on ? "border-ink-900 bg-ink-900" : "border-sand bg-paper"}`}>
        <span className="absolute top-0.5 h-4 w-4 rounded-full bg-paper transition-all" style={{ left: on ? 24 : 4 }} />
      </span>
    </button>
  );
}
