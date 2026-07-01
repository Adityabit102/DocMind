"use client";

import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api } from "@/lib/api";
import type { DocumentRecord } from "@/lib/types";
import { Reveal } from "@/components/framer/Reveal";

const STATUS_DOT: Record<string, string> = {
  indexed: "bg-ink-900",
  processing: "bg-clay",
  queued: "bg-sand",
  failed: "bg-ink-700",
};

interface JobProgress {
  id: string;
  filename: string;
  status: string;
  progress: number;
}

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentRecord[]>([]);
  const [drag, setDrag] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [jobs, setJobs] = useState<JobProgress[]>([]);
  const [query, setQuery] = useState("");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.listDocuments().then(setDocs).catch(() => setError("Backend unreachable. Is the API running on :8000?"));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleFiles(files: FileList | File[]) {
    setUploading(true);
    setError(null);
    try {
      const { job_ids } = await api.upload(files);
      const names = Array.from(files).map((f) => f.name);
      setJobs(
        job_ids.map((id, i) => ({ id, filename: names[i] ?? "file", status: "queued", progress: 0 })),
      );
      // Poll each ingestion job until it finishes, surfacing real progress.
      const pending = new Set(job_ids);
      while (pending.size > 0) {
        await new Promise((r) => setTimeout(r, 900));
        await Promise.all(
          Array.from(pending).map(async (id) => {
            try {
              const j = await api.jobStatus(id);
              setJobs((prev) => prev.map((p) => (p.id === id ? { ...p, status: j.status, progress: j.progress } : p)));
              if (j.status === "completed" || j.status === "failed") {
                pending.delete(id);
                refresh();
              }
            } catch {
              pending.delete(id);
            }
          }),
        );
      }
      refresh();
      // Clear the progress list shortly after completion.
      setTimeout(() => setJobs([]), 1500);
    } catch {
      setError("Upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function remove(id: string) {
    // Optimistic: drop it from the table immediately, then reconcile with the
    // server (the backend rebuilds its index on delete, which can take a moment).
    setDocs((prev) => prev.filter((d) => d.id !== id));
    try {
      await api.deleteDocument(id);
    } catch {
      setError("Delete failed — restoring.");
    }
    refresh();
  }

  const filtered = docs.filter((d) => d.filename.toLowerCase().includes(query.toLowerCase()));
  const totalChunks = docs.reduce((s, d) => s + d.chunk_count, 0);

  return (
    <div className="mx-auto max-w-7xl px-5 py-10">
      <Reveal>
        <p className="eyebrow">Knowledge base</p>
        <h1 className="mt-3 font-display text-4xl text-ink-900">Documents</h1>
        <p className="mt-2 max-w-xl text-ink-700">
          Drop files to index them. Each is chunked, embedded, and added to the
          hybrid FAISS + BM25 index. Supported: PDF, DOCX, MD, TXT, CSV, HTML.
        </p>
      </Reveal>

      {/* Dropzone */}
      <motion.label
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
        }}
        animate={{ scale: drag ? 1.01 : 1 }}
        className={`mt-8 flex cursor-pointer flex-col items-center justify-center rounded-xl2 border-2 border-dashed px-6 py-14 text-center transition-colors ${
          drag ? "border-ink-900 bg-cream" : "border-sand bg-paper"
        }`}
      >
        <input
          type="file"
          multiple
          className="hidden"
          accept=".pdf,.docx,.txt,.md,.csv,.html,.htm"
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
        />
        <span className="flex h-12 w-12 items-center justify-center rounded-xl2 border border-ink bg-cream">
          <svg viewBox="0 0 24 24" className="h-6 w-6 text-ink-900" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M12 16V4M6 10l6-6 6 6M4 20h16" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
        <p className="mt-4 font-display text-xl text-ink-900">
          {uploading ? "Indexing…" : "Drop documents here"}
        </p>
        <p className="mt-1 text-sm text-ink-700">or click to browse</p>
      </motion.label>

      {error && (
        <p className="mt-4 rounded-lg border border-sand bg-cream px-3 py-2 font-mono text-xs text-ink-900">
          {error}
        </p>
      )}

      {/* Per-file ingestion progress (durable job queue) */}
      {jobs.length > 0 && (
        <div className="mt-4 space-y-2">
          {jobs.map((j) => (
            <div key={j.id} className="rounded-lg border border-sand bg-paper px-4 py-2.5">
              <div className="flex items-center justify-between text-sm">
                <span className="min-w-0 flex-1 truncate text-ink-900">{j.filename}</span>
                <span className="ml-3 shrink-0 font-mono text-xs text-clay">{j.status}</span>
              </div>
              <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-sand">
                <motion.div
                  className={`h-full rounded-full ${j.status === "failed" ? "bg-ink-700" : "bg-ink-900"}`}
                  animate={{ width: `${Math.round((j.status === "completed" ? 1 : j.progress) * 100)}%` }}
                  transition={{ duration: 0.4 }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Stats + search */}
      <div className="mt-10 flex flex-wrap items-center justify-between gap-4">
        <div className="flex gap-3">
          <span className="tag">{docs.length} documents</span>
          <span className="tag">{totalChunks} chunks</span>
        </div>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search documents…"
          className="input max-w-xs"
        />
      </div>

      {/* Table (scrolls horizontally on narrow screens) */}
      <div className="mt-4 overflow-x-auto rounded-xl2 border border-sand">
        <table className="w-full min-w-[640px] text-left text-sm">
          <thead className="bg-cream font-mono text-xs uppercase tracking-wider text-ink-700">
            <tr>
              <th className="px-4 py-3">Name</th>
              <th className="px-4 py-3">Type</th>
              <th className="px-4 py-3">Pages</th>
              <th className="px-4 py-3">Chunks</th>
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3" />
            </tr>
          </thead>
          <tbody>
            {filtered.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-10 text-center text-ink-700">
                  No documents yet.
                </td>
              </tr>
            )}
            {filtered.map((d) => (
              <tr key={d.id} className="border-t border-sand bg-paper hover:bg-cream/40">
                <td className="px-4 py-3 font-medium text-ink-900">{d.filename}</td>
                <td className="px-4 py-3 uppercase text-ink-700">{d.file_type}</td>
                <td className="px-4 py-3 text-ink-700">{d.page_count}</td>
                <td className="px-4 py-3 text-ink-700">{d.chunk_count}</td>
                <td className="px-4 py-3">
                  <span className="inline-flex items-center gap-2 font-mono text-xs text-ink-700">
                    <span className={`h-2 w-2 rounded-full ${STATUS_DOT[d.status] ?? "bg-sand"}`} />
                    {d.status}
                  </span>
                </td>
                <td className="px-4 py-3 text-right">
                  <button onClick={() => remove(d.id)} className="font-mono text-xs text-ink underline">
                    delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
