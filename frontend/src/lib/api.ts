// Typed client for the DocMind FastAPI backend.

import type {
  AppSettings,
  DocumentRecord,
  DriftStatus,
  EvaluationResult,
  QueryResponse,
  SourceChunk,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export const api = {
  health: () => http<{ status: string }>("/api/v1/health"),

  // Documents
  listDocuments: () => http<DocumentRecord[]>("/api/v1/documents"),
  deleteDocument: (id: string) =>
    http<{ status: string }>(`/api/v1/documents/${id}`, { method: "DELETE" }),
  documentChunks: (id: string) =>
    http<
      { chunk_id: string; page_number: number; char_count: number; text: string }[]
    >(`/api/v1/documents/${id}/chunks`),
  async upload(
    files: FileList | File[],
  ): Promise<{ documents: DocumentRecord[]; job_ids: string[] }> {
    const form = new FormData();
    Array.from(files).forEach((f) => form.append("files", f));
    const res = await fetch(`${API_BASE}/api/v1/documents/upload`, {
      method: "POST",
      body: form,
    });
    if (!res.ok) throw new ApiError(res.status, "Upload failed");
    return res.json();
  },
  jobStatus: (id: string) =>
    http<{
      id: string;
      filename: string;
      status: "queued" | "processing" | "completed" | "failed";
      progress: number;
      error?: string | null;
    }>(`/api/v1/documents/jobs/${id}`),

  // Query
  query: (body: Record<string, unknown>) =>
    http<QueryResponse>("/api/v1/query", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  followups: (question: string, answer: string) =>
    http<{ followups: string[] }>("/api/v1/query/followups", {
      method: "POST",
      body: JSON.stringify({ question, answer }),
    }),
  feedback: (query_id: string, helpful: boolean, comment?: string) =>
    http<{ status: string }>("/api/v1/feedback", {
      method: "POST",
      body: JSON.stringify({ query_id, helpful, comment }),
    }),
  analytics: () =>
    http<{
      query_count: number;
      avg_latency_ms: number;
      total_cost_usd: number;
      total_tokens: number;
      avg_confidence_score: number;
      feedback_count: number;
      helpful_count: number;
      helpful_ratio: number;
      recent: {
        answer: string;
        confidence: string;
        latency_ms: number;
        cost_usd: number;
        model: string;
        timestamp: string;
      }[];
    }>("/api/v1/admin/analytics"),

  // Settings
  getSettings: () => http<AppSettings>("/api/v1/settings"),
  patchSettings: (patch: Partial<AppSettings>) =>
    http<AppSettings>("/api/v1/settings", {
      method: "PATCH",
      body: JSON.stringify(patch),
    }),

  // Evaluation
  runEvaluation: (test_size: number) =>
    http<EvaluationResult>("/api/v1/evaluate", {
      method: "POST",
      body: JSON.stringify({ test_size }),
    }),
  evaluationResults: () => http<EvaluationResult>("/api/v1/evaluate/results"),
  drift: () => http<DriftStatus>("/api/v1/evaluate/drift"),

  stats: () =>
    http<{ document_count: number; chunk_count: number; query_count: number }>(
      "/api/v1/admin/stats",
    ),
};

/**
 * Stream an answer over SSE. Calls `onToken` for each token, then `onDone`
 * with the collected sources + metadata.
 */
export async function streamQuery(
  body: Record<string, unknown>,
  onToken: (t: string) => void,
  onDone: (sources: SourceChunk[], meta: Record<string, unknown>) => void,
  onError?: (msg: string) => void,
  signal?: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/v1/query/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    // A user-initiated stop aborts the fetch — not an error to surface.
    if ((e as Error)?.name !== "AbortError") onError?.("Request failed");
    return;
  }
  if (!res.ok || !res.body) {
    onError?.(`Request failed (${res.status})`);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch {
      break; // aborted mid-stream — keep whatever tokens already arrived
    }
    const { done, value } = chunk;
    if (done) break;
    // Normalise CRLF → LF so frame splitting works regardless of the server's
    // line endings (sse-starlette emits \r\n\r\n between events).
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      // A frame may carry multiple "data:" lines; concatenate them per the SSE spec.
      let event: string | undefined;
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length === 0) continue;
      try {
        const data = JSON.parse(dataLines.join("\n"));
        if (event === "token") onToken(data.token);
        else if (event === "done") onDone(data.sources ?? [], data.meta ?? {});
        else if (event === "error") onError?.(data.error);
      } catch {
        /* skip malformed frame */
      }
    }
  }
}
