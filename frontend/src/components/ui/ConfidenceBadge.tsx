import type { Confidence } from "@/lib/types";

const STYLES: Record<Confidence, { dot: string; label: string }> = {
  high: { dot: "bg-ink-900", label: "High confidence" },
  medium: { dot: "bg-clay", label: "Medium confidence" },
  low: { dot: "bg-sand", label: "Low confidence" },
};

export function ConfidenceBadge({ level }: { level: Confidence }) {
  const s = STYLES[level];
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-sand bg-paper px-3 py-1 font-mono text-xs text-ink-700">
      <span className={`h-2.5 w-2.5 rounded-full ${s.dot}`} />
      {s.label}
    </span>
  );
}
