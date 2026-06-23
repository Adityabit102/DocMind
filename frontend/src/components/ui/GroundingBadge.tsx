"use client";

/** Shows how well the answer is grounded in its cited sources (0–100%). */
export function GroundingBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const tone =
    pct >= 80
      ? "border-ink-900 text-ink-900"
      : pct >= 50
        ? "border-sand text-ink-700"
        : "border-amber-600/70 text-amber-700";
  return (
    <span
      className={`tag border ${tone}`}
      title="Share of answer sentences supported by the retrieved sources"
    >
      Grounded {pct}%
    </span>
  );
}
