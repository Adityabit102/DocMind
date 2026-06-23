import Link from "next/link";

export function Footer() {
  return (
    <footer className="mt-24 border-t border-sand">
      <div className="mx-auto grid max-w-7xl gap-8 px-5 py-12 md:grid-cols-3">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-ink-900 bg-ink-900 font-display text-paper">
              D
            </span>
            <span className="font-display text-xl font-semibold text-ink-900">DocMind</span>
          </div>
          <p className="mt-3 max-w-xs text-sm text-ink-700">
            Grounded, cited answers from your documents. Hybrid retrieval,
            reranking, and traceable sources — no hallucinations by construction.
          </p>
        </div>
        <div className="text-sm">
          <p className="eyebrow mb-3">Product</p>
          <ul className="space-y-2 text-ink-700">
            <li><Link href="/chat" className="hover:text-ink-900">Chat</Link></li>
            <li><Link href="/documents" className="hover:text-ink-900">Documents</Link></li>
            <li><Link href="/evaluation" className="hover:text-ink-900">Evaluation</Link></li>
          </ul>
        </div>
        <div className="text-sm">
          <p className="eyebrow mb-3">Built with</p>
          <ul className="space-y-2 text-ink-700">
            <li>FastAPI · LangChain · FAISS</li>
            <li>Next.js · React Three Fiber</li>
            <li>RAGAS · Prometheus</li>
          </ul>
        </div>
      </div>
      <div className="border-t border-sand py-5 text-center font-mono text-xs text-clay">
        DocMind — RAG Document Q&amp;A · {new Date().getFullYear()}
      </div>
    </footer>
  );
}
