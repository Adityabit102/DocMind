import Link from "next/link";
import { Reveal } from "@/components/framer/Reveal";
import { TextArrowCTA } from "@/components/framer/TextArrowCTA";
import { Scene3D } from "@/components/three/Scene3D";

const STACK = [
  ["Frontend", "Next.js · React Three Fiber · Framer Motion"],
  ["API", "FastAPI · Uvicorn · SSE streaming"],
  ["RAG", "LangChain (LCEL) · FAISS · BM25 · RRF"],
  ["Rerank", "cross-encoder/ms-marco-MiniLM-L-6-v2"],
  ["LLM", "Ollama · OpenAI · Anthropic (local-first)"],
  ["Eval", "RAGAS · Evidently drift · Prometheus"],
];

const PIPELINE = [
  ["Load & chunk", "Documents are parsed (OCR fallback for scans) and split with overlap, stamping page + offset for citations."],
  ["Index", "Chunks are embedded into FAISS and mirrored in a BM25 keyword index."],
  ["Retrieve & fuse", "Dense and sparse hits are merged with reciprocal rank fusion."],
  ["Rerank & compress", "A cross-encoder reorders candidates; optional compression trims noise."],
  ["Generate", "The LLM answers strictly from context, citing each source — or admits it doesn't know."],
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-7xl px-5 py-12">
      <section className="grid items-center gap-10 md:grid-cols-2">
        <Reveal>
          <p className="eyebrow">About</p>
          <h1 className="mt-3 font-display text-5xl text-ink-900">DocMind</h1>
          <p className="mt-4 max-w-md text-lg text-ink-700">
            A production-grade Retrieval-Augmented Generation system that turns a
            pile of documents into a source you can interrogate — and verify.
            Built end-to-end: ingestion, hybrid retrieval, reranking, grounded
            generation, evaluation, and observability.
          </p>
          <div className="mt-6 flex flex-wrap gap-4">
            <TextArrowCTA href="/chat">Try the chat</TextArrowCTA>
            <TextArrowCTA href="http://localhost:8000/docs">API docs</TextArrowCTA>
          </div>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="card relative h-[320px] overflow-hidden">
            <Scene3D name="hero" className="absolute inset-0" />
          </div>
        </Reveal>
      </section>

      <section className="mt-20">
        <Reveal>
          <h2 className="font-display text-3xl text-ink-900">How an answer is made</h2>
        </Reveal>
        <ol className="mt-8 space-y-4">
          {PIPELINE.map(([title, body], i) => (
            <Reveal key={title} delay={i * 0.05}>
              <li className="card flex gap-5 p-6">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-ink-900 bg-ink-900 font-mono text-paper">
                  {i + 1}
                </span>
                <div>
                  <h3 className="font-display text-xl text-ink-900">{title}</h3>
                  <p className="mt-1 text-ink-700">{body}</p>
                </div>
              </li>
            </Reveal>
          ))}
        </ol>
      </section>

      <section className="mt-20">
        <Reveal>
          <h2 className="font-display text-3xl text-ink-900">Tech stack</h2>
        </Reveal>
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {STACK.map(([layer, tech], i) => (
            <Reveal key={layer} delay={i * 0.04}>
              <div className="card-cream p-5">
                <p className="font-mono text-xs uppercase tracking-wider text-clay">{layer}</p>
                <p className="mt-2 text-ink-900">{tech}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      <section className="mt-16">
        <Reveal>
          <div className="card p-8 text-center">
            <p className="text-ink-700">Ready to see it work?</p>
            <Link href="/chat" className="btn mt-4">Open the chat</Link>
          </div>
        </Reveal>
      </section>
    </div>
  );
}
