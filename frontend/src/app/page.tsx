import Link from "next/link";
import { Scene3D } from "@/components/three/Scene3D";
import { Reveal } from "@/components/framer/Reveal";
import { TextArrowCTA } from "@/components/framer/TextArrowCTA";
import { CircleExpandButton } from "@/components/framer/CircleExpandButton";
import { ProjectHoverCard } from "@/components/framer/ProjectHoverCard";
import { ImageScroller } from "@/components/framer/ImageScroller";
import { RunaroundText } from "@/components/framer/RunaroundText";

const FEATURES = [
  { index: "01", title: "Hybrid retrieval", description: "Dense FAISS embeddings fused with BM25 keyword search via reciprocal rank fusion — meaning and exact terms, together." },
  { index: "02", title: "Cross-encoder reranking", description: "Top candidates are re-scored by a cross-encoder that reads query and chunk jointly, pushing the most relevant context to the model." },
  { index: "03", title: "Grounded & cited", description: "Every claim references its source file and page. Open any citation to read the exact chunk it came from." },
  { index: "04", title: "HyDE & query expansion", description: "Vague questions are rewritten and expanded; hypothetical-answer embeddings lift recall on abstract queries." },
  { index: "05", title: "RAGAS evaluation", description: "Faithfulness, answer relevancy, context precision and recall — measured, charted, and exportable." },
  { index: "06", title: "Local-first, no keys", description: "Runs end-to-end on local HuggingFace embeddings and Ollama. Bring an OpenAI or Anthropic key only if you want to." },
];

const FORMATS = ["PDF", "DOCX", "Markdown", "TXT", "CSV", "HTML", "Scanned PDF · OCR", "Tables"];

export default function Home() {
  return (
    <>
      {/* ── Hero ─────────────────────────────────────────────── */}
      <section className="mx-auto grid max-w-7xl items-center gap-8 px-5 py-16 md:grid-cols-2 md:py-24">
        <div className="animate-fade-up">
          <p className="eyebrow">RAG Document Q&amp;A · Grounded by design</p>
          <h1 className="mt-4 font-display text-5xl leading-[1.05] text-ink-900 md:text-6xl">
            Ask your documents.
            <br />
            Get answers you can <span className="italic">trace</span>.
          </h1>
          <p className="mt-6 max-w-md text-lg text-ink-700">
            Upload PDFs, contracts, papers or notes and ask in plain language.
            DocMind retrieves the right passages, reranks them, and answers with a
            citation on every line — so you can verify, not just trust.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-4">
            <Link href="/chat">
              <CircleExpandButton>Start asking</CircleExpandButton>
            </Link>
            <TextArrowCTA href="/documents">Upload documents</TextArrowCTA>
          </div>
        </div>

        <div className="relative h-[360px] md:h-[460px]">
          <Scene3D name="hero" className="absolute inset-0" />
        </div>
      </section>

      {/* ── Format scroller ──────────────────────────────────── */}
      <section className="border-y border-sand bg-cream/40 py-6">
        <div className="mx-auto max-w-7xl px-5">
          <p className="eyebrow mb-3">Ingests anything you throw at it</p>
          <ImageScroller items={FORMATS} />
        </div>
      </section>

      {/* ── Feature grid ─────────────────────────────────────── */}
      <section className="mx-auto max-w-7xl px-5 py-20">
        <Reveal>
          <p className="eyebrow">What&apos;s inside</p>
          <h2 className="mt-3 max-w-2xl font-display text-4xl text-ink-900">
            Production RAG, not a notebook demo
          </h2>
        </Reveal>
        <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f, i) => (
            <Reveal key={f.index} delay={i * 0.05}>
              <ProjectHoverCard {...f} />
            </Reveal>
          ))}
        </div>
      </section>

      {/* ── Retrieval graph + run-around ─────────────────────── */}
      <section className="mx-auto grid max-w-7xl items-center gap-10 px-5 py-20 md:grid-cols-2">
        <Reveal className="order-2 md:order-1">
          <RunaroundText>
            Behind every answer is a retrieval graph. Your documents are split into
            overlapping chunks, embedded into a vector space, and indexed alongside
            a BM25 keyword model. When you ask a question, the closest chunks light
            up, get reranked by relevance, and only the strongest survive into the
            prompt. The model is told to answer strictly from them — and to say so
            when the documents simply don&apos;t know.
          </RunaroundText>
        </Reveal>
        <Reveal className="order-1 md:order-2">
          <div className="card relative h-[340px] overflow-hidden">
            <Scene3D name="graph" className="absolute inset-0" />
            <span className="absolute bottom-4 left-4 font-mono text-xs text-ink-700">
              embedding space · 26 chunks · 1 query
            </span>
          </div>
        </Reveal>
      </section>

      {/* ── CTA band ─────────────────────────────────────────── */}
      <section className="mx-auto max-w-7xl px-5 pb-8">
        <Reveal>
          <div className="card-cream flex flex-col items-start justify-between gap-6 p-10 md:flex-row md:items-center">
            <div>
              <h2 className="font-display text-3xl text-ink-900">Put it to work on your files</h2>
              <p className="mt-2 max-w-md text-ink-700">
                No setup theatre. Drop in documents and start asking — every answer
                comes back with its receipts.
              </p>
            </div>
            <Link href="/chat">
              <CircleExpandButton className="bg-paper">Open the chat</CircleExpandButton>
            </Link>
          </div>
        </Reveal>
      </section>
    </>
  );
}
