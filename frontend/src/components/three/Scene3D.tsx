"use client";

import dynamic from "next/dynamic";
import { Component, type ReactNode } from "react";

function Fallback() {
  return (
    <div className="flex h-full w-full items-center justify-center">
      <div className="h-16 w-16 animate-pulse rounded-xl2 border border-sand bg-cream" />
    </div>
  );
}

/** A decorative 3D scene must never crash the page — fall back to the placeholder. */
class SceneBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    return this.state.failed ? <Fallback /> : this.props.children;
  }
}

const SCENES = {
  hero: dynamic(() => import("./HeroScene"), { ssr: false, loading: Fallback }),
  graph: dynamic(() => import("./KnowledgeGraphScene"), { ssr: false, loading: Fallback }),
  ring: dynamic(() => import("./DocumentRingScene"), { ssr: false, loading: Fallback }),
} as const;

export function Scene3D({
  name,
  className = "",
}: {
  name: keyof typeof SCENES;
  className?: string;
}) {
  const Scene = SCENES[name];
  return (
    <div className={className}>
      <SceneBoundary>
        <Scene />
      </SceneBoundary>
    </div>
  );
}
