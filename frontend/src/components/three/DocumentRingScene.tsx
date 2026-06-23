"use client";

/**
 * Chat empty-state motif: a slow carousel of document "cards" orbiting a central
 * query node, each card gently facing outward. Reads as "your corpus, ready to be
 * asked." Flat matte palette materials, pointer-eased rotation — calm, not busy.
 */
import { Canvas, useFrame } from "@react-three/fiber";
import { ContactShadows, Float } from "@react-three/drei";
import { Suspense, useMemo, useRef } from "react";
import type { Group, Mesh } from "three";
import { MathUtils } from "three";

const PALETTE = {
  ink: "#92836c",
  clay: "#a8957f",
  sand: "#c8bba9",
  cream: "#ecdcc4",
  paper: "#f7f1e9",
};

function Card({ angle, radius }: { angle: number; radius: number }) {
  const x = Math.cos(angle) * radius;
  const z = Math.sin(angle) * radius;
  const lines = [0.9, 0.7, 1.0, 0.6];
  return (
    <group position={[x, 0, z]} rotation={[0, -angle + Math.PI / 2, 0]}>
      <mesh castShadow>
        <boxGeometry args={[1.1, 1.45, 0.04]} />
        <meshStandardMaterial color={PALETTE.paper} roughness={0.95} />
      </mesh>
      <mesh position={[0.4, 0.62, 0.03]} rotation={[0, 0, Math.PI / 4]}>
        <planeGeometry args={[0.22, 0.22]} />
        <meshStandardMaterial color={PALETTE.cream} roughness={1} />
      </mesh>
      {lines.map((w, i) => (
        <mesh key={i} position={[-(1.0 - w) / 2 - 0.02, 0.32 - i * 0.22, 0.025]}>
          <planeGeometry args={[w, 0.06]} />
          <meshStandardMaterial color={PALETTE.sand} roughness={1} />
        </mesh>
      ))}
    </group>
  );
}

function QueryCore() {
  const ref = useRef<Mesh>(null);
  useFrame((s) => {
    if (!ref.current) return;
    ref.current.rotation.y = s.clock.elapsedTime * 0.5;
    ref.current.scale.setScalar(1 + Math.sin(s.clock.elapsedTime * 2) * 0.08);
  });
  return (
    <mesh ref={ref}>
      <icosahedronGeometry args={[0.32, 1]} />
      <meshStandardMaterial color={PALETTE.ink} roughness={0.8} />
    </mesh>
  );
}

function Carousel() {
  const group = useRef<Group>(null);
  const cards = useMemo(() => [0, 1, 2, 3, 4].map((i) => (i / 5) * Math.PI * 2), []);
  useFrame((s) => {
    if (!group.current) return;
    group.current.rotation.y = s.clock.elapsedTime * 0.18 + s.pointer.x * 0.4;
    group.current.rotation.x = MathUtils.lerp(
      group.current.rotation.x,
      0.12 - s.pointer.y * 0.2,
      0.05,
    );
  });
  return (
    <group ref={group}>
      {cards.map((angle, i) => (
        <Card key={i} angle={angle} radius={2.1} />
      ))}
      <QueryCore />
    </group>
  );
}

export default function DocumentRingScene() {
  return (
    <Canvas camera={{ position: [0, 1.4, 5.4], fov: 42 }} dpr={[1, 2]} gl={{ alpha: true }}>
      <ambientLight intensity={0.9} />
      <directionalLight position={[4, 6, 5]} intensity={1} color={PALETTE.cream} />
      <directionalLight position={[-5, -1, 2]} intensity={0.3} color={PALETTE.clay} />
      <Suspense fallback={null}>
        <Float speed={1.1} rotationIntensity={0.15} floatIntensity={0.4}>
          <Carousel />
        </Float>
        <ContactShadows
          position={[0, -1.6, 0]}
          opacity={0.25}
          scale={10}
          blur={2.8}
          far={4}
          color={PALETTE.ink}
        />
      </Suspense>
    </Canvas>
  );
}
