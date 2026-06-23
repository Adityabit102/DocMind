"use client";

/**
 * Hero 3D motif: a slowly rotating stack of document "pages" with lines of text,
 * representing documents being ingested and indexed. Flat matte materials in the
 * palette — no metallic glints, no glow. Topic-relevant, not decorative noise.
 */
import { Canvas, useFrame } from "@react-three/fiber";
import { ContactShadows, Float } from "@react-three/drei";
import { Suspense, useRef } from "react";
import type { Group } from "three";
import { MathUtils } from "three";

const PALETTE = {
  ink: "#92836c",
  clay: "#a8957f",
  sand: "#c8bba9",
  cream: "#ecdcc4",
  paper: "#f7f1e9",
};

function TextLines({ z }: { z: number }) {
  // thin bars standing in for lines of text on a page
  const widths = [1.5, 1.2, 1.6, 0.9, 1.4, 1.1];
  return (
    <group position={[0, 0, z]}>
      {widths.map((w, i) => (
        <mesh key={i} position={[-(1.7 - w) / 2, 1.05 - i * 0.32, 0.011]}>
          <planeGeometry args={[w, 0.08]} />
          <meshStandardMaterial color={PALETTE.sand} roughness={1} />
        </mesh>
      ))}
    </group>
  );
}

function Page({ index }: { index: number }) {
  const offset = index * 0.16;
  const mesh = useRef<Group>(null);
  // The top sheet periodically lifts and tilts — a document being "read in".
  useFrame((state) => {
    if (!mesh.current || index !== 0) return;
    const t = state.clock.elapsedTime;
    const lift = (Math.sin(t * 0.6) * 0.5 + 0.5) ** 2; // 0→1 eased pulse
    mesh.current.position.z = 0.12 + lift * 0.5;
    mesh.current.rotation.x = -lift * 0.22;
  });
  return (
    <group
      ref={mesh}
      position={[index * 0.22, -index * 0.12, -index * 0.22]}
      rotation={[0, offset, offset * 0.3]}
    >
      <mesh castShadow>
        <boxGeometry args={[2, 2.6, 0.02]} />
        <meshStandardMaterial color={PALETTE.paper} roughness={0.95} />
      </mesh>
      {/* folded corner accent */}
      <mesh position={[0.85, 1.1, 0.02]} rotation={[0, 0, Math.PI / 4]}>
        <planeGeometry args={[0.3, 0.3]} />
        <meshStandardMaterial color={PALETTE.cream} roughness={1} />
      </mesh>
      <TextLines z={0.012} />
    </group>
  );
}

function PaperStack() {
  const group = useRef<Group>(null);
  useFrame((state) => {
    if (!group.current) return;
    const t = state.clock.elapsedTime;
    // Idle drift, eased toward the pointer for a subtle parallax that tracks
    // the cursor without ever feeling twitchy.
    const targetY = Math.sin(t * 0.25) * 0.3 + 0.3 + state.pointer.x * 0.5;
    const targetX = Math.cos(t * 0.2) * 0.1 - 0.05 - state.pointer.y * 0.3;
    group.current.rotation.y = MathUtils.lerp(group.current.rotation.y, targetY, 0.05);
    group.current.rotation.x = MathUtils.lerp(group.current.rotation.x, targetX, 0.05);
  });
  return (
    <group ref={group}>
      {[3, 2, 1, 0].map((i) => (
        <Page key={i} index={i} />
      ))}
    </group>
  );
}

export default function HeroScene() {
  return (
    <Canvas
      camera={{ position: [0, 0, 6.5], fov: 42 }}
      dpr={[1, 2]}
      gl={{ antialias: true, alpha: true }}
    >
      <ambientLight intensity={0.85} />
      <directionalLight position={[4, 6, 5]} intensity={1.1} color={PALETTE.cream} />
      <directionalLight position={[-5, -2, 2]} intensity={0.35} color={PALETTE.clay} />
      <Suspense fallback={null}>
        <Float speed={1.4} rotationIntensity={0.25} floatIntensity={0.6}>
          <PaperStack />
        </Float>
        <ContactShadows
          position={[0, -2.1, 0]}
          opacity={0.28}
          scale={9}
          blur={2.6}
          far={4}
          color={PALETTE.ink}
        />
      </Suspense>
    </Canvas>
  );
}
