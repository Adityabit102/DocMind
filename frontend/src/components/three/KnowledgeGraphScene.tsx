"use client";

/**
 * Retrieval motif: a 3D knowledge graph of embedding nodes connected by edges,
 * with a brighter "query" node pulsing at the centre — the moment a question
 * matches relevant chunks. Palette-only, flat materials.
 */
import { Canvas, useFrame } from "@react-three/fiber";
import { Suspense, useMemo, useRef } from "react";
import type { Group, Mesh } from "three";
import { MathUtils, Vector3 } from "three";

const PALETTE = { ink: "#92836c", clay: "#a8957f", sand: "#c8bba9", cream: "#ecdcc4" };

function useNodes(count: number) {
  return useMemo(() => {
    const nodes: Vector3[] = [];
    for (let i = 0; i < count; i++) {
      // fibonacci sphere for an even, organic distribution
      const phi = Math.acos(1 - (2 * (i + 0.5)) / count);
      const theta = Math.PI * (1 + Math.sqrt(5)) * i;
      const r = 2.4;
      nodes.push(
        new Vector3(
          r * Math.sin(phi) * Math.cos(theta),
          r * Math.sin(phi) * Math.sin(theta),
          r * Math.cos(phi),
        ),
      );
    }
    return nodes;
  }, [count]);
}

function Edges({ nodes }: { nodes: Vector3[] }) {
  const lines = useMemo(() => {
    const segs: [Vector3, Vector3][] = [];
    nodes.forEach((a, i) => {
      // connect each node to its 2 nearest neighbours
      const dists = nodes
        .map((b, j) => ({ j, d: a.distanceTo(b) }))
        .filter((x) => x.j !== i)
        .sort((x, y) => x.d - y.d)
        .slice(0, 2);
      dists.forEach(({ j }) => segs.push([a, nodes[j]]));
    });
    return segs;
  }, [nodes]);

  return (
    <>
      {lines.map(([a, b], i) => {
        const points = new Float32Array([a.x, a.y, a.z, b.x, b.y, b.z]);
        return (
          <line key={i}>
            <bufferGeometry>
              <bufferAttribute attach="attributes-position" args={[points, 3]} />
            </bufferGeometry>
            <lineBasicMaterial color={PALETTE.sand} transparent opacity={0.5} />
          </line>
        );
      })}
    </>
  );
}

function QueryNode() {
  const ref = useRef<Mesh>(null);
  useFrame((s) => {
    if (!ref.current) return;
    const p = 1 + Math.sin(s.clock.elapsedTime * 2) * 0.15;
    ref.current.scale.setScalar(p);
    ref.current.rotation.y = s.clock.elapsedTime * 0.4;
  });
  return (
    <mesh ref={ref}>
      <icosahedronGeometry args={[0.35, 1]} />
      <meshStandardMaterial color={PALETTE.ink} roughness={0.8} />
    </mesh>
  );
}

/** A small bead that travels out from the query node to a matched chunk and
 *  back — the visual of retrieval lighting up the nearest neighbours. */
function RetrievalPulses({ nodes }: { nodes: Vector3[] }) {
  const targets = useMemo(
    () => [nodes[3], nodes[9], nodes[15], nodes[21]].filter(Boolean),
    [nodes],
  );
  const refs = useRef<(Mesh | null)[]>([]);
  useFrame((s) => {
    targets.forEach((target, i) => {
      const mesh = refs.current[i];
      if (!mesh) return;
      // ping-pong 0→1→0 with a per-bead phase offset
      const phase = (s.clock.elapsedTime * 0.5 + i * 0.25) % 1;
      const tri = phase < 0.5 ? phase * 2 : (1 - phase) * 2;
      mesh.position.lerpVectors(new Vector3(0, 0, 0), target, tri);
      const visible = tri > 0.02;
      mesh.scale.setScalar(visible ? 0.07 : 0.0001);
    });
  });
  return (
    <>
      {targets.map((_, i) => (
        <mesh
          key={i}
          ref={(m) => {
            refs.current[i] = m;
          }}
        >
          <sphereGeometry args={[1, 12, 12]} />
          <meshStandardMaterial color={PALETTE.ink} roughness={0.6} />
        </mesh>
      ))}
    </>
  );
}

function Node({ position, color, seed }: { position: Vector3; color: string; seed: number }) {
  const ref = useRef<Mesh>(null);
  useFrame((s) => {
    if (!ref.current) return;
    // gentle individual bob so the cloud feels alive, not rigid
    const b = Math.sin(s.clock.elapsedTime * 0.8 + seed) * 0.04;
    ref.current.scale.setScalar(1 + b);
  });
  return (
    <mesh ref={ref} position={position}>
      <sphereGeometry args={[0.11, 16, 16]} />
      <meshStandardMaterial color={color} roughness={0.9} />
    </mesh>
  );
}

function Graph() {
  const group = useRef<Group>(null);
  const nodes = useNodes(32);
  useFrame((s) => {
    if (!group.current) return;
    group.current.rotation.y = s.clock.elapsedTime * 0.12 + s.pointer.x * 0.4;
    group.current.rotation.x = MathUtils.lerp(
      group.current.rotation.x,
      -s.pointer.y * 0.3,
      0.05,
    );
  });
  return (
    <group ref={group}>
      <Edges nodes={nodes} />
      {nodes.map((n, i) => (
        <Node
          key={i}
          position={n}
          color={i % 3 === 0 ? PALETTE.clay : PALETTE.cream}
          seed={i}
        />
      ))}
      <RetrievalPulses nodes={nodes} />
      <QueryNode />
    </group>
  );
}

export default function KnowledgeGraphScene() {
  return (
    <Canvas camera={{ position: [0, 0, 7], fov: 45 }} dpr={[1, 2]} gl={{ alpha: true }}>
      <ambientLight intensity={0.9} />
      <directionalLight position={[3, 4, 5]} intensity={0.8} />
      <Suspense fallback={null}>
        <Graph />
      </Suspense>
    </Canvas>
  );
}
