import React, { useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";
import { useStudioStore, type CameraPreset } from "../store";
import type { TelemetryV2 } from "../types";

const BONE_PAIRS: Array<[number, number, string]> = [
  [0, 3, "#f078ff"], [3, 6, "#f078ff"], [6, 9, "#f078ff"], [9, 12, "#f078ff"], [12, 15, "#f078ff"],
  [0, 1, "#38bdf8"], [1, 4, "#38bdf8"], [4, 7, "#38bdf8"], [7, 10, "#38bdf8"],
  [0, 2, "#fb923c"], [2, 5, "#fb923c"], [5, 8, "#fb923c"], [8, 11, "#fb923c"],
  [9, 13, "#60a5fa"], [13, 16, "#60a5fa"], [16, 18, "#60a5fa"], [18, 20, "#60a5fa"], [20, 22, "#60a5fa"],
  [9, 14, "#4ade80"], [14, 17, "#4ade80"], [17, 19, "#4ade80"], [19, 21, "#4ade80"], [21, 23, "#4ade80"],
];

function nearestFfcFrame(timeRelMs: number[]): number {
  let best = 0;
  let bestAbs = Infinity;
  for (let i = 0; i < timeRelMs.length; i++) {
    const a = Math.abs(timeRelMs[i]);
    if (a < bestAbs) {
      bestAbs = a;
      best = i;
    }
  }
  return best;
}

function CameraRig({
  preset,
  resetNonce,
  camWorldPos,
  targetCenter,
  controlsRef,
}: {
  preset: CameraPreset;
  resetNonce: number;
  camWorldPos: [number, number, number];
  targetCenter: [number, number, number];
  controlsRef: React.MutableRefObject<any>;
}) {
  const { camera } = useThree();
  const appliedKeyRef = useRef<string>("");
  const rigKey = `${preset}|${resetNonce}|${targetCenter[0].toFixed(3)},${targetCenter[1].toFixed(3)},${targetCenter[2].toFixed(3)}`;

  useFrame(() => {
    if (appliedKeyRef.current === rigKey) return;
    appliedKeyRef.current = rigKey;
    const [cx, cy, cz] = targetCenter;
    if (preset === "side") {
      camera.position.set(cx, cy + 0.5, cz + 4.4);
    } else if (preset === "rear") {
      camera.position.set(cx - 4.4, cy + 0.6, cz);
    } else if (preset === "top") {
      camera.position.set(cx, cy + 5.2, cz + 0.01);
    } else {
      const dx = camWorldPos[0] - cx;
      const dy = camWorldPos[1] - cy;
      const dz = camWorldPos[2] - cz;
      const dist = Math.hypot(dx, dy, dz) || 1;
      const clamped = Math.min(Math.max(dist, 3.2), 7.5) / dist;
      camera.position.set(cx + dx * clamped, Math.max(0.4, cy + dy * clamped), cz + dz * clamped);
    }
    camera.lookAt(cx, cy, cz);
    if (controlsRef.current) {
      controlsRef.current.target.set(cx, cy, cz);
      controlsRef.current.update();
    }
  });
  return null;
}

function AnimatedSkeleton3D({
  telemetry,
  ghostTelemetry,
  showUncertainty,
}: {
  telemetry: TelemetryV2;
  ghostTelemetry: TelemetryV2 | null;
  showUncertainty: boolean;
}) {
  const jointMeshRefs = useRef<Array<THREE.Mesh | null>>([]);
  const boneRefs = useRef<Array<THREE.Mesh | null>>([]);
  const ghostBoneRefs = useRef<Array<THREE.Mesh | null>>([]);
  const plumbRef = useRef<THREE.Mesh | null>(null);
  const upVec = useMemo(() => new THREE.Vector3(0, 1, 0), []);
  const vA = useMemo(() => new THREE.Vector3(), []);
  const vB = useMemo(() => new THREE.Vector3(), []);
  const dir = useMemo(() => new THREE.Vector3(), []);
  const mid = useMemo(() => new THREE.Vector3(), []);
  const quat = useMemo(() => new THREE.Quaternion(), []);

  // Align ghost horizontal (X, Z) pelvis anchor at FFC (t_rel = 0 ms) onto primary bowler's FFC anchor
  const ghostShiftXZ = useMemo<[number, number]>(() => {
    if (!ghostTelemetry || ghostTelemetry.num_frames === 0 || telemetry.num_frames === 0) return [0, 0];
    const pIdx = nearestFfcFrame(telemetry.time_rel_ffc_ms);
    const gIdx = nearestFfcFrame(ghostTelemetry.time_rel_ffc_ms);
    const pPelv = telemetry.joints3d_world_m[pIdx]?.[0] ?? [0, 0, 0];
    const gPelv = ghostTelemetry.joints3d_world_m[gIdx]?.[0] ?? [0, 0, 0];
    return [pPelv[0] - gPelv[0], pPelv[2] - gPelv[2]];
  }, [telemetry, ghostTelemetry]);

  useFrame(() => {
    const f = Math.min(Math.max(useStudioStore.getState().currentFrame, 0), telemetry.num_frames - 1);
    const pts = telemetry.joints3d_world_m[f];
    const uncs = telemetry.joint_uncertainties_m[f];
    if (!pts) return;

    // Update joints + uncertainty spheres
    for (let i = 0; i < 24; i++) {
      const m = jointMeshRefs.current[i];
      if (!m || !pts[i]) continue;
      m.position.set(pts[i][0], pts[i][1], pts[i][2]);
      const r = showUncertainty && uncs ? Math.min(Math.max(uncs[i] * 0.75, 0.022), 0.14) : 0.028;
      m.scale.setScalar(r);
    }

    // Update bone cylinders cleanly in-place (zero React allocations!)
    for (let b = 0; b < BONE_PAIRS.length; b++) {
      const mesh = boneRefs.current[b];
      if (!mesh) continue;
      const [i0, i1] = BONE_PAIRS[b];
      vA.set(pts[i0][0], pts[i0][1], pts[i0][2]);
      vB.set(pts[i1][0], pts[i1][1], pts[i1][2]);
      dir.subVectors(vB, vA);
      const len = dir.length();
      if (len < 1e-4) continue;
      mid.addVectors(vA, vB).multiplyScalar(0.5);
      mesh.position.copy(mid);
      mesh.scale.set(1, len, 1);
      quat.setFromUnitVectors(upVec, dir.normalize());
      mesh.quaternion.copy(quat);
    }

    // Gravity plumb line beneath pelvis
    if (plumbRef.current && pts[0]) {
      plumbRef.current.position.set(pts[0][0], 1.1, pts[0][2]);
    }

    // FFC-Aligned, Real-Time (ms) Interpolated Ghost Skeleton (handles mixed 25 fps vs 240 fps!)
    if (ghostTelemetry && ghostTelemetry.num_frames > 0) {
      const tMs = telemetry.time_rel_ffc_ms[f] ?? 0;
      const gTimes = ghostTelemetry.time_rel_ffc_ms;
      const gn = gTimes.length;
      let iLo = 0;
      let iHi = gn - 1;
      if (tMs <= gTimes[0]) {
        iLo = 0;
        iHi = 0;
      } else if (tMs >= gTimes[gn - 1]) {
        iLo = gn - 1;
        iHi = gn - 1;
      } else {
        while (iLo + 1 < iHi) {
          const mIdx = (iLo + iHi) >> 1;
          if (gTimes[mIdx] <= tMs) iLo = mIdx;
          else iHi = mIdx;
        }
      }
      const span = gTimes[iHi] - gTimes[iLo];
      const alpha = span > 1e-6 ? Math.min(Math.max((tMs - gTimes[iLo]) / span, 0), 1) : 0;
      const gp0 = ghostTelemetry.joints3d_world_m[iLo];
      const gp1 = ghostTelemetry.joints3d_world_m[iHi];
      const [dx, dz] = ghostShiftXZ;
      if (gp0 && gp1) {
        for (let b = 0; b < BONE_PAIRS.length; b++) {
          const mesh = ghostBoneRefs.current[b];
          if (!mesh) continue;
          const [i0, i1] = BONE_PAIRS[b];
          vA.set(
            gp0[i0][0] + alpha * (gp1[i0][0] - gp0[i0][0]) + dx,
            gp0[i0][1] + alpha * (gp1[i0][1] - gp0[i0][1]),
            gp0[i0][2] + alpha * (gp1[i0][2] - gp0[i0][2]) + dz
          );
          vB.set(
            gp0[i1][0] + alpha * (gp1[i1][0] - gp0[i1][0]) + dx,
            gp0[i1][1] + alpha * (gp1[i1][1] - gp0[i1][1]),
            gp0[i1][2] + alpha * (gp1[i1][2] - gp0[i1][2]) + dz
          );
          dir.subVectors(vB, vA);
          const len = dir.length();
          if (len < 1e-4) continue;
          mid.addVectors(vA, vB).multiplyScalar(0.5);
          mesh.position.copy(mid);
          mesh.scale.set(1, len, 1);
          quat.setFromUnitVectors(upVec, dir.normalize());
          mesh.quaternion.copy(quat);
        }
      }
    }
  });

  return (
    <group>
      {/* Primary SMPL-24 joints + uncertainty spheres */}
      {Array.from({ length: 24 }).map((_, i) => (
        <mesh key={`j-${i}`} ref={(el) => { jointMeshRefs.current[i] = el; }}>
          <sphereGeometry args={[1, 14, 14]} />
          <meshStandardMaterial
            color={showUncertainty ? "#38bdf8" : "#f8fafc"}
            transparent={showUncertainty}
            opacity={showUncertainty ? 0.42 : 0.95}
          />
        </mesh>
      ))}
      {/* Primary bones */}
      {BONE_PAIRS.map(([, , col], b) => (
        <mesh key={`b-${b}`} ref={(el) => { boneRefs.current[b] = el; }}>
          <cylinderGeometry args={[0.018, 0.018, 1, 8]} />
          <meshStandardMaterial color={col} roughness={0.35} />
        </mesh>
      ))}
      {/* Vertical gravity plumb line */}
      <mesh ref={plumbRef}>
        <cylinderGeometry args={[0.005, 0.005, 2.2, 6]} />
        <meshBasicMaterial color="#94a3b8" transparent opacity={0.45} />
      </mesh>
      {/* Optional Ghost Delivery B skeleton (amber translucent) */}
      {ghostTelemetry &&
        BONE_PAIRS.map((_, b) => (
          <mesh key={`gb-${b}`} ref={(el) => { ghostBoneRefs.current[b] = el; }}>
            <cylinderGeometry args={[0.014, 0.014, 1, 6]} />
            <meshStandardMaterial color="#f59e0b" transparent opacity={0.42} />
          </mesh>
        ))}
    </group>
  );
}

function MCCPitchDeck({
  camWorldPos,
  targetCenter,
}: {
  camWorldPos: [number, number, number];
  targetCenter: [number, number, number];
}) {
  // Position compact camera indicator along camera azimuth direction at 2.35 m radius so it stays unobtrusive
  const camIndicator = useMemo<[number, number, number]>(() => {
    const dx = camWorldPos[0] - targetCenter[0];
    const dy = camWorldPos[1] - targetCenter[1];
    const dz = camWorldPos[2] - targetCenter[2];
    const d = Math.hypot(dx, dy, dz) || 1;
    const s = Math.min(d, 2.35) / d;
    return [targetCenter[0] + dx * s, Math.max(0.28, targetCenter[1] + dy * s), targetCenter[2] + dz * s];
  }, [camWorldPos, targetCenter]);

  return (
    <group>
      {/* Pitch turf ground plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[targetCenter[0], -0.004, targetCenter[2]]}>
        <planeGeometry args={[12, 8]} />
        <meshStandardMaterial color="#13221d" roughness={0.95} />
      </mesh>
      <gridHelper args={[12, 24, "#263a33", "#1a2824"]} position={[targetCenter[0], 0.001, targetCenter[2]]} />

      {/* Regulation MCC Law 7 Crease Markings anchored at X = 0 (bowling crease) & X = 1.22 m (popping crease) */}
      <Line points={[[0, 0.01, -1.32], [0, 0.01, 1.32]]} color="#e2e8f0" lineWidth={1.8} />
      <Line points={[[1.22, 0.01, -1.83], [1.22, 0.01, 1.83]]} color="#38bdf8" lineWidth={2.2} />
      <Line points={[[-1.22, 0.01, -1.32], [1.22, 0.01, -1.32]]} color="#cbd5e1" lineWidth={1.5} />
      <Line points={[[-1.22, 0.01, 1.32], [1.22, 0.01, 1.32]]} color="#cbd5e1" lineWidth={1.5} />

      {/* MCC Law 8 Stumps (0.7112 m height, +-0.0960 m off/leg center spacing) */}
      {[-0.096, 0.0, 0.096].map((zOff, i) => (
        <mesh key={i} position={[0, 0.3556, zOff]}>
          <cylinderGeometry args={[0.017, 0.017, 0.7112, 10]} />
          <meshStandardMaterial color="#facc15" />
        </mesh>
      ))}

      {/* Compact Camera Viewing-Axis Indicator */}
      <group position={camIndicator}>
        <mesh>
          <boxGeometry args={[0.09, 0.06, 0.11]} />
          <meshStandardMaterial color="#fb7185" emissive="#fb7185" emissiveIntensity={0.3} transparent opacity={0.78} />
        </mesh>
      </group>
      <Line points={[camIndicator, targetCenter]} color="#fb7185" lineWidth={1.0} dashed dashSize={0.12} gapSize={0.08} />
    </group>
  );
}

interface ViewportProps {
  telemetry: TelemetryV2;
  ghostTelemetry: TelemetryV2 | null;
}

export const PitchViewport3D: React.FC<ViewportProps> = ({ telemetry, ghostTelemetry }) => {
  const cameraPreset = useStudioStore((s) => s.cameraPreset);
  const cameraResetNonce = useStudioStore((s) => s.cameraResetNonce);
  const setCameraPreset = useStudioStore((s) => s.setCameraPreset);
  const showUncertainty = useStudioStore((s) => s.showUncertaintySpheres);
  const setShowUncertainty = useStudioStore((s) => s.setShowUncertaintySpheres);
  const controlsRef = useRef<any>(null);

  const targetCenter = useMemo<[number, number, number]>(() => {
    const midIdx = Math.floor(telemetry.num_frames / 2);
    const p = telemetry.joints3d_world_m[midIdx]?.[0] ?? [0.6, 0.9, 0];
    return [p[0], 0.95, p[2]];
  }, [telemetry]);

  return (
    <div className="flex flex-col bg-[#131926] border border-[#263248] rounded-xl overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-1 px-3 py-2 border-b border-[#263248] bg-[#161e2e] text-xs">
        <span className="font-semibold text-slate-200 whitespace-nowrap">3D MCC Pitch-World Viewport</span>
        <div className="flex items-center gap-1">
          {([
            ["side", "Side 90°"],
            ["rear", "Rear 0°"],
            ["top", "Top-Down"],
            ["camera", "Cam Match"],
          ] as Array<[CameraPreset, string]>).map(([p, lbl]) => (
            <button
              key={p}
              onClick={() => setCameraPreset(p)}
              className={`px-2 py-0.5 rounded text-[11px] transition ${
                cameraPreset === p ? "bg-sky-500 text-slate-950 font-bold" : "bg-[#1e293b] text-slate-300 hover:bg-slate-700"
              }`}
            >
              {lbl}
            </button>
          ))}
          <button
            onClick={() => setShowUncertainty(!showUncertainty)}
            className={`ml-1 px-2 py-0.5 rounded text-[11px] ${
              showUncertainty ? "bg-sky-500/20 text-sky-300 border border-sky-500/40" : "bg-[#1e293b] text-slate-400"
            }`}
            title="Scale joint spheres by 3D localization uncertainty"
          >
            σ Spheres
          </button>
        </div>
      </div>

      <div className="relative h-[360px] w-full bg-[#0b0f17]">
        <Canvas camera={{ position: [targetCenter[0], 1.45, targetCenter[2] + 4.4], fov: 44 }}>
          <color attach="background" args={["#0b0f17"]} />
          <ambientLight intensity={0.7} />
          <directionalLight position={[4, 8, 5]} intensity={1.1} />
          <MCCPitchDeck camWorldPos={telemetry.camera_position_world_m} targetCenter={targetCenter} />
          <AnimatedSkeleton3D
            telemetry={telemetry}
            ghostTelemetry={ghostTelemetry}
            showUncertainty={showUncertainty}
          />
          <CameraRig
            preset={cameraPreset}
            resetNonce={cameraResetNonce}
            camWorldPos={telemetry.camera_position_world_m}
            targetCenter={targetCenter}
            controlsRef={controlsRef}
          />
          <OrbitControls ref={controlsRef} target={targetCenter} enableDamping dampingFactor={0.08} />
        </Canvas>
        <div className="absolute bottom-2 left-2 px-2 py-1 rounded bg-[#0b0f17]/80 border border-[#263248] text-[10px] font-mono text-slate-300 pointer-events-none">
          Red cube = camera viewing frustum • Yellow = MCC Law 8 stumps (0.711 m)
          {ghostTelemetry ? " • Amber = FFC-aligned ms-synced Ghost" : ""}
        </div>
      </div>
    </div>
  );
};
