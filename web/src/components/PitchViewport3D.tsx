import React, { useMemo, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";
import { useStudioStore, type CameraPreset } from "../store";
import type { TelemetryV2 } from "../types";

// Monochrome Vercel editorial bone scheme: #171717 torso/back leg/non-bowling arm, #297a3a bowling arm & front leg
function getBonePairs(bowlingArm: "right" | "left"): Array<[number, number, string]> {
  const isRight = bowlingArm === "right";
  const cSpine = "#171717";
  const cFrontLeg = "#297a3a";
  const cBackLeg = "#52525b";
  const cBowlArm = "#297a3a";
  const cNonBowlArm = "#52525b";

  const leftLegCol = isRight ? cFrontLeg : cBackLeg;
  const rightLegCol = isRight ? cBackLeg : cFrontLeg;
  const leftArmCol = isRight ? cNonBowlArm : cBowlArm;
  const rightArmCol = isRight ? cBowlArm : cNonBowlArm;

  return [
    [0, 3, cSpine], [3, 6, cSpine], [6, 9, cSpine], [9, 12, cSpine], [12, 15, cSpine],
    [0, 1, leftLegCol], [1, 4, leftLegCol], [4, 7, leftLegCol], [7, 10, leftLegCol],
    [0, 2, rightLegCol], [2, 5, rightLegCol], [5, 8, rightLegCol], [8, 11, rightLegCol],
    [9, 13, leftArmCol], [13, 16, leftArmCol], [16, 18, leftArmCol], [18, 20, leftArmCol], [20, 22, leftArmCol],
    [9, 14, rightArmCol], [14, 17, rightArmCol], [17, 19, rightArmCol], [19, 21, rightArmCol], [21, 23, rightArmCol],
  ];
}

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
  bowlingArm,
}: {
  telemetry: TelemetryV2;
  ghostTelemetry: TelemetryV2 | null;
  showUncertainty: boolean;
  bowlingArm: "right" | "left";
}) {
  const bonePairs = useMemo(() => getBonePairs(bowlingArm), [bowlingArm]);
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

    for (let i = 0; i < 24; i++) {
      const m = jointMeshRefs.current[i];
      if (!m || !pts[i]) continue;
      m.position.set(pts[i][0], pts[i][1], pts[i][2]);
      const r = showUncertainty && uncs ? Math.min(Math.max(uncs[i] * 0.72, 0.022), 0.13) : 0.026;
      m.scale.setScalar(r);
    }

    for (let b = 0; b < bonePairs.length; b++) {
      const mesh = boneRefs.current[b];
      if (!mesh) continue;
      const [i0, i1] = bonePairs[b];
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

    if (plumbRef.current && pts[0]) {
      plumbRef.current.position.set(pts[0][0], 1.1, pts[0][2]);
    }

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
        for (let b = 0; b < bonePairs.length; b++) {
          const mesh = ghostBoneRefs.current[b];
          if (!mesh) continue;
          const [i0, i1] = bonePairs[b];
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
      {Array.from({ length: 24 }).map((_, i) => (
        <mesh key={`j-${i}`} ref={(el) => { jointMeshRefs.current[i] = el; }}>
          <sphereGeometry args={[1, 14, 14]} />
          <meshStandardMaterial
            color={showUncertainty ? "#297a3a" : "#171717"}
            transparent={showUncertainty}
            opacity={showUncertainty ? 0.28 : 0.95}
          />
        </mesh>
      ))}
      {bonePairs.map(([, , col], b) => (
        <mesh key={`b-${b}`} ref={(el) => { boneRefs.current[b] = el; }}>
          <cylinderGeometry args={[0.017, 0.017, 1, 8]} />
          <meshStandardMaterial color={col} roughness={0.45} />
        </mesh>
      ))}
      <mesh ref={plumbRef}>
        <cylinderGeometry args={[0.004, 0.004, 2.2, 6]} />
        <meshBasicMaterial color="#a8a8a8" transparent opacity={0.55} />
      </mesh>
      {ghostTelemetry &&
        bonePairs.map((_, b) => (
          <mesh key={`gb-${b}`} ref={(el) => { ghostBoneRefs.current[b] = el; }}>
            <cylinderGeometry args={[0.013, 0.013, 1, 6]} />
            <meshStandardMaterial color="#a1a1aa" transparent opacity={0.48} />
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
      {/* Crisp light-paper pitch floor (#f0f0f0) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[targetCenter[0], -0.004, targetCenter[2]]}>
        <planeGeometry args={[12, 8]} />
        <meshStandardMaterial color="#f0f0f0" roughness={0.98} />
      </mesh>
      <gridHelper args={[12, 24, "#d4d4d8", "#e4e4e7"]} position={[targetCenter[0], 0.001, targetCenter[2]]} />

      {/* MCC Law 7 Crease Markings */}
      <Line points={[[0, 0.01, -1.32], [0, 0.01, 1.32]]} color="#171717" lineWidth={1.6} />
      <Line points={[[1.22, 0.01, -1.83], [1.22, 0.01, 1.83]]} color="#297a3a" lineWidth={2.2} />
      <Line points={[[-1.22, 0.01, -1.32], [1.22, 0.01, -1.32]]} color="#7d7d7d" lineWidth={1.3} />
      <Line points={[[-1.22, 0.01, 1.32], [1.22, 0.01, 1.32]]} color="#7d7d7d" lineWidth={1.3} />

      {/* MCC Law 8 Stumps (0.7112 m) */}
      {[-0.096, 0.0, 0.096].map((zOff, i) => (
        <mesh key={i} position={[0, 0.3556, zOff]}>
          <cylinderGeometry args={[0.017, 0.017, 0.7112, 10]} />
          <meshStandardMaterial color="#171717" />
        </mesh>
      ))}

      {/* Monochrome Camera Viewing-Axis Indicator */}
      <group position={camIndicator}>
        <mesh>
          <boxGeometry args={[0.09, 0.06, 0.11]} />
          <meshStandardMaterial color="#171717" />
        </mesh>
      </group>
      <Line points={[camIndicator, targetCenter]} color="#7d7d7d" lineWidth={1.0} dashed dashSize={0.12} gapSize={0.08} />
    </group>
  );
}

interface ViewportProps {
  telemetry: TelemetryV2;
  ghostTelemetry: TelemetryV2 | null;
  bowlingArm?: "right" | "left";
}

export const PitchViewport3D: React.FC<ViewportProps> = ({
  telemetry,
  ghostTelemetry,
  bowlingArm = "right",
}) => {
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
    <div className="v-card overflow-hidden flex flex-col">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 border-b border-[#ebebeb] bg-[#ffffff]">
        <div className="flex items-center gap-2">
          <span className="v-eyebrow text-[#171717]">▲ 3D_PITCH_WORLD</span>
          <span className="font-mono text-[11px] text-[#666666] hidden sm:inline">MCC Laws 6–8</span>
        </div>
        <div className="flex items-center gap-1">
          {([
            ["side", "Side 90°"],
            ["rear", "Rear 0°"],
            ["top", "Top"],
            ["camera", "Cam"],
          ] as Array<[CameraPreset, string]>).map(([p, lbl]) => (
            <button
              key={p}
              onClick={() => setCameraPreset(p)}
              className={`px-2 py-1 rounded-[5px] font-mono text-[11px] transition ${
                cameraPreset === p
                  ? "bg-[#171717] text-[#ffffff]"
                  : "bg-[#f4f4f5] text-[#4d4d4d] hover:text-[#171717]"
              }`}
            >
              {lbl}
            </button>
          ))}
          <button
            onClick={() => setShowUncertainty(!showUncertainty)}
            className={`ml-1 px-2 py-1 rounded-[5px] font-mono text-[11px] border transition ${
              showUncertainty
                ? "border-[#297a3a] text-[#297a3a] bg-[#297a3a]/[0.06]"
                : "border-[#ebebeb] text-[#666666] bg-[#ffffff]"
            }`}
            title="Scale joint spheres by 3D localization uncertainty (1-sigma)"
          >
            σ Spheres
          </button>
        </div>
      </div>

      <div className="relative h-[340px] w-full bg-[#fafafa]">
        <Canvas camera={{ position: [targetCenter[0], 1.45, targetCenter[2] + 4.4], fov: 44 }}>
          <color attach="background" args={["#fafafa"]} />
          <ambientLight intensity={0.85} />
          <directionalLight position={[4, 8, 5]} intensity={0.95} />
          <MCCPitchDeck camWorldPos={telemetry.camera_position_world_m} targetCenter={targetCenter} />
          <AnimatedSkeleton3D
            telemetry={telemetry}
            ghostTelemetry={ghostTelemetry}
            showUncertainty={showUncertainty}
            bowlingArm={bowlingArm}
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
        <div className="absolute bottom-2.5 left-2.5 px-2.5 py-1 rounded-[5px] bg-[#ffffff]/92 border border-[#ebebeb] font-mono text-[10px] text-[#4d4d4d] pointer-events-none">
          Green = active arm/front leg & popping crease • Black = stumps (0.711 m) & camera
          {ghostTelemetry ? " • Grey = FFC-synced Ghost" : ""}
        </div>
      </div>
    </div>
  );
};
