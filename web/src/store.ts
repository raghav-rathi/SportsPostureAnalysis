import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

export type CameraPreset = "side" | "rear" | "top" | "camera";
export type RightTab = "metrics" | "calibration" | "compare" | "dual" | "coach";

interface StudioState {
  // Transient high-frequency playback bus (read via subscribeWithSelector in Canvas / Three.js / uPlot!)
  currentFrame: number;
  isPlaying: boolean;
  playbackRate: number;

  // Studio UI controls
  showSkeleton: boolean;
  showWireframe: boolean;
  showUncertaintySpheres: boolean;
  calibrationMode: boolean;
  activeLandmarkKey: string;
  draftLandmarks: Record<string, [number, number]>;
  previewWireframe: Record<string, [number, number]> | null;
  previewReprojErrPx: number | null;
  cameraPreset: CameraPreset;
  cameraResetNonce: number;
  activeTab: RightTab;
  ghostDeliveryId: string | null;

  setCurrentFrame: (frame: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setPlaybackRate: (rate: number) => void;
  setShowSkeleton: (v: boolean) => void;
  setShowWireframe: (v: boolean) => void;
  setShowUncertaintySpheres: (v: boolean) => void;
  setCalibrationMode: (v: boolean) => void;
  setActiveLandmarkKey: (k: string) => void;
  setDraftLandmark: (k: string, pt: [number, number]) => void;
  clearDraftLandmarks: () => void;
  setPreviewCalibration: (wf: Record<string, [number, number]> | null, err: number | null) => void;
  setCameraPreset: (p: CameraPreset) => void;
  setActiveTab: (t: RightTab) => void;
  setGhostDeliveryId: (id: string | null) => void;
}

export const useStudioStore = create<StudioState>()(
  subscribeWithSelector((set) => ({
    currentFrame: 0,
    isPlaying: false,
    playbackRate: 0.25,

    showSkeleton: true,
    showWireframe: true,
    showUncertaintySpheres: true,
    calibrationMode: false,
    activeLandmarkKey: "bowling_crease_off",
    draftLandmarks: {},
    previewWireframe: null,
    previewReprojErrPx: null,
    cameraPreset: "side",
    cameraResetNonce: 0,
    activeTab: "metrics",
    ghostDeliveryId: null,

    setCurrentFrame: (frame) => set({ currentFrame: Math.max(0, Math.round(frame)) }),
    setIsPlaying: (isPlaying) => set({ isPlaying }),
    setPlaybackRate: (playbackRate) => set({ playbackRate }),
    setShowSkeleton: (showSkeleton) => set({ showSkeleton }),
    setShowWireframe: (showWireframe) => set({ showWireframe }),
    setShowUncertaintySpheres: (showUncertaintySpheres) => set({ showUncertaintySpheres }),
    setCalibrationMode: (calibrationMode) => set({ calibrationMode }),
    setActiveLandmarkKey: (activeLandmarkKey) => set({ activeLandmarkKey }),
    setDraftLandmark: (k, pt) =>
      set((s) => ({ draftLandmarks: { ...s.draftLandmarks, [k]: pt } })),
    clearDraftLandmarks: () => set({ draftLandmarks: {}, previewWireframe: null, previewReprojErrPx: null }),
    setPreviewCalibration: (previewWireframe, previewReprojErrPx) =>
      set({ previewWireframe, previewReprojErrPx }),
    setCameraPreset: (cameraPreset) =>
      set((s) => ({ cameraPreset, cameraResetNonce: s.cameraResetNonce + 1 })),
    setActiveTab: (activeTab) => set({ activeTab }),
    setGhostDeliveryId: (ghostDeliveryId) => set({ ghostDeliveryId }),
  }))
);
