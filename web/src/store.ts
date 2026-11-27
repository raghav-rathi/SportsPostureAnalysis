import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";
import type { TelestratorTool } from "./types";

export type CameraPreset = "side" | "rear" | "top" | "camera";
export type TopView = "analysis" | "compare" | "guide" | "report";
export type AnalysisMode = "coach" | "lab";

interface StudioState {
  // High-frequency transient playback bus (60-120 Hz canvas / 3D / uPlot sync without React re-renders!)
  currentFrame: number;
  isPlaying: boolean;
  playbackRate: number;

  // Navigation & progressive disclosure
  topView: TopView;
  analysisMode: AnalysisMode;

  // Video & 3D overlays
  showSkeleton: boolean;
  showWireframe: boolean;
  showUncertaintySpheres: boolean;
  cameraPreset: CameraPreset;
  cameraResetNonce: number;
  ghostDeliveryId: string | null;

  // Interactive 4-Click Calibration & Bowler Re-Track modes
  calibrationMode: boolean;
  retrackMode: boolean;
  activeLandmarkKey: string;
  draftLandmarks: Record<string, [number, number]>;
  previewWireframe: Record<string, [number, number]> | null;
  previewReprojErrPx: number | null;

  // Interactive Telestrator Drawing Tool
  activeTool: TelestratorTool;

  setCurrentFrame: (frame: number) => void;
  setIsPlaying: (playing: boolean) => void;
  setPlaybackRate: (rate: number) => void;
  setTopView: (v: TopView) => void;
  setAnalysisMode: (m: AnalysisMode) => void;
  setShowSkeleton: (v: boolean) => void;
  setShowWireframe: (v: boolean) => void;
  setShowUncertaintySpheres: (v: boolean) => void;
  setCalibrationMode: (v: boolean) => void;
  setRetrackMode: (v: boolean) => void;
  setActiveLandmarkKey: (k: string) => void;
  setDraftLandmark: (k: string, pt: [number, number]) => void;
  clearDraftLandmarks: () => void;
  setPreviewCalibration: (wf: Record<string, [number, number]> | null, err: number | null) => void;
  setCameraPreset: (p: CameraPreset) => void;
  setGhostDeliveryId: (id: string | null) => void;
  setActiveTool: (t: TelestratorTool) => void;
}

export const useStudioStore = create<StudioState>()(
  subscribeWithSelector((set) => ({
    currentFrame: 0,
    isPlaying: false,
    playbackRate: 0.25,

    topView: "analysis",
    analysisMode: "coach",

    showSkeleton: true,
    showWireframe: true,
    showUncertaintySpheres: true,
    cameraPreset: "side",
    cameraResetNonce: 0,
    ghostDeliveryId: null,

    calibrationMode: false,
    retrackMode: false,
    activeLandmarkKey: "bowling_crease_off",
    draftLandmarks: {},
    previewWireframe: null,
    previewReprojErrPx: null,

    activeTool: "cursor",

    setCurrentFrame: (frame) => set({ currentFrame: Math.max(0, Math.round(frame)) }),
    setIsPlaying: (isPlaying) => set({ isPlaying }),
    setPlaybackRate: (playbackRate) => set({ playbackRate }),
    setTopView: (topView) => set({ topView }),
    setAnalysisMode: (analysisMode) => set({ analysisMode }),
    setShowSkeleton: (showSkeleton) => set({ showSkeleton }),
    setShowWireframe: (showWireframe) => set({ showWireframe }),
    setShowUncertaintySpheres: (showUncertaintySpheres) => set({ showUncertaintySpheres }),
    setCalibrationMode: (calibrationMode) =>
      set((s) => ({
        calibrationMode,
        retrackMode: calibrationMode ? false : s.retrackMode,
        activeTool: calibrationMode ? "cursor" : s.activeTool,
      })),
    setRetrackMode: (retrackMode) =>
      set((s) => ({
        retrackMode,
        calibrationMode: retrackMode ? false : s.calibrationMode,
        activeTool: retrackMode ? "cursor" : s.activeTool,
      })),
    setActiveLandmarkKey: (activeLandmarkKey) => set({ activeLandmarkKey }),
    setDraftLandmark: (k, pt) => set((s) => ({ draftLandmarks: { ...s.draftLandmarks, [k]: pt } })),
    clearDraftLandmarks: () => set({ draftLandmarks: {}, previewWireframe: null, previewReprojErrPx: null }),
    setPreviewCalibration: (previewWireframe, previewReprojErrPx) => set({ previewWireframe, previewReprojErrPx }),
    setCameraPreset: (cameraPreset) => set((s) => ({ cameraPreset, cameraResetNonce: s.cameraResetNonce + 1 })),
    setGhostDeliveryId: (ghostDeliveryId) => set({ ghostDeliveryId }),
    setActiveTool: (activeTool) =>
      set((s) => ({
        activeTool,
        calibrationMode: activeTool !== "cursor" ? false : s.calibrationMode,
        retrackMode: activeTool !== "cursor" ? false : s.retrackMode,
      })),
  }))
);
