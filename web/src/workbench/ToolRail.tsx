/**
 * ToolRail: compact Teenage-Engineering-inspired instrument strip for the magenta telestration toolkit.
 * Single-row horizontal toolbar (~40 px tall) that sits directly above the video/split stage with zero dead
 * vertical whitespace, scrolling cleanly horizontally on narrow mobile screens.
 * Shortcuts (ignored while typing): 1–0 select tools, Esc → select, ⌘/Ctrl+Z undo, P pin toggle, +/- zoom.
 * (J/K/L and , . are reserved for the playhead shuttle.)
 */
import React, { useEffect } from "react";
import {
  ArrowDownToLine,
  ArrowUpRight,
  Circle,
  Eraser,
  Focus,
  Minus,
  MousePointer2,
  PenLine,
  Pin,
  PinOff,
  Ruler,
  Triangle,
  Type,
  Undo2,
  ZoomIn,
  ZoomOut,
  type LucideIcon,
} from "lucide-react";
import type { TelestratorTool } from "../types";

export interface ToolRailProps {
  activeTool: TelestratorTool;
  onSelectTool: (t: TelestratorTool) => void;
  pinAllFrames: boolean;
  onTogglePinAll: () => void;
  canUndo: boolean;
  onUndo: () => void;
  onClearFrame: () => void;
  zoom: number;
  onZoomChange: (z: number) => void;
  shapeCount: number;
}

type ToolDef = { id: TelestratorTool; label: string; hint: string; key: string; Icon: LucideIcon };

export const TOOL_DEFS: ToolDef[] = [
  { id: "cursor", label: "Select", hint: "Tap a joint for its live angle; drag to pan when zoomed", key: "1", Icon: MousePointer2 },
  { id: "line", label: "Line", hint: "Drag a line — shows its angle to horizontal", key: "2", Icon: Minus },
  { id: "angle", label: "Angle", hint: "Tap 3 points — snaps to joints and shows the measured 3D angle", key: "3", Icon: Triangle },
  { id: "plumb", label: "Plumb", hint: "True vertical from calibration — drag to read lean", key: "4", Icon: ArrowDownToLine },
  { id: "circle", label: "Circle", hint: "Drag from centre", key: "5", Icon: Circle },
  { id: "arrow", label: "Arrow", hint: "Drag a direction cue", key: "6", Icon: ArrowUpRight },
  { id: "freehand", label: "Draw", hint: "Freehand pen (smoothed)", key: "7", Icon: PenLine },
  { id: "ruler", label: "Ruler", hint: "Distance — metres ± σ when the clip is calibrated", key: "8", Icon: Ruler },
  { id: "text", label: "Text", hint: "Tap to pin a label", key: "9", Icon: Type },
  { id: "spotlight", label: "Spot", hint: "Drag a circle — dims everything else", key: "0", Icon: Focus },
];

const ZOOM_MIN = 1;
const ZOOM_MAX = 4;
const clampZoom = (z: number) => Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, Math.round(z * 100) / 100));

function isTypingTarget(t: EventTarget | null): boolean {
  if (!t || !(t instanceof HTMLElement)) return false;
  if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement) return true;
  return t.isContentEditable;
}

const ToolChip: React.FC<{
  active?: boolean;
  disabled?: boolean;
  title: string;
  onClick: () => void;
  children: React.ReactNode;
  label?: string;
}> = ({ active, disabled, title, onClick, children, label }) => (
  <button
    type="button"
    title={title}
    aria-label={title}
    aria-pressed={active}
    disabled={disabled}
    onClick={onClick}
    className={`shrink-0 inline-flex items-center justify-center gap-1 h-7 ${
      label ? "px-2" : "w-7"
    } rounded-[6px] transition-[background-color,box-shadow,color] duration-150 ease-out disabled:opacity-35 disabled:cursor-not-allowed ${
      active
        ? "bg-[#171717] text-white shadow-[inset_0_-2px_0_0_#C026D3]"
        : "bg-white text-[#4d4d4d] shadow-[0_0_0_1px_#ebebeb] hover:text-[#171717] hover:shadow-[0_0_0_1px_#c9c9c9]"
    }`}
  >
    {children}
    {label && (
      <span className={`font-mono text-[10px] leading-none tracking-[0.02em] ${active ? "text-white" : "text-[#4d4d4d]"}`}>
        {label}
      </span>
    )}
  </button>
);

const Divider = () => <div className="shrink-0 bg-[#e5e5e5] w-px h-5 mx-0.5" />;

export const ToolRail: React.FC<ToolRailProps> = ({
  activeTool,
  onSelectTool,
  pinAllFrames,
  onTogglePinAll,
  canUndo,
  onUndo,
  onClearFrame,
  zoom,
  onZoomChange,
  shapeCount,
}) => {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || isTypingTarget(e.target)) return;
      if ((e.metaKey || e.ctrlKey) && (e.key === "z" || e.key === "Z") && !e.shiftKey) {
        if (canUndo) {
          e.preventDefault();
          onUndo();
        }
        return;
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const def = TOOL_DEFS.find((d) => d.key === e.key);
      if (def) {
        e.preventDefault();
        onSelectTool(def.id);
      } else if (e.key === "Escape") {
        onSelectTool("cursor");
      } else if (e.key === "p" || e.key === "P") {
        onTogglePinAll();
      } else if (e.key === "+" || e.key === "=") {
        onZoomChange(clampZoom(zoom * 1.25));
      } else if (e.key === "-" || e.key === "_") {
        onZoomChange(clampZoom(zoom / 1.25));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canUndo, onUndo, onSelectTool, onTogglePinAll, onZoomChange, zoom]);

  return (
    <div
      role="toolbar"
      aria-label="Telestration tools"
      aria-orientation="horizontal"
      className="flex flex-row items-center gap-1 px-2 py-1.5 w-full overflow-x-auto rounded-[8px] bg-[#fafafa] shadow-[0_0_0_1px_#ebebeb] [scrollbar-width:none]"
    >
      <span
        className="shrink-0 inline-flex items-center gap-1 pr-1 font-mono text-[9.5px] tracking-[0.06em] uppercase text-[#8f8f8f]"
        title="Magenta = your drawings (never a measurement)"
      >
        <span className="w-1.5 h-1.5 rounded-full" style={{ background: "#C026D3" }} />
        Draw
      </span>

      {TOOL_DEFS.map(({ id, label, hint, key, Icon }) => (
        <ToolChip
          key={id}
          active={activeTool === id}
          title={`${label} (${key}) — ${hint}`}
          onClick={() => onSelectTool(id)}
          label={label}
        >
          <Icon size={13} strokeWidth={1.85} />
        </ToolChip>
      ))}

      <Divider />

      <ToolChip
        active={pinAllFrames}
        title={pinAllFrames ? "New drawings stay on every frame (P)" : "New drawings stay on this frame only (P)"}
        onClick={onTogglePinAll}
        label={pinAllFrames ? "All" : "Pin"}
      >
        {pinAllFrames ? <Pin size={12.5} strokeWidth={1.85} /> : <PinOff size={12.5} strokeWidth={1.85} />}
      </ToolChip>
      <ToolChip disabled={!canUndo} title="Undo (⌘Z)" onClick={onUndo}>
        <Undo2 size={13} strokeWidth={1.85} />
      </ToolChip>
      <ToolChip disabled={shapeCount === 0} title="Clear drawings on this frame" onClick={onClearFrame}>
        <Eraser size={13} strokeWidth={1.85} />
      </ToolChip>

      <Divider />

      <div className="flex items-center gap-0.5 shrink-0">
        <ToolChip disabled={zoom <= ZOOM_MIN} title="Zoom out (−)" onClick={() => onZoomChange(clampZoom(zoom / 1.25))}>
          <ZoomOut size={13} strokeWidth={1.85} />
        </ToolChip>
        <button
          type="button"
          onClick={() => onZoomChange(1)}
          title="Reset zoom"
          className="shrink-0 font-mono text-[10px] tabular-nums text-[#4d4d4d] hover:text-[#171717] px-1 py-0.5 rounded"
        >
          {zoom.toFixed(1)}×
        </button>
        <ToolChip disabled={zoom >= ZOOM_MAX} title="Zoom in (+)" onClick={() => onZoomChange(clampZoom(zoom * 1.25))}>
          <ZoomIn size={13} strokeWidth={1.85} />
        </ToolChip>
      </div>

      {shapeCount > 0 && (
        <span className="shrink-0 pl-1 font-mono text-[10px] tabular-nums text-[#8f8f8f]" title="Drawings on this delivery">
          {shapeCount}✎
        </span>
      )}
    </div>
  );
};

export default ToolRail;
