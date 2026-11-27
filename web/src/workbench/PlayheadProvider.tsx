/**
 * PlayheadProvider: the one clock behind every delivery visual (docs/research_app_landscape_v5.md §6.1 "One clock").
 *
 * - Smooth requestAnimationFrame playback that advances at `fps * rate` frames per second of wall-clock time
 *   (so "0.25× of 240 fps" puts 60 capture frames per second on screen), with a fractional accumulator so the
 *   integer `frame` never drifts or stutters.
 * - Optional `loopRange` [lo, hi]: playback loops seamlessly inside it. Without one, playback wraps the whole clip.
 * - Global keyboard shortcuts (ignored while typing in inputs / textareas / selects / contentEditable):
 *     Space            play / pause
 *     , or ArrowLeft   step −1 frame          (Shift: −5)
 *     . or ArrowRight  step +1 frame          (Shift: +5)
 *     J / K / L        shuttle: reverse (press again = faster) / pause / forward (press again = faster)
 */
import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

export type PlaybackRate = 0.1 | 0.25 | 0.5 | 1;

export const PLAYBACK_RATES: PlaybackRate[] = [0.1, 0.25, 0.5, 1];

export interface PlayheadContextValue {
  frame: number;
  setFrame: (f: number | ((prev: number) => number)) => void;
  numFrames: number;
  fps: number;
  playing: boolean;
  setPlaying: (p: boolean | ((prev: boolean) => boolean)) => void;
  rate: PlaybackRate;
  setRate: (r: PlaybackRate) => void;
  loopRange: [number, number] | null;
  setLoopRange: (r: [number, number] | null) => void;
  step: (delta: number) => void;
  timeRelFfcMs: number;
  seekToEventFrame: (f: number, pause?: boolean) => void;
  /** Shuttle direction (+1 forward, −1 reverse from `J`). Extra field; safe to ignore. */
  direction?: 1 | -1;
}

const noop = () => {};

/** Static fallback so components (e.g. PitchStage3D with an explicit `frame`) can render without a provider. */
const DEFAULT_PLAYHEAD: PlayheadContextValue = {
  frame: 0,
  setFrame: noop,
  numFrames: 1,
  fps: 30,
  playing: false,
  setPlaying: noop,
  rate: 0.25,
  setRate: noop,
  loopRange: null,
  setLoopRange: noop,
  step: noop,
  timeRelFfcMs: 0,
  seekToEventFrame: noop,
  direction: 1,
};

const PlayheadContext = createContext<PlayheadContextValue | null>(null);

function isTypingTarget(t: EventTarget | null): boolean {
  if (!t || !(t instanceof HTMLElement)) return false;
  if (t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement) return true;
  return t.isContentEditable;
}

const clampInt = (v: number, lo: number, hi: number) => Math.min(Math.max(Math.round(v), lo), hi);

export const PlayheadProvider: React.FC<{
  numFrames: number;
  fps: number;
  timeRelFfcMs: number[];
  initialFrame?: number;
  children: React.ReactNode;
  /** Extra (optional): start playing on mount, e.g. the Home hero. */
  autoPlay?: boolean;
  /** Extra (optional): initial rate (default 0.25×). */
  initialRate?: PlaybackRate;
  /** Extra (optional): disable global keyboard shortcuts when several providers share a page. Default true. */
  keyboard?: boolean;
}> = ({ numFrames, fps, timeRelFfcMs, initialFrame = 0, children, autoPlay = false, initialRate = 0.25, keyboard = true }) => {
  const n = Math.max(1, Math.floor(numFrames || 1));
  const safeFps = Number.isFinite(fps) && fps > 0 ? fps : 30;

  const [frame, setFrameState] = useState<number>(() => clampInt(initialFrame, 0, n - 1));
  const [playing, setPlayingState] = useState<boolean>(autoPlay);
  const [rate, setRateState] = useState<PlaybackRate>(initialRate);
  const [loopRange, setLoopRangeState] = useState<[number, number] | null>(null);
  const [direction, setDirection] = useState<1 | -1>(1);

  // Fractional playhead position (frames). The integer `frame` is derived from it.
  const posRef = useRef<number>(frame);
  const frameRef = useRef<number>(frame);
  const playingRef = useRef<boolean>(playing);
  frameRef.current = frame;
  playingRef.current = playing;

  // Re-clamp when the clip changes length.
  useEffect(() => {
    setFrameState((f) => {
      const c = clampInt(f, 0, n - 1);
      posRef.current = c;
      return c;
    });
    setLoopRangeState((r) => (r ? [clampInt(r[0], 0, n - 1), clampInt(r[1], 0, n - 1)] : r));
  }, [n]);

  const setFrame = useCallback(
    (f: number | ((prev: number) => number)) => {
      const next = typeof f === "function" ? f(frameRef.current) : f;
      if (!Number.isFinite(next)) return;
      const c = clampInt(next, 0, n - 1);
      posRef.current = c;
      frameRef.current = c;
      setFrameState(c);
    },
    [n]
  );

  const setPlaying = useCallback((p: boolean | ((prev: boolean) => boolean)) => {
    const next = typeof p === "function" ? p(playingRef.current) : p;
    playingRef.current = next;
    if (!next) setDirection(1);
    setPlayingState(next);
  }, []);

  const setRate = useCallback((r: PlaybackRate) => setRateState(r), []);

  const setLoopRange = useCallback(
    (r: [number, number] | null) => {
      if (!r) {
        setLoopRangeState(null);
        return;
      }
      const lo = clampInt(Math.min(r[0], r[1]), 0, n - 1);
      const hi = clampInt(Math.max(r[0], r[1]), 0, n - 1);
      setLoopRangeState(hi > lo ? [lo, hi] : null);
    },
    [n]
  );

  const step = useCallback(
    (delta: number) => {
      setPlaying(false);
      setFrame((f) => f + delta);
    },
    [setFrame, setPlaying]
  );

  const seekToEventFrame = useCallback(
    (f: number, pause: boolean = true) => {
      if (pause) setPlaying(false);
      setFrame(f);
    },
    [setFrame, setPlaying]
  );

  // ── rAF clock ─────────────────────────────────────────────────────────────
  const rateRef = useRef(rate);
  const loopRef = useRef(loopRange);
  const dirRef = useRef(direction);
  rateRef.current = rate;
  loopRef.current = loopRange;
  dirRef.current = direction;

  useEffect(() => {
    if (!playing || n <= 1) return;
    let raf = 0;
    let last = performance.now();
    posRef.current = frameRef.current;
    const tick = (now: number) => {
      const dt = Math.min(0.1, Math.max(0, (now - last) / 1000)); // clamp tab-switch jumps
      last = now;
      const lr = loopRef.current;
      const lo = lr ? lr[0] : 0;
      const hi = lr ? lr[1] : n - 1;
      const span = Math.max(1, hi - lo + 1);
      let pos = posRef.current + dirRef.current * dt * safeFps * rateRef.current;
      // If the user scrubbed outside the loop while playing, re-enter it.
      if (lr && (pos < lo - 1 || pos > hi + 1)) pos = dirRef.current > 0 ? lo : hi;
      if (pos >= hi + 1) pos = lo + ((pos - lo) % span);
      else if (pos < lo) pos = hi + 1 - ((lo - pos) % span);
      posRef.current = pos;
      const fi = clampInt(Math.floor(pos), lo, hi);
      if (fi !== frameRef.current) {
        frameRef.current = fi;
        setFrameState(fi);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, n, safeFps]);

  // ── Keyboard shortcuts ────────────────────────────────────────────────────
  useEffect(() => {
    if (!keyboard) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
      if (isTypingTarget(e.target)) return;
      const big = e.shiftKey ? 5 : 1;
      const k = e.key;
      if (e.code === "Space" || k === " ") {
        e.preventDefault();
        setDirection(1);
        setPlaying((p) => !p);
      } else if (k === "," || k === "<" || k === "ArrowLeft") {
        e.preventDefault();
        step(-big);
      } else if (k === "." || k === ">" || k === "ArrowRight") {
        e.preventDefault();
        step(big);
      } else if (k === "k" || k === "K") {
        e.preventDefault();
        setPlaying(false);
      } else if (k === "l" || k === "L" || k === "j" || k === "J") {
        e.preventDefault();
        const want: 1 | -1 = k === "l" || k === "L" ? 1 : -1;
        if (playingRef.current && dirRef.current === want) {
          // Same direction again → next faster detent.
          const i = PLAYBACK_RATES.indexOf(rateRef.current);
          setRateState(PLAYBACK_RATES[Math.min(PLAYBACK_RATES.length - 1, i + 1)]);
        }
        setDirection(want);
        dirRef.current = want;
        playingRef.current = true;
        setPlayingState(true);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [keyboard, setPlaying, step]);

  const tRel = timeRelFfcMs?.[frame];
  const value = useMemo<PlayheadContextValue>(
    () => ({
      frame,
      setFrame,
      numFrames: n,
      fps: safeFps,
      playing,
      setPlaying,
      rate,
      setRate,
      loopRange,
      setLoopRange,
      step,
      timeRelFfcMs: Number.isFinite(tRel) ? (tRel as number) : 0,
      seekToEventFrame,
      direction,
    }),
    [frame, setFrame, n, safeFps, playing, setPlaying, rate, setRate, loopRange, setLoopRange, step, tRel, seekToEventFrame, direction]
  );

  return <PlayheadContext.Provider value={value}>{children}</PlayheadContext.Provider>;
};

/** The shared clock. Outside a provider it returns an inert frame-0 clock (never throws). */
export function usePlayhead(): PlayheadContextValue {
  return useContext(PlayheadContext) ?? DEFAULT_PLAYHEAD;
}

/** `null` when rendered outside a PlayheadProvider. */
export function usePlayheadOptional(): PlayheadContextValue | null {
  return useContext(PlayheadContext);
}
