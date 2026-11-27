import { useSyncExternalStore } from "react";

/**
 * Tiny dependency-free hash router for Studio v5.
 *
 * Routes:
 *   #/                                 home (Command Center)
 *   #/deliveries?bowler=               ball library
 *   #/d/:id[/:tab][?f=frame]           delivery workbench (tab: story | practice | details | card)
 *   #/compare?a=&b=&ref=               compare & ghost studio (ref = archetype id or delivery id)
 *   #/spell?bowler=&session=           spell, release tunnel & workload hub   (alias: #/progress)
 *   #/practice?bowler=&d=              practice & drill lab
 *   #/upload?bowler=&mode=             new analysis: live | clip | net | dual (alias: #/new)
 *   #/card/:id                         shareable coach card
 */
export type DeliveryTab = "story" | "practice" | "details" | "card" | "lab"; // "lab" = legacy alias → details
export type UploadMode = "live" | "clip" | "net" | "dual";

export type Route =
  | { name: "home" }
  | { name: "deliveries"; bowler: string | null }
  | { name: "delivery"; id: string; tab: DeliveryTab; frame?: number | null }
  | { name: "compare"; a: string | null; b: string | null; ref?: string | null }
  | { name: "spell"; bowler: string | null; session?: string | null }
  | { name: "practice"; bowler: string | null; delivery?: string | null }
  | { name: "upload"; bowler: string | null; mode?: UploadMode }
  | { name: "card"; id: string };

/** Legacy v4 route objects still accepted by `navigate` / `routeToHash`. */
export type LegacyRoute = { name: "progress"; bowler: string | null } | { name: "new"; bowler: string | null };
export type RouteInput = Route | LegacyRoute;

const TABS: DeliveryTab[] = ["story", "practice", "details", "card", "lab"];
const MODES: UploadMode[] = ["live", "clip", "net", "dual"];

function normaliseTab(t: string | undefined): DeliveryTab {
  const tab = (t || "story") as DeliveryTab;
  if (!TABS.includes(tab)) return "story";
  return tab === "lab" ? "details" : tab;
}

export function parseHash(hash: string): Route {
  const raw = hash.replace(/^#/, "") || "/";
  const [pathPart, queryPart = ""] = raw.split("?");
  const q = new URLSearchParams(queryPart);
  const parts = pathPart.split("/").filter(Boolean);
  if (parts.length === 0) return { name: "home" };
  switch (parts[0]) {
    case "deliveries":
      return { name: "deliveries", bowler: q.get("bowler") };
    case "d": {
      const id = decodeURIComponent(parts[1] ?? "");
      if (!id) return { name: "home" };
      const f = q.get("f");
      const frame = f !== null && f !== "" && Number.isFinite(Number(f)) ? Math.max(0, Math.round(Number(f))) : null;
      return { name: "delivery", id, tab: normaliseTab(parts[2]), frame };
    }
    case "compare":
      return { name: "compare", a: q.get("a"), b: q.get("b"), ref: q.get("ref") };
    case "spell":
    case "progress":
    case "workload":
      return { name: "spell", bowler: q.get("bowler"), session: q.get("session") };
    case "practice":
      return { name: "practice", bowler: q.get("bowler"), delivery: q.get("d") };
    case "upload":
    case "new": {
      const m = q.get("mode") as UploadMode | null;
      return { name: "upload", bowler: q.get("bowler"), mode: m && MODES.includes(m) ? m : "clip" };
    }
    case "card": {
      const id = decodeURIComponent(parts[1] ?? "");
      return id ? { name: "card", id } : { name: "home" };
    }
    default:
      return { name: "home" };
  }
}

const qs = (o: Record<string, string | number | null | undefined>) => {
  const p = new URLSearchParams();
  Object.entries(o).forEach(([k, v]) => {
    if (v !== null && v !== undefined && v !== "") p.set(k, String(v));
  });
  const s = p.toString();
  return s ? `?${s}` : "";
};

export function routeToHash(r: RouteInput): string {
  switch (r.name) {
    case "home":
      return "#/";
    case "deliveries":
      return `#/deliveries${qs({ bowler: r.bowler })}`;
    case "delivery": {
      const tab = normaliseTab(r.tab);
      return `#/d/${encodeURIComponent(r.id)}${tab === "story" ? "" : `/${tab}`}${qs({ f: r.frame ?? null })}`;
    }
    case "compare":
      return `#/compare${qs({ a: r.a, b: r.b, ref: r.ref ?? null })}`;
    case "spell":
      return `#/spell${qs({ bowler: r.bowler, session: r.session ?? null })}`;
    case "progress":
      return `#/spell${qs({ bowler: r.bowler })}`;
    case "practice":
      return `#/practice${qs({ bowler: r.bowler, d: r.delivery ?? null })}`;
    case "upload":
      return `#/upload${qs({ bowler: r.bowler, mode: r.mode && r.mode !== "clip" ? r.mode : null })}`;
    case "new":
      return `#/upload${qs({ bowler: r.bowler })}`;
    case "card":
      return `#/card/${encodeURIComponent(r.id)}`;
  }
}

export function navigate(r: RouteInput | string) {
  const h = typeof r === "string" ? r : routeToHash(r);
  if (window.location.hash !== h) window.location.hash = h;
  window.scrollTo({ top: 0 });
}

function subscribe(cb: () => void) {
  window.addEventListener("hashchange", cb);
  return () => window.removeEventListener("hashchange", cb);
}

export function useRoute(): Route {
  const hash = useSyncExternalStore(subscribe, () => window.location.hash, () => "");
  return parseHash(hash);
}

// ── Current bowler (persisted; scopes Home / Deliveries / Spell / Practice) ──
const BOWLER_KEY = "spa.bowlerId";
const bowlerListeners = new Set<() => void>();

export function getCurrentBowlerId(): string | null {
  try {
    return window.localStorage.getItem(BOWLER_KEY);
  } catch {
    return null;
  }
}

export function setCurrentBowlerId(id: string | null) {
  try {
    if (id) window.localStorage.setItem(BOWLER_KEY, id);
    else window.localStorage.removeItem(BOWLER_KEY);
  } catch {
    /* ignore */
  }
  bowlerListeners.forEach((l) => l());
}

export function useCurrentBowlerId(): string | null {
  return useSyncExternalStore(
    (cb) => {
      bowlerListeners.add(cb);
      return () => bowlerListeners.delete(cb);
    },
    getCurrentBowlerId,
    () => null
  );
}

// ── Legacy Pro Lab switch (v4). v5 has no Pro switch; kept so older components compile. ──
const PRO_LAB_KEY = "spa.proLab";
const proLabListeners = new Set<() => void>();

export function getProLab(): boolean {
  try {
    return window.localStorage.getItem(PRO_LAB_KEY) === "1";
  } catch {
    return false;
  }
}

export function setProLab(v: boolean) {
  try {
    window.localStorage.setItem(PRO_LAB_KEY, v ? "1" : "0");
  } catch {
    /* ignore */
  }
  proLabListeners.forEach((l) => l());
}

export function useProLab(): boolean {
  return useSyncExternalStore(
    (cb) => {
      proLabListeners.add(cb);
      return () => proLabListeners.delete(cb);
    },
    getProLab,
    () => false
  );
}
