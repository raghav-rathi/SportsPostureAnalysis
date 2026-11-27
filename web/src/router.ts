import { useSyncExternalStore } from "react";

/**
 * Tiny dependency-free hash router for Studio v4.
 *
 * Routes:
 *   #/                       home
 *   #/deliveries             deliveries library
 *   #/d/:id[/:tab]           delivery (tab: story | practice | details | lab | card)
 *   #/compare?a=&b=          compare 2 balls
 *   #/progress?bowler=       progress
 *   #/new?bowler=            guided new analysis
 */
export type DeliveryTab = "story" | "practice" | "details" | "lab" | "card";

export type Route =
  | { name: "home" }
  | { name: "deliveries"; bowler: string | null }
  | { name: "delivery"; id: string; tab: DeliveryTab }
  | { name: "compare"; a: string | null; b: string | null }
  | { name: "progress"; bowler: string | null }
  | { name: "new"; bowler: string | null };

const TABS: DeliveryTab[] = ["story", "practice", "details", "lab", "card"];

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
      const tab = (parts[2] as DeliveryTab) || "story";
      return { name: "delivery", id, tab: TABS.includes(tab) ? tab : "story" };
    }
    case "compare":
      return { name: "compare", a: q.get("a"), b: q.get("b") };
    case "progress":
      return { name: "progress", bowler: q.get("bowler") };
    case "new":
      return { name: "new", bowler: q.get("bowler") };
    default:
      return { name: "home" };
  }
}

export function routeToHash(r: Route): string {
  const qs = (o: Record<string, string | null | undefined>) => {
    const p = new URLSearchParams();
    Object.entries(o).forEach(([k, v]) => v && p.set(k, v));
    const s = p.toString();
    return s ? `?${s}` : "";
  };
  switch (r.name) {
    case "home":
      return "#/";
    case "deliveries":
      return `#/deliveries${qs({ bowler: r.bowler })}`;
    case "delivery":
      return `#/d/${encodeURIComponent(r.id)}${r.tab === "story" ? "" : `/${r.tab}`}`;
    case "compare":
      return `#/compare${qs({ a: r.a, b: r.b })}`;
    case "progress":
      return `#/progress${qs({ bowler: r.bowler })}`;
    case "new":
      return `#/new${qs({ bowler: r.bowler })}`;
  }
}

export function navigate(r: Route | string) {
  const h = typeof r === "string" ? r : routeToHash(r);
  if (window.location.hash !== h) {
    window.location.hash = h;
  }
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

// ── Pro Lab opt-in switch (persisted) ───────────────────────────────────────
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
