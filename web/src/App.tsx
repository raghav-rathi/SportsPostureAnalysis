import React, { Suspense, lazy, useState } from "react";
import { useRoute } from "./router";
import { TopNav } from "./components/TopNav";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { HowToFilmSheet } from "./components/story/HowToFilmSheet";
import { Spinner } from "./components/story/ui";
import { HomeView } from "./views/HomeView";
import { DeliveriesView } from "./views/DeliveriesView";

// Heavy screens are code-split; Home + Deliveries stay in the initial bundle.
const DeliveryView = lazy(() => import("./views/DeliveryView").then((m) => ({ default: m.DeliveryView })));
const CompareView = lazy(() => import("./views/CompareView").then((m) => ({ default: m.CompareView })));
const SpellWorkloadView = lazy(() => import("./views/SpellWorkloadView").then((m) => ({ default: m.SpellWorkloadView })));
const PracticeView = lazy(() => import("./views/PracticeView").then((m) => ({ default: m.PracticeView })));
const UploadFlowView = lazy(() => import("./views/UploadFlowView").then((m) => ({ default: m.UploadFlowView })));

// Warm the code-split chunks on idle so nothing feels "hidden" behind a load.
if (typeof window !== "undefined") {
  const warm = () => {
    void import("./views/DeliveryView");
    void import("./views/CompareView");
    void import("./views/SpellWorkloadView");
    void import("./views/PracticeView");
    void import("./views/UploadFlowView");
  };
  const ric = (window as any).requestIdleCallback as ((cb: () => void) => void) | undefined;
  if (ric) ric(warm);
  else window.setTimeout(warm, 2500);
}

const Loading: React.FC = () => (
  <div className="flex items-center gap-2 text-[13px] text-[#666666] py-16 justify-center">
    <Spinner /> Loading…
  </div>
);

/**
 * Studio v5 shell. Routes: home · deliveries · delivery (story|practice|details|card) · compare ·
 * spell (alias progress) · practice · upload (alias new; mode live|clip|net|dual) · card.
 */
export function App() {
  const route = useRoute();
  const [filmGuideOpen, setFilmGuideOpen] = useState(false);
  const openFilmGuide = () => setFilmGuideOpen(true);
  const wide = route.name === "delivery" || route.name === "compare" || route.name === "spell";

  return (
    <div className="min-h-screen bg-[#fafafa] text-[#171717] flex flex-col">
      <TopNav route={route} onOpenFilmGuide={openFilmGuide} />
      <main
        className={`flex-1 w-full min-w-0 overflow-x-clip mx-auto px-4 md:px-6 py-6 pb-24 md:pb-6 ${
          wide ? "max-w-[1600px]" : "max-w-[1440px]"
        }`}
      >
        <ErrorBoundary resetKey={window.location.hash}>
          <Suspense fallback={<Loading />}>
            {route.name === "home" && <HomeView onOpenFilmGuide={openFilmGuide} />}
            {route.name === "deliveries" && <DeliveriesView bowler={route.bowler} />}
            {route.name === "delivery" && (
              <DeliveryView key={route.id} id={route.id} tab={route.tab} frame={route.frame ?? null} onOpenFilmGuide={openFilmGuide} />
            )}
            {route.name === "card" && <DeliveryView key={route.id} id={route.id} tab="card" onOpenFilmGuide={openFilmGuide} />}
            {route.name === "compare" && <CompareView a={route.a} b={route.b} refId={route.ref ?? null} />}
            {route.name === "spell" && <SpellWorkloadView bowlerId={route.bowler} sessionId={route.session ?? null} />}
            {route.name === "practice" && <PracticeView bowlerId={route.bowler} deliveryId={route.delivery ?? null} />}
            {route.name === "upload" && <UploadFlowView initialBowlerId={route.bowler} mode={route.mode ?? "clip"} />}
          </Suspense>
        </ErrorBoundary>
      </main>
      <HowToFilmSheet open={filmGuideOpen} onClose={() => setFilmGuideOpen(false)} />
    </div>
  );
}

export default App;
