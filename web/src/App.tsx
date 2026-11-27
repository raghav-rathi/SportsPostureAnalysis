import React, { useState } from "react";
import { useRoute } from "./router";
import { TopNav } from "./components/TopNav";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { HowToFilmSheet } from "./components/story/HowToFilmSheet";
import { HomeView } from "./views/HomeView";
import { DeliveriesView } from "./views/DeliveriesView";
import { DeliveryView } from "./views/DeliveryView";
import { CompareView } from "./views/CompareView";
import { ProgressView } from "./views/ProgressView";
import { UploadFlowView } from "./views/UploadFlowView";

/**
 * Studio v4 shell: calm 3-item nav (Home · Deliveries · Progress) + New analysis,
 * hash routes, and the How-to-film side sheet. All lab tooling lives behind Pro Lab
 * inside the Delivery page.
 */
export function App() {
  const route = useRoute();
  const [filmGuideOpen, setFilmGuideOpen] = useState(false);
  const openFilmGuide = () => setFilmGuideOpen(true);
  const wide = route.name === "delivery" && route.tab === "lab";

  return (
    <div className="min-h-screen bg-[#fafafa] text-[#171717] flex flex-col">
      <TopNav route={route} onOpenFilmGuide={openFilmGuide} />
      <main className={`flex-1 w-full min-w-0 overflow-x-clip mx-auto px-4 md:px-6 py-6 ${wide ? "max-w-[1440px]" : "max-w-[1320px]"}`}>
        <ErrorBoundary resetKey={window.location.hash}>
        {route.name === "home" && <HomeView onOpenFilmGuide={openFilmGuide} />}
        {route.name === "deliveries" && <DeliveriesView bowler={route.bowler} />}
        {route.name === "delivery" && <DeliveryView key={route.id} id={route.id} tab={route.tab} onOpenFilmGuide={openFilmGuide} />}
        {route.name === "compare" && <CompareView a={route.a} b={route.b} />}
        {route.name === "progress" && <ProgressView bowlerId={route.bowler} />}
        {route.name === "new" && <UploadFlowView initialBowlerId={route.bowler} />}
        </ErrorBoundary>
      </main>
      <HowToFilmSheet open={filmGuideOpen} onClose={() => setFilmGuideOpen(false)} />
    </div>
  );
}

export default App;
