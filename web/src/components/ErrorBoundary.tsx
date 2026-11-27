import React from "react";

/** Catches render / lazy-chunk errors and offers a calm reload instead of a blank screen. */
export class ErrorBoundary extends React.Component<{ children: React.ReactNode; resetKey?: string }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidUpdate(prev: { resetKey?: string }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }

  componentDidCatch(error: Error) {
    // eslint-disable-next-line no-console
    console.warn("Studio view error:", error);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="v-card p-10 text-center max-w-lg mx-auto my-10">
        <h2 className="text-[18px] font-[500] tracking-[-0.02em] text-[#171717]">Something didn't load</h2>
        <p className="mt-2 text-[13px] text-[#4d4d4d] leading-relaxed">
          This part of the app hit a problem — usually after an update. Reloading fixes it.
        </p>
        <div className="mt-5 flex justify-center gap-2">
          <button onClick={() => window.location.reload()} className="v-btn-primary h-9 px-4 text-[13px]">
            Reload app
          </button>
          <button
            onClick={() => {
              window.location.hash = "#/";
              this.setState({ error: null });
            }}
            className="v-btn-ghost h-9 px-4 text-[13px]"
          >
            Go home
          </button>
        </div>
      </div>
    );
  }
}
