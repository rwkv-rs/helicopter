"use client";

import { useComparisonStore } from "./store";

export function ComparisonTabs() {
  const { state, dispatch } = useComparisonStore();
  if (!state.data) return null;
  return (
    <nav className="comparison-tabs" aria-label="对比维度">
      {state.data.comparisons.map((comparison) => (
        <button
          className={`comparison-tab${comparison.id === state.comparisonId ? " active" : ""}`}
          key={comparison.id}
          type="button"
          onClick={() =>
            dispatch({ type: "select-comparison", comparisonId: comparison.id })
          }
        >
          {comparison.label}
        </button>
      ))}
    </nav>
  );
}

