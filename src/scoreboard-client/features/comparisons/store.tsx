"use client";

import {
  createContext,
  type Dispatch,
  type ReactNode,
  useContext,
  useEffect,
  useMemo,
  useReducer,
} from "react";

import { MockComparisonDataSource } from "./mock";
import type {
  ComparisonDataSource,
  ComparisonDataset,
  ComparisonId,
  DomainId,
  HistoryPoint,
} from "./types";

interface ComparisonState {
  status: "loading" | "ready" | "error";
  data: ComparisonDataset | null;
  error: string | null;
  comparisonId: ComparisonId;
  domainId: DomainId;
  selectedHistoryPointId: string | null;
}

type ComparisonAction =
  | { type: "loaded"; data: ComparisonDataset }
  | { type: "failed"; error: string }
  | { type: "select-comparison"; comparisonId: ComparisonId }
  | { type: "select-domain"; domainId: DomainId }
  | { type: "select-history-point"; pointId: string | null };

const initialState: ComparisonState = {
  status: "loading",
  data: null,
  error: null,
  comparisonId: "generation",
  domainId: "regular",
  selectedHistoryPointId: null,
};

function comparisonReducer(
  state: ComparisonState,
  action: ComparisonAction,
): ComparisonState {
  switch (action.type) {
    case "loaded":
      return { ...state, status: "ready", data: action.data, error: null };
    case "failed":
      return { ...state, status: "error", error: action.error };
    case "select-comparison":
      return {
        ...state,
        comparisonId: action.comparisonId,
        selectedHistoryPointId: null,
      };
    case "select-domain":
      return { ...state, domainId: action.domainId };
    case "select-history-point":
      return { ...state, selectedHistoryPointId: action.pointId };
  }
}

interface ComparisonContextValue {
  state: ComparisonState;
  dispatch: Dispatch<ComparisonAction>;
  selectedHistoryPoint: HistoryPoint | null;
}

const ComparisonContext = createContext<ComparisonContextValue | null>(null);
const DEFAULT_DATA_SOURCE = new MockComparisonDataSource();

export function ComparisonProvider({
  children,
  dataSource = DEFAULT_DATA_SOURCE,
}: {
  children: ReactNode;
  dataSource?: ComparisonDataSource;
}) {
  const [state, dispatch] = useReducer(comparisonReducer, initialState);

  useEffect(() => {
    let cancelled = false;
    dataSource
      .load()
      .then((data) => {
        if (!cancelled) dispatch({ type: "loaded", data });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          dispatch({
            type: "failed",
            error: error instanceof Error ? error.message : String(error),
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [dataSource]);

  const selectedHistoryPoint =
    state.data?.history.find((point) => point.id === state.selectedHistoryPointId) ?? null;
  const value = useMemo(
    () => ({ state, dispatch, selectedHistoryPoint }),
    [selectedHistoryPoint, state],
  );

  return <ComparisonContext.Provider value={value}>{children}</ComparisonContext.Provider>;
}

export function useComparisonStore(): ComparisonContextValue {
  const value = useContext(ComparisonContext);
  if (!value) throw new Error("useComparisonStore must be used inside ComparisonProvider");
  return value;
}
