"use client";

import {
  createContext,
  type Dispatch,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
} from "react";

import { ApiComparisonDataSource } from "../lib/comparison_api";
import type {
  AnswerSampleGroups,
  ComparisonDataSource,
  ComparisonDataset,
  ComparisonId,
  DomainId,
  HistoryPoint,
  ScoreCellSelection,
} from "../lib/comparison_types";

interface ComparisonState {
  status: "loading" | "ready" | "error";
  data: ComparisonDataset | null;
  error: string | null;
  comparisonId: ComparisonId;
  domainId: DomainId;
  selectedHistoryPointId: string | null;
  selectedScoreCell: ScoreCellSelection | null;
}

type ComparisonAction =
  | { type: "loaded"; data: ComparisonDataset }
  | { type: "failed"; error: string }
  | { type: "select-comparison"; comparisonId: ComparisonId }
  | { type: "select-domain"; domainId: DomainId }
  | { type: "select-history-point"; pointId: string | null }
  | { type: "select-score-cell"; selection: ScoreCellSelection | null };

const initialState: ComparisonState = {
  status: "loading",
  data: null,
  error: null,
  comparisonId: "generation",
  domainId: "regular",
  selectedHistoryPointId: null,
  selectedScoreCell: null,
};

function comparisonReducer(
  state: ComparisonState,
  action: ComparisonAction,
): ComparisonState {
  switch (action.type) {
    case "loaded":
      return {
        ...state,
        status: "ready",
        data: action.data,
        error: null,
        comparisonId: action.data.comparisons[0]?.id ?? state.comparisonId,
      };
    case "failed":
      return { ...state, status: "error", error: action.error };
    case "select-comparison":
      return {
        ...state,
        comparisonId: action.comparisonId,
        selectedHistoryPointId: null,
        selectedScoreCell: null,
      };
    case "select-domain":
      return { ...state, domainId: action.domainId, selectedScoreCell: null };
    case "select-history-point":
      return { ...state, selectedHistoryPointId: action.pointId };
    case "select-score-cell":
      return { ...state, selectedScoreCell: action.selection };
  }
}

interface ComparisonContextValue {
  state: ComparisonState;
  dispatch: Dispatch<ComparisonAction>;
  selectedHistoryPoint: HistoryPoint | null;
  loadAnswerSamples: (
    selection: ScoreCellSelection,
    limit?: number,
  ) => Promise<AnswerSampleGroups>;
}

const ComparisonContext = createContext<ComparisonContextValue | null>(null);
const DEFAULT_DATA_SOURCE = new ApiComparisonDataSource();

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
  const loadAnswerSamples = useCallback(
    (selection: ScoreCellSelection, limit = 10) =>
      dataSource.loadAnswerSamples(selection, limit),
    [dataSource],
  );
  const value = useMemo(
    () => ({ state, dispatch, selectedHistoryPoint, loadAnswerSamples }),
    [loadAnswerSamples, selectedHistoryPoint, state],
  );

  return <ComparisonContext.Provider value={value}>{children}</ComparisonContext.Provider>;
}

export function useComparisonStore(): ComparisonContextValue {
  const value = useContext(ComparisonContext);
  if (!value) throw new Error("useComparisonStore must be used inside ComparisonProvider");
  return value;
}
