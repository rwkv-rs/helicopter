"use client";

import {
  createContext,
  type ReactNode,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";

import { ApiEvaluationDataSource } from "../lib/evaluation_api";
import type {
  EvaluationDataSource,
  EvaluationDataset,
  EvaluationSummary,
} from "../lib/evaluation_types";

interface EvaluationContextValue {
  status: "loading" | "ready" | "error";
  data: EvaluationDataset | null;
  error: string | null;
  selected: EvaluationSummary | null;
  select: (evaluation: EvaluationSummary | null) => void;
  dataSource: EvaluationDataSource;
}

const DEFAULT_SOURCE = new ApiEvaluationDataSource();
const EvaluationContext = createContext<EvaluationContextValue | null>(null);

export function EvaluationProvider({
  children,
  dataSource = DEFAULT_SOURCE,
}: {
  children: ReactNode;
  dataSource?: EvaluationDataSource;
}) {
  const [status, setStatus] =
    useState<EvaluationContextValue["status"]>("loading");
  const [data, setData] = useState<EvaluationDataset | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, select] = useState<EvaluationSummary | null>(null);

  useEffect(() => {
    let cancelled = false;
    dataSource
      .load()
      .then((payload) => {
        if (cancelled) return;
        setData(payload);
        setStatus("ready");
      })
      .catch((reason: unknown) => {
        if (cancelled) return;
        setError(reason instanceof Error ? reason.message : String(reason));
        setStatus("error");
      });
    return () => {
      cancelled = true;
    };
  }, [dataSource]);

  const value = useMemo(
    () => ({ status, data, error, selected, select, dataSource }),
    [status, data, error, selected, dataSource],
  );
  return (
    <EvaluationContext.Provider value={value}>
      {children}
    </EvaluationContext.Provider>
  );
}

export function useEvaluations(): EvaluationContextValue {
  const value = useContext(EvaluationContext);
  if (!value) {
    throw new Error("useEvaluations must be used inside EvaluationProvider");
  }
  return value;
}
