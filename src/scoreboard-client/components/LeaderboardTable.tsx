"use client";

import { Fragment, useState } from "react";
import type {
  CellMeta,
  DeltaCell,
  DetailCell,
  LeaderboardRow,
  ParamColumn,
} from "../lib/dtos/api/leaderboard";
import { deltaClass, pct, signedPct } from "./format";

interface Props {
  paramColumns: ParamColumn[];
  isDelta: boolean;
  rows: LeaderboardRow[];
  onCellClick?: (meta: CellMeta) => void;
}

const ROW_AXIS_COLUMNS = [
  { className: "axis-benchmark", label: "benchmark" },
  { className: "axis-samples", label: "samples" },
  { className: "axis-method", label: "eval_method" },
  { className: "axis-kmetric", label: "k_metric" },
] as const;
const ROW_AXIS_WIDTH = 484;
const SCREEN_GROUP_CAPACITY = 6;
const OVERFLOW_SCORE_COL_WIDTH = 56;

function Tooltip({ text }: { text: string }) {
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  return (
    <>
      <span
        onMouseMove={(event) => setPos({ x: event.clientX + 14, y: event.clientY + 14 })}
        onMouseLeave={() => setPos(null)}
        className="cell-info"
      >
        ⓘ
      </span>
      {pos ? (
        <div className="tooltip-pop" style={{ left: pos.x, top: pos.y }}>
          {text}
        </div>
      ) : null}
    </>
  );
}

function ScoreCell({
  cell,
  onClick,
  className = "",
}: {
  cell: DetailCell;
  onClick?: (meta: CellMeta) => void;
  className?: string;
}) {
  const meta = cell.meta;
  const clickable = Boolean(meta?.clickable && onClick);
  const handleClick = clickable && meta && onClick ? () => onClick(meta) : undefined;
  return (
    <td
      className={`score ${className}${clickable ? " clickable" : ""}`}
      onClick={handleClick}
    >
      <span className="score-cell">{pct(cell.percent)}</span>
      {meta?.tooltip ? <Tooltip text={meta.tooltip} /> : null}
    </td>
  );
}

function DeltaCells({ cell, onClick }: { cell: DeltaCell; onClick?: (meta: CellMeta) => void }) {
  const prevClickable = Boolean(cell.prev_meta?.clickable && onClick);
  const latestClickable = Boolean(cell.latest_meta?.clickable && onClick);
  const handlePrevClick = prevClickable && cell.prev_meta && onClick ? () => onClick(cell.prev_meta!) : undefined;
  const handleLatestClick =
    latestClickable && cell.latest_meta && onClick ? () => onClick(cell.latest_meta!) : undefined;
  return (
    <>
      <td
        className={`score group-start subcol-prev${prevClickable ? " clickable" : ""}`}
        onClick={handlePrevClick}
      >
        <span className="score-cell">{pct(cell.prev)}</span>
        {cell.prev_meta?.tooltip ? <Tooltip text={cell.prev_meta.tooltip} /> : null}
      </td>
      <td
        className={`score subcol-latest${latestClickable ? " clickable" : ""}`}
        onClick={handleLatestClick}
      >
        <span className="score-cell">{pct(cell.latest)}</span>
        {cell.latest_meta?.tooltip ? <Tooltip text={cell.latest_meta.tooltip} /> : null}
      </td>
      <td className={`score delta subcol-delta ${deltaClass(cell.delta)}`}>
        <span className="score-cell">{signedPct(cell.delta)}</span>
      </td>
    </>
  );
}

export function LeaderboardTable({ paramColumns, isDelta, rows, onCellClick }: Props) {
  if (!rows.length) {
    return <div className="empty">该领域暂无满足展示阈值的数据。</div>;
  }
  const span = isDelta ? 3 : 1;
  const scoreColumnCount = paramColumns.length * span;
  const shouldFitViewport = paramColumns.length <= SCREEN_GROUP_CAPACITY;
  const minTableWidth = shouldFitViewport
    ? "100%"
    : `${ROW_AXIS_WIDTH + scoreColumnCount * OVERFLOW_SCORE_COL_WIDTH}px`;
  const scoreColumnWidth = shouldFitViewport
    ? `calc((100% - ${ROW_AXIS_WIDTH}px) / ${scoreColumnCount})`
    : `${OVERFLOW_SCORE_COL_WIDTH}px`;

  return (
    <div className="pivot-wrap">
      <table className="pivot bench-table" style={{ minWidth: minTableWidth }}>
        <colgroup>
          {ROW_AXIS_COLUMNS.map((column) => (
            <col key={column.className} className={`col-${column.className}`} />
          ))}
          {paramColumns.map((column) =>
            isDelta ? (
              <Fragment key={column.param}>
                <col className="col-score" style={{ width: scoreColumnWidth }} />
                <col className="col-score" style={{ width: scoreColumnWidth }} />
                <col className="col-score col-delta-width" style={{ width: scoreColumnWidth }} />
              </Fragment>
            ) : (
              <col key={column.param} className="col-score" style={{ width: scoreColumnWidth }} />
            )
          )}
        </colgroup>
        <thead>
          <tr className="group-row">
            {ROW_AXIS_COLUMNS.map((column) => (
              <th key={column.className} className={`col-meta axis-header ${column.className}`} rowSpan={2}>
                {column.label}
              </th>
            ))}
            {paramColumns.map((column) => (
              <th key={column.param} className="param-group" colSpan={span}>
                {column.param_label.toUpperCase()}
              </th>
            ))}
          </tr>
          <tr className="subhead-row">
            {paramColumns.map((column) =>
              isDelta ? (
                <Fragment3 key={column.param} prev={column.prev_label ?? "—"} latest={column.latest_label} />
              ) : (
                <th key={column.param} className="col-arch group-start">
                  {column.latest_label}
                </th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={`${row.benchmark_name}:${row.eval_method}:${row.k_metric}:${rowIndex}`}>
              <td className="bench axis-benchmark">{row.benchmark_name}</td>
              <td className="dim axis-samples">{row.num_samples ?? "—"}</td>
              <td className="dim axis-method">{row.eval_method}</td>
              <td className="dim axis-kmetric">{row.k_metric}</td>
              {row.cells.map((cell, cellIndex) =>
                isDelta ? (
                  <DeltaCells key={cellIndex} cell={cell as DeltaCell} onClick={onCellClick} />
                ) : (
                  <ScoreCell key={cellIndex} cell={cell as DetailCell} onClick={onCellClick} className="group-start" />
                )
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Fragment3({ prev, latest }: { prev: string; latest: string }) {
  return (
    <>
      <th className="col-arch group-start subcol-prev">{prev}</th>
      <th className="col-arch subcol-latest">{latest}</th>
      <th className="col-arch col-delta subcol-delta">delta</th>
    </>
  );
}
