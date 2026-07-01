import { Fragment } from "react";

import type { CellMeta, DeltaCell, DetailCell, LeaderboardRow, ParamColumn } from "../lib/dtos/api/leaderboard";
import { deltaClass, pct, signedPct } from "./format";

interface Props {
  paramColumns: ParamColumn[];
  isDelta: boolean;
  rows: LeaderboardRow[];
  onCellClick?: (meta: CellMeta) => void;
}

export function LeaderboardTable({ paramColumns, isDelta, rows, onCellClick }: Props) {
  if (!rows.length) return <div className="empty">该领域暂无满足展示阈值的数据。</div>;
  return (
    <div className="pivot-wrap">
      <table className="pivot bench-table">
        <thead>
          <tr>
            <th>benchmark</th>
            <th>samples</th>
            <th>eval_method</th>
            <th>k_metric</th>
            {paramColumns.map((column) => (
              <th key={column.param} colSpan={isDelta ? 3 : 1}>
                {column.param_label.toUpperCase()}
              </th>
            ))}
          </tr>
          {isDelta ? (
            <tr>
              <th />
              <th />
              <th />
              <th />
              {paramColumns.map((column) => (
                <Fragment key={column.param}>
                  <th>{column.prev_label || "—"}</th>
                  <th>{column.latest_label}</th>
                  <th>delta</th>
                </Fragment>
              ))}
            </tr>
          ) : null}
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.benchmark_name}:${row.eval_method}:${row.k_metric}`}>
              <td>{row.benchmark_name}</td>
              <td>{row.num_samples ?? "—"}</td>
              <td>{row.eval_method}</td>
              <td>{row.k_metric}</td>
              {row.cells.map((cell, index) =>
                isDelta ? (
                  <DeltaCells key={index} cell={cell as DeltaCell} onCellClick={onCellClick} />
                ) : (
                  <ScoreCell
                    key={index}
                    value={(cell as DetailCell).percent}
                    meta={(cell as DetailCell).meta}
                    onCellClick={onCellClick}
                  />
                )
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DeltaCells({ cell, onCellClick }: { cell: DeltaCell; onCellClick?: (meta: CellMeta) => void }) {
  return (
    <>
      <ScoreCell value={cell.prev} meta={cell.prev_meta} onCellClick={onCellClick} />
      <ScoreCell value={cell.latest} meta={cell.latest_meta} onCellClick={onCellClick} />
      <td className={`score delta ${deltaClass(cell.delta)}`}>{signedPct(cell.delta)}</td>
    </>
  );
}

function ScoreCell({
  value,
  meta,
  onCellClick,
}: {
  value: number | null;
  meta: CellMeta | null;
  onCellClick?: (meta: CellMeta) => void;
}) {
  if (!meta?.clickable || !onCellClick) {
    return <td className={meta?.clickable ? "score clickable" : "score"}>{pct(value)}</td>;
  }
  return (
    <td className="score clickable">
      <button
        type="button"
        className="score-button"
        title={meta.tooltip || "查看评测明细"}
        onClick={() => onCellClick(meta)}
      >
        {pct(value)}
      </button>
    </td>
  );
}
