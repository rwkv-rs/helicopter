import { Fragment } from "react";

import type { DeltaCell, OverviewRow, ParamColumn } from "../lib/dtos/api/leaderboard";
import { deltaClass, pct, signedPct } from "./format";

interface Props {
  rows: OverviewRow[];
  columns: ParamColumn[];
  isDelta: boolean;
}

export function OverviewTable({ rows, columns, isDelta }: Props) {
  if (!rows.length) return <div className="empty">暂无领域均分数据。</div>;
  const span = isDelta ? 3 : 1;
  return (
    <div className="pivot-wrap">
      <table className="pivot bench-table overview-table">
        <thead>
          <tr>
            <th rowSpan={2}>field_name</th>
            {columns.map((column) => (
              <th key={column.param} colSpan={span}>
                {column.param_label.toUpperCase()}
              </th>
            ))}
          </tr>
          <tr>
            {columns.map((column) =>
              isDelta ? (
                <Fragment key={column.param}>
                  <th>{column.prev_label ?? "—"}</th>
                  <th>{column.latest_label}</th>
                  <th>delta</th>
                </Fragment>
              ) : (
                <th key={column.param}>{column.latest_label}</th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.domain_key}>
              <td>{row.domain_title}</td>
              {row.cells.map((cell, index) => {
                if (!isDelta) {
                  return (
                    <td key={index} className="score">
                      {pct((cell as { percent: number | null }).percent)}
                    </td>
                  );
                }
                const deltaCell = cell as DeltaCell;
                return (
                  <Fragment key={index}>
                    <td className="score">{pct(deltaCell.prev)}</td>
                    <td className="score">{pct(deltaCell.latest)}</td>
                    <td className={`score delta ${deltaClass(deltaCell.delta)}`}>{signedPct(deltaCell.delta)}</td>
                  </Fragment>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
