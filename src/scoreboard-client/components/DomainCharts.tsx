"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ChartPayload } from "../lib/dtos/api/leaderboard";

const PALETTE = ["#5b8cff", "#3ecf8e", "#f5b14c", "#f0637a", "#a78bfa", "#22d3ee", "#fb923c"];
const AXIS = { fill: "#9aa3b2", fontSize: 11 };
const GRID = "#303642";

function pctTick(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function tooltipValue(value: unknown): string {
  return typeof value === "number" ? pctTick(value) : String(value ?? "");
}

function pivot(data: { model: string; score: number }[], dimKey: string) {
  const byDim = new Map<string, Record<string, number | string>>();
  for (const item of data as Record<string, string | number>[]) {
    const dim = String(item[dimKey]);
    const row = byDim.get(dim) ?? { [dimKey]: dim };
    row[String(item.model)] = item.score as number;
    byDim.set(dim, row);
  }
  return [...byDim.values()];
}

function GroupedBar({
  data,
  dimKey,
  models,
  height,
}: {
  data: { model: string; score: number }[];
  dimKey: string;
  models: string[];
  height: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={pivot(data, dimKey)} margin={{ top: 8, right: 16, bottom: 20, left: 0 }}>
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey={dimKey} tick={AXIS} angle={-25} textAnchor="end" height={64} interval={0} />
        <YAxis tick={AXIS} tickFormatter={pctTick} domain={[0, 1]} />
        <Tooltip
          contentStyle={{ background: "#181b22", border: "1px solid #303642", borderRadius: 6, fontSize: 12 }}
          formatter={tooltipValue}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {models.map((model, index) => (
          <Bar key={model} dataKey={model} fill={PALETTE[index % PALETTE.length]} radius={[3, 3, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function DomainCharts({ chart }: { chart: ChartPayload[keyof ChartPayload] | null | undefined }) {
  if (!chart) return null;

  if (chart.type === "knowledge_bar") {
    return <GroupedBar data={chart.data} dimKey="subject" models={chart.models} height={Math.max(360, chart.subjects.length * 28)} />;
  }
  if (chart.type === "coding_bar") {
    return <GroupedBar data={chart.data} dimKey="dataset" models={chart.models} height={380} />;
  }
  if (chart.type === "instruction_bar") {
    return <GroupedBar data={chart.data} dimKey="domain" models={chart.models} height={400} />;
  }
  if (chart.type === "aime_line") {
    const byK = new Map<number, Record<string, number>>();
    for (const series of chart.series) {
      for (const point of series.points) {
        const row = byK.get(point.k) ?? { k: point.k };
        row[series.name] = point.acc;
        byK.set(point.k, row);
      }
    }
    const rows = [...byK.values()].sort((left, right) => left.k - right.k);
    return (
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={GRID} />
          <XAxis dataKey="k" tick={AXIS} />
          <YAxis tick={AXIS} tickFormatter={pctTick} domain={[0, "auto"]} />
          <Tooltip
            contentStyle={{ background: "#181b22", border: "1px solid #303642", borderRadius: 6, fontSize: 12 }}
            formatter={tooltipValue}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {chart.series.map((series, index) => (
            <Line key={series.name} dataKey={series.name} stroke={PALETTE[index % PALETTE.length]} dot strokeWidth={2} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    );
  }
  return null;
}
