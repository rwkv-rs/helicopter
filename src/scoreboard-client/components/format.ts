export function pct(value: number | null): string {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

export function signedPct(value: number | null): string {
  if (value == null) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}%`;
}

export function deltaClass(value: number | null): string {
  if (value == null || value === 0) return "";
  return value > 0 ? "up" : "down";
}
