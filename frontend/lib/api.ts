const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    cache: 'no-store',
  });
  if (!res.ok) {
    throw new Error(`API ${path} -> ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export type StateSnapshot = {
  timestamp: string;
  tags: Record<string, number>;
  lims_values: Record<string, number>;
  lims_age_hours: Record<string, number>;
};

export type Scenario = {
  id: string;
  name: string;
  description: string;
  timestamp: string;
};

export type HistoryPoint = { t: string; v: number | null };

export type Interval = { mean: number; low: number; high: number; unit: string };

export type Prediction = {
  sulfur: Interval | null;
  t50: Interval | null;
  t90: Interval | null;
  d15: Interval | null;
  confidence: 'high' | 'medium' | 'low';
  spec_risk: Record<string, number>;
};

export type Variant = {
  id: string;
  action: Record<string, number>;
  delta: Record<string, number>;
  metrics: Record<'safety' | 'yield' | 'energy' | 'wear', number>;
  predicted: Prediction | null;
  feasible: boolean;
  infeasible_reason: string | null;
};

export type TraceStep = {
  agent: string;
  duration_ms: number;
  input_summary: string;
  output: Record<string, unknown>;
};

export type Recommendation = {
  decision_id: string;
  mode: 'recommend' | 'silent' | 'refuse';
  timestamp: string;
  variants: Variant[];
  default_weights: Record<string, number>;
  trace: TraceStep[];
  explanation_text: string;
  warnings: string[];
};

export const api = {
  health: () => request<{ status: string }>('/health'),
  range: () => request<{ min_date: string; max_date: string }>('/state/range'),
  state: (ts: string) => request<StateSnapshot>(`/state?ts=${encodeURIComponent(ts)}`),
  history: (tag: string, ts_from: string, ts_to: string) =>
    request<{ tag: string; points: HistoryPoint[] }>(
      `/history?tag=${encodeURIComponent(tag)}&ts_from=${encodeURIComponent(ts_from)}&ts_to=${encodeURIComponent(ts_to)}`,
    ),
  scenarios: () => request<Scenario[]>('/scenarios'),
  recommend: (timestamp: string, weights?: Record<string, number>) =>
    request<Recommendation>('/recommend', {
      method: 'POST',
      body: JSON.stringify({ timestamp, weights }),
    }),
  ask: (decision_id: string, question: string) =>
    request<{ answer: string }>('/ask', {
      method: 'POST',
      body: JSON.stringify({ decision_id, question }),
    }),
};
