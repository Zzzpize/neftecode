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

export type Recommendation = {
  decision_id: string;
  mode: 'recommend' | 'silent' | 'refuse';
  payload: Record<string, unknown>;
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
  runScenario: (id: string) =>
    request<{ scenario_id: string; timestamp: string; decision_id: string }>(
      `/scenarios/${id}/run`,
      { method: 'POST' },
    ),
  recommend: (timestamp: string) =>
    request<Recommendation>('/recommend', {
      method: 'POST',
      body: JSON.stringify({ timestamp }),
    }),
  ask: (decision_id: string, question: string) =>
    request<{ answer: string }>('/ask', {
      method: 'POST',
      body: JSON.stringify({ decision_id, question }),
    }),
};
