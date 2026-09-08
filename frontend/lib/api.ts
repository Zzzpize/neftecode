const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    throw new Error(`API ${path} → ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>('/health'),
  recommend: (timestamp: string) =>
    request<{ decision_id: string; mode: string; payload: Record<string, unknown> }>(
      '/recommend',
      { method: 'POST', body: JSON.stringify({ timestamp }) },
    ),
  ask: (decision_id: string, question: string) =>
    request<{ answer: string }>('/ask', {
      method: 'POST',
      body: JSON.stringify({ decision_id, question }),
    }),
  state: (ts: string) => request<Record<string, unknown>>(`/state?ts=${ts}`),
  trace: (decision_id: string) => request<{ messages: unknown[] }>(`/trace/${decision_id}`),
  scenarios: () => request<{ id: string; name: string; description: string }[]>('/scenarios'),
};
