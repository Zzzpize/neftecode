'use client';

import { useQuery } from '@tanstack/react-query';
import { useTime } from '@/components/TimeStore';
import { api } from '@/lib/api';

const AGENT_META: Record<string, { label: string; color: string }> = {
  data:         { label: 'Data',         color: 'bg-sky-500' },
  quality:      { label: 'Quality',      color: 'bg-emerald-500' },
  reliability:  { label: 'Reliability',  color: 'bg-amber-500' },
  optimization: { label: 'Optimization', color: 'bg-fuchsia-500' },
  orchestrator: { label: 'Orchestrator', color: 'bg-cyan-500' },
};

export function AgentTrace() {
  const { timestamp } = useTime();

  const rec = useQuery({
    queryKey: ['recommend', timestamp],
    queryFn: () => api.recommend(timestamp),
    enabled: !!timestamp,
  });

  const trace = rec.data?.trace ?? [];

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <h2 className="mb-3 flex items-baseline justify-between">
        <span className="text-sm font-medium uppercase tracking-wider text-neutral-400">
          Трейс агентов
        </span>
        {trace.length > 0 && (
          <span className="text-xs text-neutral-500">
            {trace.reduce((s, t) => s + t.duration_ms, 0).toFixed(0)} мс
          </span>
        )}
      </h2>

      {trace.length === 0 ? (
        <div className="text-sm text-neutral-500">Ожидание...</div>
      ) : (
        <div className="space-y-2">
          {trace.map((step, i) => {
            const meta = AGENT_META[step.agent] ?? { label: step.agent, color: 'bg-neutral-500' };
            return (
              <div key={i} className="rounded border border-neutral-800 bg-neutral-950 p-2">
                <div className="mb-1 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className={`inline-block h-2 w-2 rounded-full ${meta.color}`} />
                    <span className="text-xs font-semibold">{meta.label}</span>
                  </div>
                  <span className="font-mono text-[10px] text-neutral-500">
                    {step.duration_ms.toFixed(1)} мс
                  </span>
                </div>
                <div className="mb-1 text-[10px] text-neutral-500">{step.input_summary}</div>
                <div className="font-mono text-[10px] text-neutral-400">
                  {Object.entries(step.output).map(([k, v]) => (
                    <div key={k}>
                      <span className="text-neutral-600">{k}:</span> {String(v)}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
