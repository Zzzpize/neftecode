'use client';

import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { useTime } from '@/components/TimeStore';
import { api } from '@/lib/api';

const SULFUR_LIMIT = 10;

export function TagChart({ tag = 'pak_sulfur_ppm', hoursBack = 12 }: { tag?: string; hoursBack?: number }) {
  const { timestamp } = useTime();

  const range = getRange(timestamp, hoursBack);
  const q = useQuery({
    queryKey: ['history', tag, range.from, range.to],
    queryFn: () => api.history(tag, range.from, range.to),
    enabled: !!timestamp,
  });

  const data = (q.data?.points ?? [])
    .filter((p) => p.v != null)
    .map((p) => ({ t: p.t.slice(11, 16), v: p.v as number }));

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <h2 className="mb-3 flex items-baseline justify-between">
        <span className="text-sm font-medium uppercase tracking-wider text-neutral-400">
          {tag}
        </span>
        <span className="text-xs text-neutral-500">за {hoursBack} ч</span>
      </h2>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
            <XAxis dataKey="t" tick={{ fontSize: 10, fill: '#737373' }} minTickGap={40} />
            <YAxis tick={{ fontSize: 10, fill: '#737373' }} width={40} domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{ background: '#171717', border: '1px solid #262626', fontSize: 12 }}
              labelStyle={{ color: '#a3a3a3' }}
            />
            {tag === 'pak_sulfur_ppm' && (
              <ReferenceLine y={SULFUR_LIMIT} stroke="#dc2626" strokeDasharray="3 3" />
            )}
            <Line type="monotone" dataKey="v" stroke="#38bdf8" dot={false} strokeWidth={1.5} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function getRange(ts: string, hoursBack: number) {
  const to = new Date(ts);
  const from = new Date(to.getTime() - hoursBack * 3600_000);
  return {
    from: iso(from),
    to: iso(to),
  };
}

function iso(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  );
}
