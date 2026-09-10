'use client';

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, Cell,
} from 'recharts';
import { useTime } from '@/components/TimeStore';
import { scoreVariant, useWeights } from '@/components/WeightsStore';
import { api, type Variant } from '@/lib/api';

export function ParetoChart() {
  const { timestamp } = useTime();
  const { weights } = useWeights();

  const rec = useQuery({
    queryKey: ['recommend', timestamp],
    queryFn: () => api.recommend(timestamp),
    enabled: !!timestamp,
  });

  const data = useMemo(() => {
    const variants = rec.data?.variants ?? [];
    if (!variants.length) return { points: [], bestId: null };
    const scored = variants.map((v) => ({
      id: v.id,
      x: v.predicted?.sulfur?.mean ?? 0,
      y: v.metrics.yield,
      safety: v.metrics.safety,
      energy: v.metrics.energy,
      wear: v.metrics.wear,
      score: scoreVariant(v.metrics, weights),
      feasible: v.feasible,
    }));
    const best = scored.reduce((a, b) => (b.score > a.score ? b : a), scored[0]);
    return { points: scored, bestId: best.id };
  }, [rec.data, weights]);

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <h2 className="mb-3 flex items-baseline justify-between">
        <span className="text-sm font-medium uppercase tracking-wider text-neutral-400">
          Парето-фронт вариантов
        </span>
        <span className="text-xs text-neutral-500">
          {data.points.length ? `${data.points.length} допустимых` : ''}
        </span>
      </h2>

      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 16, bottom: 30, left: 8 }}>
            <XAxis
              type="number"
              dataKey="x"
              name="сера"
              domain={['dataMin - 0.5', 'dataMax + 0.5']}
              tick={{ fontSize: 10, fill: '#737373' }}
              label={{ value: 'Сера, мг/кг (меньше — лучше)', position: 'insideBottom', offset: -18, fill: '#737373', fontSize: 11 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="выход"
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: '#737373' }}
              label={{ value: 'Выход', angle: -90, position: 'insideLeft', offset: 8, fill: '#737373', fontSize: 11 }}
            />
            <ZAxis type="number" dataKey="score" range={[80, 320]} />
            <ReferenceLine x={10} stroke="#dc2626" strokeDasharray="3 3" label={{ value: 'лимит', fill: '#dc2626', fontSize: 10, position: 'top' }} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              contentStyle={{ background: '#171717', border: '1px solid #262626', fontSize: 12 }}
              labelStyle={{ color: '#a3a3a3' }}
              itemStyle={{ color: '#e5e5e5' }}
              formatter={(_v, name, item) => {
                const p = item.payload as (typeof data.points)[number];
                if (name === 'сера') return [`${p.x.toFixed(2)} ppm`, name];
                if (name === 'выход') return [`${(p.y * 100).toFixed(0)}%`, name];
                return [`${p.score.toFixed(3)}`, 'score'];
              }}
            />
            <Scatter data={data.points} isAnimationActive={false}>
              {data.points.map((p) => (
                <Cell
                  key={p.id}
                  fill={p.id === data.bestId ? '#22d3ee' : '#525252'}
                  stroke={p.id === data.bestId ? '#67e8f9' : 'transparent'}
                  strokeWidth={p.id === data.bestId ? 2 : 0}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-2 text-xs text-neutral-500">
        Цвет — выбранный по текущим приоритетам. Размер — общий score.
      </div>
    </div>
  );
}

export type { Variant };
