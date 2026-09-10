'use client';

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useTime } from '@/components/TimeStore';
import { scoreVariant, useWeights } from '@/components/WeightsStore';
import { api } from '@/lib/api';

const METRICS: { key: 'safety' | 'yield' | 'energy' | 'wear'; label: string; color: string }[] = [
  { key: 'safety', label: 'Безоп.', color: 'bg-emerald-500' },
  { key: 'yield',  label: 'Выход',  color: 'bg-sky-500' },
  { key: 'energy', label: 'Энерг.', color: 'bg-amber-500' },
  { key: 'wear',   label: 'Износ',  color: 'bg-fuchsia-500' },
];

export function ParetoChart() {
  const { timestamp } = useTime();
  const { weights } = useWeights();

  const rec = useQuery({
    queryKey: ['recommend', timestamp],
    queryFn: () => api.recommend(timestamp),
    enabled: !!timestamp,
  });

  const { list, bestId } = useMemo(() => {
    const variants = rec.data?.variants ?? [];
    const withScore = variants.map((v) => ({
      ...v,
      score: scoreVariant(v.metrics, weights),
    }));
    const best = withScore.reduce(
      (a, b) => (b.score > a.score ? b : a),
      withScore[0],
    );
    return { list: withScore, bestId: best?.id ?? null };
  }, [rec.data, weights]);

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <h2 className="mb-3 flex items-baseline justify-between">
        <span className="text-sm font-medium uppercase tracking-wider text-neutral-400">
          Все варианты
        </span>
        {list.length > 0 && (
          <span className="text-xs text-neutral-500">
            {list.length} на Парето-фронте
          </span>
        )}
      </h2>

      {list.length === 0 ? (
        <div className="text-sm text-neutral-500">Нет вариантов</div>
      ) : (
        <div className="space-y-1.5 max-h-96 overflow-y-auto pr-1">
          {list.map((v) => (
            <VariantRow key={v.id} variant={v} isBest={v.id === bestId} />
          ))}
        </div>
      )}

      <div className="mt-3 text-xs text-neutral-500">
        Порядок вариантов фиксирован. Голубой — выбран по текущим приоритетам.
      </div>
    </div>
  );
}

function VariantRow({
  variant,
  isBest,
}: {
  variant: { id: string; score: number; metrics: Record<string, number>; predicted: { sulfur: { mean: number } | null } | null };
  isBest: boolean;
}) {
  const sulfur = variant.predicted?.sulfur?.mean ?? 0;
  return (
    <div
      className={`rounded border px-2 py-1.5 transition-colors ${
        isBest
          ? 'border-cyan-500/70 bg-cyan-950/30'
          : 'border-neutral-800 bg-neutral-950/40'
      }`}
    >
      <div className="mb-1 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          {isBest && <span className="text-[10px] uppercase tracking-wider text-cyan-400">выбран</span>}
          <span className="text-neutral-400">сера {sulfur.toFixed(2)}</span>
        </div>
        <span className={`font-mono ${isBest ? 'text-cyan-300' : 'text-neutral-500'}`}>
          score {(variant.score * 100).toFixed(0)}%
        </span>
      </div>
      <div className="flex gap-1">
        {METRICS.map(({ key, label, color }) => {
          const v = variant.metrics[key] ?? 0;
          return (
            <div key={key} className="flex-1 min-w-0">
              <div className="mb-0.5 flex justify-between text-[9px] text-neutral-500">
                <span>{label}</span>
                <span>{(v * 100).toFixed(0)}</span>
              </div>
              <div className="h-1.5 rounded bg-neutral-800 overflow-hidden">
                <div className={`h-full ${color}`} style={{ width: `${v * 100}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
