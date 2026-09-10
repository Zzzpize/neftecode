'use client';

import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useTime } from '@/components/TimeStore';
import { scoreVariant, useWeights } from '@/components/WeightsStore';
import { api, type Variant } from '@/lib/api';

const METRIC_META: Record<string, { label: string; color: string }> = {
  safety: { label: 'Безопасность', color: 'text-emerald-400' },
  yield:  { label: 'Выход',        color: 'text-sky-400' },
  energy: { label: 'Энергия',      color: 'text-amber-400' },
  wear:   { label: 'Износ',        color: 'text-fuchsia-400' },
};

export function RecommendationCard() {
  const { timestamp, setDecisionId } = useTime();
  const { weights } = useWeights();

  const rec = useQuery({
    queryKey: ['recommend', timestamp],
    queryFn: () => api.recommend(timestamp),
    enabled: !!timestamp,
  });

  const best = useMemo<Variant | null>(() => {
    const variants = rec.data?.variants ?? [];
    if (!variants.length) return null;
    return variants.reduce(
      (a, b) => (scoreVariant(b.metrics, weights) > scoreVariant(a.metrics, weights) ? b : a),
      variants[0],
    );
  }, [rec.data, weights]);

  if (rec.data?.decision_id) {
    setDecisionId(rec.data.decision_id);
  }

  if (rec.isLoading) return <Card>Обсчитываю рекомендацию...</Card>;
  if (rec.error) return <Card>Ошибка: {(rec.error as Error).message}</Card>;
  if (!rec.data) return <Card>Нет данных</Card>;

  if (rec.data.mode === 'silent') {
    return (
      <Card mode="silent" title="Режим стабилен">
        <p className="text-sm text-neutral-400">
          {rec.data.explanation_text || 'Вмешательство не требуется.'}
        </p>
      </Card>
    );
  }

  if (rec.data.mode === 'refuse' || !best) {
    return (
      <Card mode="refuse" title="Рекомендация невозможна">
        <p className="text-sm text-neutral-400">{rec.data.explanation_text}</p>
      </Card>
    );
  }

  return (
    <Card mode="recommend" title="Рекомендация">
      <div className="mb-4 text-xs text-neutral-500">{rec.data.explanation_text}</div>

      <div className="mb-4">
        <div className="mb-1 text-xs uppercase tracking-wider text-neutral-500">Изменить</div>
        <div className="space-y-1">
          {Object.entries(best.action).map(([tag, val]) => {
            const d = best.delta[tag];
            return (
              <div key={tag} className="flex justify-between font-mono text-xs">
                <span className="text-neutral-400">{tag}</span>
                <span>
                  <span className="text-neutral-500">→ </span>
                  <span className="text-neutral-100">{val}</span>
                  <span className={`ml-2 ${d > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    ({d > 0 ? '+' : ''}{d.toFixed(2)})
                  </span>
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {best.predicted?.sulfur && (
        <div className="mb-4">
          <div className="mb-1 text-xs uppercase tracking-wider text-neutral-500">Прогноз серы</div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-semibold">{best.predicted.sulfur.mean.toFixed(2)}</span>
            <span className="text-xs text-neutral-500">
              [{best.predicted.sulfur.low.toFixed(1)} — {best.predicted.sulfur.high.toFixed(1)}] мг/кг
            </span>
            {best.predicted.sulfur.mean < 10 ? (
              <span className="ml-auto text-xs text-emerald-400">в норме</span>
            ) : (
              <span className="ml-auto text-xs text-red-400">выше лимита</span>
            )}
          </div>
        </div>
      )}

      <div className="mb-4 grid grid-cols-4 gap-2">
        {(['safety', 'yield', 'energy', 'wear'] as const).map((k) => {
          const meta = METRIC_META[k];
          const v = best.metrics[k];
          return (
            <div key={k} className="rounded border border-neutral-800 bg-neutral-950 p-2">
              <div className="text-[10px] uppercase tracking-wider text-neutral-500">{meta.label}</div>
              <div className={`mt-0.5 text-sm font-semibold ${meta.color}`}>{(v * 100).toFixed(0)}%</div>
            </div>
          );
        })}
      </div>

      <div className="text-xs text-neutral-600">
        Из {rec.data.variants.length} вариантов Парето-фронта. Двигай приоритеты слева — выбор меняется.
      </div>
    </Card>
  );
}

function Card({
  children,
  title,
  mode,
}: {
  children: React.ReactNode;
  title?: string;
  mode?: 'recommend' | 'silent' | 'refuse';
}) {
  const border = {
    recommend: 'border-cyan-800/60',
    silent:    'border-emerald-800/60',
    refuse:    'border-red-800/60',
  }[mode ?? 'recommend'] ?? 'border-neutral-800';

  return (
    <div className={`rounded-lg border ${border} bg-neutral-900 p-4`}>
      {title && (
        <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-neutral-400">
          {title}
        </h2>
      )}
      {children}
    </div>
  );
}
