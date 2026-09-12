'use client';

import { useMemo } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useTime } from '@/components/TimeStore';
import { scoreVariant, useWeights, type Weights } from '@/components/WeightsStore';
import { api, type Variant } from '@/lib/api';

const METRIC_META: {
  key: keyof Weights;
  label: string;
  color: string;
  bar: string;
}[] = [
  { key: 'safety', label: 'Безопасность', color: 'text-emerald-400', bar: 'bg-emerald-500' },
  { key: 'yield',  label: 'Выход',        color: 'text-sky-400',     bar: 'bg-sky-500' },
  { key: 'energy', label: 'Энергия',      color: 'text-amber-400',   bar: 'bg-amber-500' },
  { key: 'wear',   label: 'Износ',        color: 'text-fuchsia-400', bar: 'bg-fuchsia-500' },
];

export function RecommendationCard() {
  const { timestamp, setDecisionId } = useTime();
  const { weights } = useWeights();
  const queryClient = useQueryClient();

  const rec = useQuery({
    queryKey: ['recommend', timestamp],
    queryFn: () => api.recommend(timestamp),
    enabled: !!timestamp,
  });

  const refreshText = useMutation({
    mutationFn: () => api.recommend(timestamp, weights),
    onSuccess: (data) => {
      queryClient.setQueryData(['recommend', timestamp], data);
    },
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
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 mt-0.5">
            <div className="h-8 w-8 rounded-full bg-emerald-500/20 flex items-center justify-center">
              <span className="text-emerald-400 text-lg">✓</span>
            </div>
          </div>
          <div>
            <p className="text-sm text-neutral-200">
              {rec.data.explanation_text || 'Вмешательство не требуется.'}
            </p>
            <p className="mt-2 text-xs text-neutral-500">
              Все ключевые показатели в норме, риск нарушения спецификации низкий.
              Система не предлагает изменений режима.
            </p>
          </div>
        </div>
      </Card>
    );
  }

  if (rec.data.mode === 'refuse' || !best) {
    return (
      <Card mode="refuse" title="Рекомендация не выдана">
        <RefuseView explanation={rec.data.explanation_text} />
      </Card>
    );
  }

  const totalScore = scoreVariant(best.metrics, weights);

  return (
    <Card mode="recommend" title="Рекомендация">
      <div className="mb-3 flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <ExplanationBlock text={rec.data.explanation_text} />
        </div>
        <button
          onClick={() => refreshText.mutate()}
          disabled={refreshText.isPending}
          className="flex-shrink-0 rounded border border-neutral-800 bg-neutral-950 px-2 py-1 text-[10px] uppercase tracking-wider text-neutral-400 hover:border-neutral-700 hover:text-neutral-200 disabled:opacity-40"
          title="Перегенерировать пояснение под текущие приоритеты"
        >
          {refreshText.isPending ? 'обновляю...' : '↻ под мои приоритеты'}
        </button>
      </div>

      <div className="mb-4">
        <div className="mb-1 text-xs uppercase tracking-wider text-neutral-500">Изменить</div>
        <div className="space-y-1">
          {Object.entries(best.action)
            .sort((a, b) => Math.abs(best.delta[b[0]] ?? 0) - Math.abs(best.delta[a[0]] ?? 0))
            .slice(0, 5)
            .map(([tag, val]) => {
              const d = best.delta[tag] ?? 0;
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

      <div className="mb-2">
        <div className="mb-1.5 flex items-baseline justify-between">
          <span className="text-xs uppercase tracking-wider text-neutral-500">
            Из чего складывается score этого варианта
          </span>
          <span className="text-sm font-mono">
            {(totalScore * 100).toFixed(1)}%
          </span>
        </div>

        <div className="space-y-1">
          {METRIC_META.map(({ key, label, color, bar }) => {
            const value = best.metrics[key] ?? 0;
            const weight = weights[key];
            const contribution = value * weight;
            return (
              <div key={key} className="text-xs">
                <div className="flex justify-between mb-0.5">
                  <span className="text-neutral-400">{label}</span>
                  <span className="font-mono text-neutral-500">
                    <span className="text-neutral-400">вес {(weight * 100).toFixed(0)}%</span>
                    <span className="mx-1">×</span>
                    <span className={color}>значение {(value * 100).toFixed(0)}%</span>
                    <span className="mx-1">=</span>
                    <span className="text-neutral-100">{(contribution * 100).toFixed(1)}%</span>
                  </span>
                </div>
                <div className="h-1 rounded bg-neutral-800 overflow-hidden">
                  <div className={`h-full ${bar}`} style={{ width: `${contribution / Math.max(0.01, totalScore) * 100}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="mt-3 text-[11px] text-neutral-500">
        Из {rec.data.variants.length} вариантов Парето-фронта.
        «Значение» — оценка этого варианта по метрике. «Вес» — как её задал ты слева.
        Score = сумма вкладов. Двигаешь ползунки — вклад меняется, побеждает другой вариант.
      </div>
    </Card>
  );
}

const SECTION_KEYS = [
  'Время', 'Проблема', 'Действие', 'Эффект',
  'Проверки', 'Проверка', 'Уверенность', 'Обоснование',
];

function ExplanationBlock({ text }: { text: string }) {
  if (!text) return null;
  const sections = parseExplanation(text);
  if (sections.length === 0) {
    return <div className="mb-4 text-xs text-neutral-400 leading-relaxed">{text}</div>;
  }
  return (
    <div className="mb-4 space-y-1.5">
      {sections.map((s, i) => (
        <div key={i} className="text-xs">
          <span className="text-neutral-500 uppercase tracking-wider text-[10px]">{s.key}</span>
          <div className="text-neutral-300 leading-snug">{s.value}</div>
        </div>
      ))}
    </div>
  );
}

function parseExplanation(text: string): { key: string; value: string }[] {
  const pattern = new RegExp(`(${SECTION_KEYS.join('|')})\\s*:`, 'g');
  const matches = [...text.matchAll(pattern)];
  if (matches.length === 0) return [];
  const out: { key: string; value: string }[] = [];
  for (let i = 0; i < matches.length; i++) {
    const start = matches[i].index! + matches[i][0].length;
    const end = i + 1 < matches.length ? matches[i + 1].index! : text.length;
    const value = text.slice(start, end).trim().replace(/[.,;]+$/, '');
    if (value) out.push({ key: matches[i][1], value });
  }
  return out;
}

function RefuseView({ explanation }: { explanation: string }) {
  const reasons = parseRefuseReasons(explanation);
  return (
    <div className="flex items-start gap-3">
      <div className="flex-shrink-0 mt-0.5">
        <div className="h-8 w-8 rounded-full bg-red-500/20 flex items-center justify-center">
          <span className="text-red-400 text-lg">⚠</span>
        </div>
      </div>
      <div className="flex-1 min-w-0">
        <p className="mb-3 text-sm text-neutral-200">
          Данные текущего момента непригодны для безопасной рекомендации.
        </p>
        {reasons.length > 0 && (
          <ul className="space-y-1.5">
            {reasons.map((r, i) => (
              <li key={i} className="text-xs text-neutral-400 flex gap-2">
                <span className="text-red-400 mt-0.5">•</span>
                <span>
                  <span className="text-neutral-300">{r.title}</span>
                  {r.detail && (
                    <span className="text-neutral-500"> — {r.detail}</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        )}
        <p className="mt-3 text-[11px] text-neutral-500">
          Правильное поведение системы: при плохих данных лучше молчать, чем советовать наугад.
        </p>
      </div>
    </div>
  );
}

function parseRefuseReasons(text: string): { title: string; detail?: string }[] {
  if (!text) return [];
  const parts: { title: string; detail?: string }[] = [];
  const anomaly = text.match(/аномалия[^)]*\)/i);
  if (anomaly) parts.push({ title: 'Обнаружена аномалия', detail: anomaly[0] });
  if (/вне исторического/i.test(text)) {
    parts.push({ title: 'Состояние вне исторического диапазона режимов' });
  }
  const stale = text.match(/[Зз]астывшие теги:\s*([^У]+?)(?=\s*(?:Устаревшие|$))/);
  if (stale) {
    const tags = stale[1].trim().replace(/,\s*$/, '').split(',').map((s) => s.trim());
    parts.push({
      title: `Застывшие теги (${tags.length})`,
      detail: tags.slice(0, 4).join(', ') + (tags.length > 4 ? ` и ещё ${tags.length - 4}` : ''),
    });
  }
  if (/[Уу]старевшие лаб/i.test(text)) {
    parts.push({ title: 'Устаревшие лабораторные значения', detail: 'старше 24 часов' });
  }
  if (parts.length === 0 && text) {
    parts.push({ title: text });
  }
  return parts;
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
