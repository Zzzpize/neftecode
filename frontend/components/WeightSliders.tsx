'use client';

import { PRESETS, useWeights, type Weights } from '@/components/WeightsStore';

const METRIC_LABELS: { key: keyof Weights; label: string; color: string }[] = [
  { key: 'safety', label: 'Безопасность', color: 'accent-emerald-500' },
  { key: 'yield',  label: 'Выход',        color: 'accent-sky-500' },
  { key: 'energy', label: 'Энергия',      color: 'accent-amber-500' },
  { key: 'wear',   label: 'Износ',        color: 'accent-fuchsia-500' },
];

const PRESET_LABELS: { key: keyof typeof PRESETS; label: string }[] = [
  { key: 'balanced', label: 'Баланс' },
  { key: 'safer',    label: 'Безопаснее' },
  { key: 'faster',   label: 'Быстрее' },
  { key: 'cheaper',  label: 'Дешевле' },
  { key: 'gentle',   label: 'Мягче' },
];

export function WeightSliders() {
  const { weights, setWeights, applyPreset } = useWeights();

  const update = (key: keyof Weights, val: number) => {
    setWeights({ ...weights, [key]: val });
  };

  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <h2 className="mb-2 text-sm font-medium uppercase tracking-wider text-neutral-400">
        Приоритеты
      </h2>
      <div className="mb-3 flex flex-wrap gap-1">
        {PRESET_LABELS.map((p) => (
          <button
            key={p.key}
            onClick={() => applyPreset(p.key)}
            className="rounded border border-neutral-800 bg-neutral-950 px-2 py-1 text-xs text-neutral-400 hover:border-neutral-700 hover:text-neutral-200"
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="space-y-3">
        {METRIC_LABELS.map(({ key, label, color }) => (
          <div key={key}>
            <div className="mb-1 flex justify-between text-xs">
              <span className="text-neutral-400">{label}</span>
              <span className="font-mono text-neutral-500">{(weights[key] * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={weights[key]}
              onChange={(e) => update(key, parseFloat(e.target.value))}
              className={`w-full ${color}`}
            />
          </div>
        ))}
      </div>

      <div className="mt-3 text-xs text-neutral-500">
        Двигай ползунки - рекомендация справа мгновенно перестроится.
      </div>
    </div>
  );
}
