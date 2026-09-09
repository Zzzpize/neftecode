'use client';

import { useQuery } from '@tanstack/react-query';
import { useTime } from '@/components/TimeStore';
import { api } from '@/lib/api';

const KEY_TAGS: { key: string; label: string; unit: string }[] = [
  { key: 'avt_T55', label: 'T55 · Выход из печи П3', unit: '°C' },
  { key: 'avt_F30', label: 'F30 · Расход фр. 290-350', unit: 'м³/ч' },
  { key: 'avt_F65', label: 'F65 · Производительность К-2', unit: 'м³/ч' },
  { key: 'hydro_T5', label: 'T5 · Температура реактора ГО', unit: '°C' },
  { key: 'hydro_F14', label: 'F14 · Расход H₂', unit: 'м³/ч' },
  { key: 'pak_sulfur_ppm', label: 'Сера в ГОДТ (ПАК)', unit: 'ppm' },
  { key: 'pak_d15', label: 'D15 (ПАК)', unit: 'кг/м³' },
];

const SULFUR_LIMIT = 10;

export function StatePanel() {
  const { timestamp } = useTime();
  const state = useQuery({
    queryKey: ['state', timestamp],
    queryFn: () => api.state(timestamp),
    enabled: !!timestamp,
  });

  if (state.isLoading) return <Card title="Состояние процесса">Загрузка...</Card>;
  if (state.error) return <Card title="Состояние процесса">Ошибка</Card>;
  if (!state.data) return <Card title="Состояние процесса">Нет данных</Card>;

  const sulfur = state.data.tags['pak_sulfur_ppm'];
  const light =
    sulfur == null
      ? { color: 'bg-neutral-600', label: 'нет данных' }
      : sulfur < 7
        ? { color: 'bg-emerald-500', label: 'норма' }
        : sulfur < SULFUR_LIMIT
          ? { color: 'bg-yellow-500', label: 'внимание' }
          : { color: 'bg-red-500', label: 'нарушение' };

  const limsAges = Object.values(state.data.lims_age_hours);
  const oldestLims = limsAges.length ? Math.max(...limsAges) : null;

  return (
    <Card title="Состояние процесса">
      <div className="flex items-center gap-2 mb-4">
        <span className={`inline-block w-3 h-3 rounded-full ${light.color}`} />
        <span className="text-sm">{light.label}</span>
      </div>

      <table className="w-full text-xs">
        <tbody>
          {KEY_TAGS.map(({ key, label, unit }) => {
            const v = state.data!.tags[key];
            return (
              <tr key={key} className="border-b border-neutral-800 last:border-0">
                <td className="py-1.5 text-neutral-400">{label}</td>
                <td className="py-1.5 text-right font-mono">
                  {v == null ? '-' : v.toFixed(2)} <span className="text-neutral-600">{unit}</span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>

      <div className="mt-4 text-xs text-neutral-500">
        <div>ЛИМС: {Object.keys(state.data.lims_values).length} показателей</div>
        {oldestLims != null && (
          <div>Самый старый: {oldestLims.toFixed(0)} ч</div>
        )}
      </div>
    </Card>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4">
      <h2 className="mb-3 text-sm font-medium uppercase tracking-wider text-neutral-400">
        {title}
      </h2>
      {children}
    </div>
  );
}
