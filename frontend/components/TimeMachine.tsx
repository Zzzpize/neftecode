'use client';

import { useQuery } from '@tanstack/react-query';
import { useTime } from '@/components/TimeStore';
import { api } from '@/lib/api';

export function TimeMachine() {
  const { timestamp, setTimestamp } = useTime();

  const scenarios = useQuery({
    queryKey: ['scenarios'],
    queryFn: api.scenarios,
  });

  const range = useQuery({
    queryKey: ['range'],
    queryFn: api.range,
  });

  const inputValue = timestamp.slice(0, 16);

  return (
    <div className="flex items-center gap-3 rounded-md border border-neutral-800 bg-neutral-900 px-3 py-2 text-sm">
      <label className="text-neutral-500">Момент:</label>
      <input
        type="datetime-local"
        value={inputValue}
        min={range.data?.min_date.slice(0, 16)}
        max={range.data?.max_date.slice(0, 16)}
        onChange={(e) => setTimestamp(e.target.value + ':00')}
        className="bg-neutral-950 border border-neutral-800 rounded px-2 py-1 font-mono text-xs"
      />
      {scenarios.data && (
        <>
          <span className="text-neutral-600">|</span>
          <span className="text-neutral-500">Сценарий:</span>
          <select
            className="bg-neutral-950 border border-neutral-800 rounded px-2 py-1 text-xs"
            onChange={(e) => {
              const s = scenarios.data?.find((x) => x.id === e.target.value);
              if (s) setTimestamp(s.timestamp);
            }}
            defaultValue=""
          >
            <option value="">выбрать...</option>
            {scenarios.data.map((s) => (
              <option key={s.id} value={s.id}>
                {s.name}
              </option>
            ))}
          </select>
        </>
      )}
    </div>
  );
}
