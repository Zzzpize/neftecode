'use client';

import { createContext, useContext, useState } from 'react';

type Ctx = {
  timestamp: string;
  setTimestamp: (t: string) => void;
  decisionId: string | null;
  setDecisionId: (id: string | null) => void;
};

const TimeCtx = createContext<Ctx | null>(null);

const DEFAULT_TS = '2024-06-15T14:20:00';

export function TimeProvider({ children }: { children: React.ReactNode }) {
  const [timestamp, setTimestamp] = useState<string>(DEFAULT_TS);
  const [decisionId, setDecisionId] = useState<string | null>(null);
  return (
    <TimeCtx.Provider value={{ timestamp, setTimestamp, decisionId, setDecisionId }}>
      {children}
    </TimeCtx.Provider>
  );
}

export function useTime() {
  const ctx = useContext(TimeCtx);
  if (!ctx) throw new Error('useTime outside TimeProvider');
  return ctx;
}
