'use client';

import { createContext, useContext, useState } from 'react';

export type Weights = { safety: number; yield: number; energy: number; wear: number };

type Ctx = {
  weights: Weights;
  setWeights: (w: Weights) => void;
  applyPreset: (preset: keyof typeof PRESETS) => void;
};

export const PRESETS = {
  balanced:  { safety: 0.4,  yield: 0.25, energy: 0.2,  wear: 0.15 },
  safer:     { safety: 0.7,  yield: 0.1,  energy: 0.05, wear: 0.15 },
  faster:    { safety: 0.2,  yield: 0.6,  energy: 0.1,  wear: 0.1  },
  cheaper:   { safety: 0.2,  yield: 0.15, energy: 0.55, wear: 0.1  },
  gentle:    { safety: 0.3,  yield: 0.15, energy: 0.15, wear: 0.4  },
} as const;

const WCtx = createContext<Ctx | null>(null);

export function WeightsProvider({ children }: { children: React.ReactNode }) {
  const [weights, setWeights] = useState<Weights>(PRESETS.balanced);
  const applyPreset = (preset: keyof typeof PRESETS) => setWeights(PRESETS[preset]);
  return (
    <WCtx.Provider value={{ weights, setWeights, applyPreset }}>
      {children}
    </WCtx.Provider>
  );
}

export function useWeights() {
  const ctx = useContext(WCtx);
  if (!ctx) throw new Error('useWeights outside WeightsProvider');
  return ctx;
}

export function scoreVariant(metrics: Record<string, number>, weights: Weights): number {
  return (
    weights.safety * (metrics.safety ?? 0) +
    weights.yield  * (metrics.yield  ?? 0) +
    weights.energy * (metrics.energy ?? 0) +
    weights.wear   * (metrics.wear   ?? 0)
  );
}
