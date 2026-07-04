import { useMemo } from 'react';
import type { PitchPoint } from '../types/PitchPoint';

export interface PitchTrace {
  x: number[];      // time in seconds
  y: (number | null)[]; // frequency in Hz (null = unvoiced, breaks the line)
  customdata: number[]; // confidence per point
}

/**
 * Transform raw pitch points into Plotly trace arrays.
 *
 * time_ms is converted to seconds. Unvoiced frames (frequency === null)
 * produce null y-values so Plotly breaks the line rather than interpolating.
 */
export function usePitchTrace(points: PitchPoint[] | null): PitchTrace | null {
  return useMemo(() => {
    if (!points || points.length === 0) return null;

    const x: number[] = new Array(points.length);
    const y: (number | null)[] = new Array(points.length);
    const customdata: number[] = new Array(points.length);

    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      x[i] = p.time_ms / 1000;
      y[i] = p.frequency;
      customdata[i] = p.confidence;
    }

    return { x, y, customdata };
  }, [points]);
}
