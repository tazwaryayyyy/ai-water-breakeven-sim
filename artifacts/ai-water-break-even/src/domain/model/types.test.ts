import { describe, expect, it } from 'vitest';
import {
  applicableMilestones,
  DEFAULT_CONFIG,
  PARAMETER_KEYS,
  PARAMETER_LIMITS,
  probabilityAt,
} from './types';

describe('applicableMilestones', () => {
  it('only returns milestones within the horizon', () => {
    expect(applicableMilestones(10)).toEqual([10]);
    expect(applicableMilestones(15)).toEqual([10]);
    expect(applicableMilestones(20)).toEqual([10, 20]);
    expect(applicableMilestones(29)).toEqual([10, 20]);
    expect(applicableMilestones(30)).toEqual([10, 20, 30]);
    expect(applicableMilestones(49)).toEqual([10, 20, 30]);
    expect(applicableMilestones(50)).toEqual([10, 20, 30, 50]);
    expect(applicableMilestones(80)).toEqual([10, 20, 30, 50]);
  });

  it('never duplicates milestones under horizon clamping', () => {
    const milestones = applicableMilestones(10);
    expect(new Set(milestones).size).toBe(milestones.length);
  });
});

describe('probabilityAt', () => {
  it('reads the milestone-specific cumulative probability', () => {
    const probabilities = { y10: 0.1, y20: 0.2, y30: 0.3, y50: 0.5 };
    expect(probabilityAt(probabilities, 10)).toBe(0.1);
    expect(probabilityAt(probabilities, 20)).toBe(0.2);
    expect(probabilityAt(probabilities, 30)).toBe(0.3);
    expect(probabilityAt(probabilities, 50)).toBe(0.5);
  });
});

describe('PARAMETER_LIMITS', () => {
  it('covers every model parameter and run control', () => {
    for (const key of [...PARAMETER_KEYS, 'horizon', 'simulations', 'seed'] as const) {
      const { min, max, step } = PARAMETER_LIMITS[key];
      expect(min).toBeLessThan(max);
      expect(step).toBeGreaterThan(0);
    }
  });

  it('keeps the default configuration legal', () => {
    for (const key of [...PARAMETER_KEYS, 'horizon', 'simulations', 'seed'] as const) {
      const { min, max } = PARAMETER_LIMITS[key];
      expect(DEFAULT_CONFIG[key]).toBeGreaterThanOrEqual(min);
      expect(DEFAULT_CONFIG[key]).toBeLessThanOrEqual(max);
    }
  });
});
