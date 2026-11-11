import { describe, expect, it } from 'vitest';
import { FastPRNG } from './prng';

describe('FastPRNG', () => {
  it('reproduces the same stream for the same seed', () => {
    const a = new FastPRNG(42);
    const b = new FastPRNG(42);
    for (let i = 0; i < 1000; i++) {
      expect(a.nextUint32()).toBe(b.nextUint32());
    }
  });

  it('produces different streams for different seeds', () => {
    const a = new FastPRNG(1);
    const b = new FastPRNG(2);
    const drawsA = Array.from({ length: 10 }, () => a.nextFloat());
    const drawsB = Array.from({ length: 10 }, () => b.nextFloat());
    expect(drawsA).not.toEqual(drawsB);
  });

  it('produces uniforms in [0, 1)', () => {
    const rng = new FastPRNG(7);
    for (let i = 0; i < 10000; i++) {
      const value = rng.nextFloat();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });

  it('keeps truncated-normal draws within bounds', () => {
    const rng = new FastPRNG(99);
    for (let i = 0; i < 5000; i++) {
      const value = rng.nextTruncatedNormal(0.05, 0.2, 0, 0.3);
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThanOrEqual(0.3);
    }
  });
});
