import { describe, expect, it } from 'vitest';
import { bandOfSorted, copyColumn, medianOfInt32, sortedQuantile } from './statistics';

describe('sortedQuantile', () => {
  it('handles empty and single-element buffers', () => {
    expect(sortedQuantile(new Float64Array(0), 0, 0.5)).toBe(0);
    expect(sortedQuantile(Float64Array.from([7]), 1, 0.5)).toBe(7);
  });

  it('returns exact endpoints', () => {
    const sorted = Float64Array.from([1, 2, 3, 4]);
    expect(sortedQuantile(sorted, 4, 0)).toBe(1);
    expect(sortedQuantile(sorted, 4, 1)).toBe(4);
  });

  it('interpolates fractional ranks (Hyndman–Fan type 7)', () => {
    const sorted = Float64Array.from([1, 2, 3, 4]);
    // p = 0.25 → index 0.75 → 1·0.25 + 2·0.75
    expect(sortedQuantile(sorted, 4, 0.25)).toBeCloseTo(1.75, 12);
    // p = 0.5 → index 1.5 → 2.5
    expect(sortedQuantile(sorted, 4, 0.5)).toBeCloseTo(2.5, 12);
  });

  it('reads the five band statistics', () => {
    const band = bandOfSorted(Float64Array.from([1, 2, 3, 4, 5]), 5);
    expect(band.p5).toBeCloseTo(1.2, 12);
    expect(band.median).toBe(3);
    expect(band.p95).toBeCloseTo(4.8, 12);
  });
});

describe('medianOfInt32', () => {
  const scratchOf = (n: number) => new Int32Array(n);

  it('returns 0 for an empty sample', () => {
    expect(medianOfInt32(new Int32Array(0), 0, scratchOf(1))).toBe(0);
  });

  it('returns the element for a single-element sample', () => {
    expect(medianOfInt32(Int32Array.from([7]), 1, scratchOf(1))).toBe(7);
  });

  it('returns the middle element for odd-length samples', () => {
    expect(medianOfInt32(Int32Array.from([5, 1, 3]), 3, scratchOf(3))).toBe(3);
  });

  it('returns the fractional arithmetic median for even-length samples', () => {
    expect(medianOfInt32(Int32Array.from([2, 3]), 2, scratchOf(2))).toBe(2.5);
    expect(medianOfInt32(Int32Array.from([2, 3, 4, 5]), 4, scratchOf(4))).toBe(3.5);
    expect(medianOfInt32(Int32Array.from([1, 4]), 2, scratchOf(2))).toBe(2.5);
  });
});

describe('copyColumn', () => {
  it('extracts one year column from the simulation matrix layout', () => {
    // Two paths, horizon 3, row-major per simulation.
    const matrix = Float64Array.from([10, 11, 12, 20, 21, 22]);
    const scratch = new Float64Array(2);
    copyColumn(matrix, 2, 3, 1, scratch);
    expect(Array.from(scratch)).toEqual([11, 21]);
  });
});
