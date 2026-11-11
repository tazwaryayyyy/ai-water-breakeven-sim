import { describe, expect, it } from 'vitest';
import {
  allocateExtraction,
  allocateMatrices,
  runScenarioPass,
  runSensitivityPass,
  runSync,
  sanitizeConfig,
  summarizeMatrices,
  unpackQuantiles,
} from '@/domain/model/simulation-engine';
import { DEFAULT_CONFIG, type ScenarioConfig, type SimulationResult } from '@/domain/model/types';
import { buildEnsembleCSV, buildReproducibilityPayload, csvEscape } from './useExport';

/** Build a full UI-facing result from production engine functions only. */
function buildResult(overrides: Partial<ScenarioConfig> = {}): SimulationResult {
  const config = sanitizeConfig({
    ...DEFAULT_CONFIG,
    simulations: 250,
    horizon: 12,
    seed: 1234,
    ...overrides,
  });
  const matrices = allocateMatrices(config.simulations, config.horizon);
  const buffers = allocateExtraction(config.simulations, config.horizon);
  const summary = summarizeMatrices(runSync(config, matrices, buffers), config, buffers);
  const scenarioDescriptions: Record<string, string> = {
    Baseline: 'b',
    'Accelerated Savings': 'a',
    'High Growth': 'h',
  };
  return {
    summary: {
      config,
      years: Array.from(summary.years),
      burden: unpackQuantiles(summary.burdenQuantiles, config.horizon),
      savings: unpackQuantiles(summary.savingsQuantiles, config.horizon),
      balance: unpackQuantiles(summary.balanceQuantiles, config.horizon),
      breakEvenCounts: Array.from(summary.breakEvenYearCounts),
      neverCount: summary.neverCount,
      probabilities: summary.probabilities,
      neverProbability: summary.neverCount / config.simulations,
      medianBreakEvenYear: summary.medianBreakEvenYear,
      finalMedianBalance: summary.finalMedianBalance,
      finalMedianBurden: summary.finalMedianBurden,
      finalMedianSavings: summary.finalMedianSavings,
    },
    sensitivity: runSensitivityPass(config, 'finalBalance').results,
    scenarios: runScenarioPass(config).map((outcome) => ({
      ...outcome,
      medianBalance: Array.from(outcome.medianBalance),
      description: scenarioDescriptions[outcome.name] ?? '',
    })),
    computeMs: 1,
  };
}

/** Minimal RFC 4180 single-line parser for structural assertions. */
function parseCsvLine(line: string): string[] {
  const fields: string[] = [];
  let current = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (quoted) {
      if (char === '"') {
        if (line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          quoted = false;
        }
      } else {
        current += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ',') {
      fields.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  fields.push(current);
  return fields;
}

describe('csvEscape', () => {
  it('leaves plain values untouched', () => {
    expect(csvEscape('Baseline')).toBe('Baseline');
    expect(csvEscape(12.5)).toBe('12.5');
  });

  it('quotes values containing commas, quotes, or newlines', () => {
    expect(csvEscape('a,b')).toBe('"a,b"');
    expect(csvEscape('say "hi"')).toBe('"say ""hi"""');
    expect(csvEscape('line\nbreak')).toBe('"line\nbreak"');
    expect(csvEscape('cr\r')).toBe('"cr\r"');
  });
});

describe('buildEnsembleCSV', () => {
  const result = buildResult();
  const csv = buildEnsembleCSV(result);
  const lines = csv.split('\n');

  it('has a consistent rectangular schema on every row', () => {
    expect(lines.length).toBeGreaterThan(10);
    for (const line of lines) {
      expect(parseCsvLine(line)).toHaveLength(6);
    }
    expect(parseCsvLine(lines[0])).toEqual(['record', 'series', 'year', 'statistic', 'value', 'unit']);
  });

  it('includes seed, horizon, simulation count, units, and model version', () => {
    const meta = new Map(lines.map((line) => parseCsvLine(line)).filter((f) => f[0] === 'meta').map((f) => [f[3], f[4]]));
    expect(meta.get('seed')).toBe(String(result.summary.config.seed));
    expect(meta.get('horizon')).toBe(String(result.summary.config.horizon));
    expect(meta.get('simulations')).toBe(String(result.summary.config.simulations));
    expect(meta.get('volumes')).toBe('millions of cubic meters');
    expect(meta.has('model_version')).toBe(true);
    expect(meta.has('full_payload')).toBe(true);
  });

  it('includes all model assumptions', () => {
    const assumptionRows = lines.map(parseCsvLine).filter((f) => f[0] === 'assumption');
    expect(assumptionRows).toHaveLength(8);
    const starting = assumptionRows.find((f) => f[1] === 'startingWaterUse');
    expect(starting?.[4]).toBe(String(result.summary.config.startingWaterUse));
  });

  it('preserves all five quantiles for burden, savings, and balance', () => {
    const horizon = result.summary.config.horizon;
    const trajectory = lines.map(parseCsvLine).filter((f) => f[0] === 'trajectory');
    expect(trajectory).toHaveLength(3 * horizon * 5);
    for (const series of ['burden', 'savings', 'balance'] as const) {
      for (const statistic of ['p5', 'p25', 'median', 'p75', 'p95']) {
        expect(trajectory.some((f) => f[1] === series && f[3] === statistic && f[2] === '1')).toBe(true);
      }
    }
  });

  it('preserves the break-even year distribution and never count', () => {
    const breakEven = lines.map(parseCsvLine).filter((f) => f[0] === 'break_even');
    expect(breakEven).toHaveLength(result.summary.config.horizon + 1);
    const never = breakEven.find((f) => f[1] === 'never');
    expect(never?.[4]).toBe(String(result.summary.neverCount));
    const total = breakEven.reduce((sum, f) => sum + Number(f[4]), 0);
    expect(total).toBe(result.summary.config.simulations);
  });

  it('preserves scenario comparison results with the applicable cutoffs', () => {
    const scenarioRows = lines.map(parseCsvLine).filter((f) => f[0] === 'scenario');
    for (const scenario of result.scenarios) {
      const own = scenarioRows.filter((f) => f[1] === scenario.name);
      // One median-balance row per year plus the summary statistics.
      expect(own.filter((f) => f[3] === 'median_balance')).toHaveLength(result.summary.config.horizon);
      expect(own.find((f) => f[3] === 'break_even_by_year_12')?.[4]).toBe(
        String(scenario.probability30),
      );
      expect(own.find((f) => f[3] === 'break_even_by_year_50')).toBeUndefined();
      expect(own.find((f) => f[3] === 'final_median_balance')?.[4]).toBe(
        String(scenario.finalMedianBalance),
      );
    }
  });
});

describe('buildReproducibilityPayload', () => {
  const payload = buildReproducibilityPayload(buildResult());

  it('records the model version, configuration, seed, and complete summary', () => {
    expect(payload.modelVersion).toBeTypeOf('string');
    expect(payload.config).toEqual(expect.objectContaining({ seed: 1234, horizon: 12 }));
    expect(payload.summary).toHaveProperty('probabilities');
    expect(payload.summary).toHaveProperty('breakEvenCounts');
    expect(payload.summary).toHaveProperty('neverCount');
    expect(payload.sensitivity).toHaveLength(8);
    expect(payload.scenarios).toHaveLength(3);
  });

  it('documents the sampling assumptions and calculation definitions', () => {
    expect(payload.engine.sampling).toHaveProperty('adoptionRate');
    expect(payload.engine.sampling.adoptionRate).toMatchObject({
      distribution: 'truncated-normal',
      spread: 0.1,
    });
    expect(payload.engine.sensitivity.metrics).toHaveProperty('meanCappedYear');
    expect(payload.engine.samplingNotes.length).toBeGreaterThan(0);
    expect(payload.reproducibility.note).toContain(payload.modelVersion);
  });
});
