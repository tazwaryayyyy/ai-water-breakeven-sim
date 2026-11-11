import { describe, expect, it } from 'vitest';
import { FastPRNG } from '../math/prng';
import {
  allocateExtraction,
  allocateMatrices,
  classifySustainedBreakEven,
  computeMetric,
  createSensitivityContext,
  runScenarioPass,
  runSensitivityPass,
  runSensitivityProbe,
  runSync,
  SAMPLING_SPEC,
  sampleBoundedParameter,
  sanitizeConfig,
  sensitivityProbeValues,
  sensitivitySeedOffset,
  SCENARIO_DEFINITIONS,
  SCENARIO_PRESETS,
  simulateChunk,
  summarizeMatrices,
} from './simulation-engine';
import {
  DEFAULT_CONFIG,
  PARAMETER_KEYS,
  PARAMETER_LIMITS,
  type ParameterKey,
  type ScenarioConfig,
} from './types';

const ADOPTION_LAG = Math.exp(-0.105);
const SAVINGS_GROWTH = 1.022;

function relativeExpect(actual: number, expected: number, tolerance = 1e-9): void {
  expect(
    Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)),
    `expected ${actual} ≈ ${expected}`,
  ).toBe(true);
}

/** Mirror of the production draw sequence for one path (interior config). */
function replaySampledParameters(config: ScenarioConfig, seed: number) {
  const rng = new FastPRNG(seed);
  const sWater = Math.max(
    0,
    config.startingWaterUse * Math.exp(rng.nextNormal(0, SAMPLING_SPEC.startingWaterUse.spread)),
  );
  const gAI = sampleBoundedParameter(
    rng,
    config.aiGrowthRate,
    SAMPLING_SPEC.aiGrowthRate.spread,
    0,
    0.3,
  );
  const effGain = sampleBoundedParameter(
    rng,
    config.efficiencyImprovement,
    SAMPLING_SPEC.efficiencyImprovement.spread,
    0,
    0.12,
  );
  const sPot = Math.max(
    0,
    config.savingsPotential * Math.exp(rng.nextNormal(0, SAMPLING_SPEC.savingsPotential.spread)),
  );
  const intEff = sampleBoundedParameter(
    rng,
    config.interventionEffectiveness,
    SAMPLING_SPEC.interventionEffectiveness.spread,
    0,
    1,
  );
  const maxAdopt = sampleBoundedParameter(
    rng,
    config.adoptionRate,
    SAMPLING_SPEC.adoptionRate.spread,
    0,
    1,
  );
  const realFactor = sampleBoundedParameter(
    rng,
    config.realizationFactor,
    SAMPLING_SPEC.realizationFactor.spread,
    0,
    1,
  );
  const attrFrac = sampleBoundedParameter(
    rng,
    config.attributionFraction,
    SAMPLING_SPEC.attributionFraction.spread,
    0,
    1,
  );
  return { sWater, gAI, effGain, sPot, intEff, maxAdopt, realFactor, attrFrac };
}

/** Annual (non-cumulative) savings of the documented equation at year t. */
function annualSavings(k: number, t: number): number {
  return k * (SAVINGS_GROWTH ** t - (SAVINGS_GROWTH * ADOPTION_LAG) ** t);
}

/** Annual (non-cumulative) burden of the documented equation at year t. */
function annualBurden(sWater: number, r: number, t: number): number {
  return sWater * r ** t;
}

describe('path equations (production simulateChunk vs closed form)', () => {
  const seed = 271828;
  const horizon = 50;
  const config: ScenarioConfig = { ...DEFAULT_CONFIG, simulations: 1, horizon, seed };
  const matrices = allocateMatrices(1, horizon);
  const sampled = replaySampledParameters(config, seed);

  runSync(config, matrices, allocateExtraction(1, horizon));

  const k =
    sampled.sPot * sampled.maxAdopt * sampled.intEff * sampled.realFactor * sampled.attrFrac;
  const r = (1 + sampled.gAI) * (1 - sampled.effGain);

  it('uses first-year growth factors in year 1 (not second-year)', () => {
    // W_s(1) = S₀·1.022·(1−q)·A·E·R·F — a pre-multiplied accumulator bug
    // would produce 1.022²−(1.022q)² here instead.
    relativeExpect(matrices.savings[0], annualSavings(k, 1));
    relativeExpect(matrices.burden[0], annualBurden(sampled.sWater, r, 1));
  });

  it('matches the documented equation in year 2', () => {
    relativeExpect(matrices.savings[1], annualSavings(k, 1) + annualSavings(k, 2));
    relativeExpect(matrices.burden[1], annualBurden(sampled.sWater, r, 1) + annualBurden(sampled.sWater, r, 2));
  });

  it('matches the documented recurrence across all subsequent years', () => {
    let cumSavings = 0;
    let cumBurden = 0;
    for (let t = 1; t <= horizon; t++) {
      cumSavings += annualSavings(k, t);
      cumBurden += annualBurden(sampled.sWater, r, t);
      relativeExpect(matrices.savings[t - 1], cumSavings);
      relativeExpect(matrices.burden[t - 1], cumBurden);
    }
  });

  it('has cumulative savings equal to the sum of annual increments', () => {
    const increments = Array.from({ length: horizon }, (_, t) => annualSavings(k, t + 1));
    for (let t = 1; t <= horizon; t++) {
      relativeExpect(matrices.savings[t - 1], increments.slice(0, t).reduce((a, b) => a + b, 0), 1e-8);
    }
  });

  it('computes the net balance as cumulative savings minus cumulative burden', () => {
    for (let t = 1; t <= horizon; t++) {
      relativeExpect(matrices.balance[t - 1], matrices.savings[t - 1] - matrices.burden[t - 1], 1e-8);
    }
  });
});

describe('parameter directions (production summaries)', () => {
  const run = (overrides: Partial<ScenarioConfig>) => {
    const config = sanitizeConfig({ ...DEFAULT_CONFIG, simulations: 400, horizon: 30, ...overrides });
    const matrices = allocateMatrices(config.simulations, config.horizon);
    const buffers = allocateExtraction(config.simulations, config.horizon);
    return summarizeMatrices(runSync(config, matrices, buffers), config, buffers);
  };

  it.each<ParameterKey>([
    'interventionEffectiveness',
    'adoptionRate',
    'realizationFactor',
    'attributionFraction',
  ])('savings increase when %s goes from 0 to 1', (key) => {
    const off = run({ [key]: 0 });
    const on = run({ [key]: 1 });
    expect(off.finalMedianSavings).toBe(0);
    expect(on.finalMedianSavings).toBeGreaterThan(0);
  });

  it('burden increases with the AI growth rate', () => {
    expect(run({ aiGrowthRate: 0.2 }).finalMedianBurden).toBeGreaterThan(
      run({ aiGrowthRate: 0 }).finalMedianBurden,
    );
  });

  it('burden decreases with efficiency improvement', () => {
    expect(run({ efficiencyImprovement: 0.08 }).finalMedianBurden).toBeLessThan(
      run({ efficiencyImprovement: 0 }).finalMedianBurden,
    );
  });

  it('a zero starting water use removes the burden entirely', () => {
    expect(run({ startingWaterUse: 0 }).finalMedianBurden).toBe(0);
  });

  it('a zero savings potential removes the savings entirely', () => {
    expect(run({ savingsPotential: 0 }).finalMedianSavings).toBe(0);
  });
});

describe('sustained break-even classification', () => {
  const classify = (balances: number[]): number =>
    classifySustainedBreakEven(Float64Array.from(balances), 0, balances.length);

  it('qualifies at year 1 when the path is nonnegative throughout', () => {
    expect(classify([0, 1, 2, 3])).toBe(1);
    expect(classify([5, 5, 5])).toBe(1);
  });

  it('qualifies at year 2 when only year 1 is negative', () => {
    expect(classify([-1, 0, 5, 5])).toBe(2);
  });

  it('does not classify a temporary crossing as sustained', () => {
    // Positive in years 1–2, negative again in year 3, nonnegative from 4 on.
    expect(classify([1, 2, -1, 3])).toBe(4);
  });

  it('never reaches break-even when the final year is negative', () => {
    expect(classify([1, 2, 3, -1])).toBe(-1);
    expect(classify([-1, -2, -3])).toBe(-1);
  });

  it('treats a balance of exactly zero as nonnegative', () => {
    expect(classify([-1, 0, 0, 0])).toBe(2);
    expect(classify([0, 0, 0])).toBe(1);
  });

  it('classifies end-to-end ensembles', () => {
    // Savings always win: every path crosses at year 1.
    const dominant = sanitizeConfig({
      ...DEFAULT_CONFIG,
      startingWaterUse: 0,
      simulations: 50,
      horizon: 10,
    });
    const m1 = allocateMatrices(dominant.simulations, dominant.horizon);
    runSync(dominant, m1, allocateExtraction(dominant.simulations, dominant.horizon));
    for (let i = 0; i < dominant.simulations; i++) expect(m1.breakEvenYears[i]).toBe(1);

    // Burden always wins: no path ever crosses.
    const losing = sanitizeConfig({
      ...DEFAULT_CONFIG,
      savingsPotential: 0,
      simulations: 50,
      horizon: 10,
    });
    const m2 = allocateMatrices(losing.simulations, losing.horizon);
    runSync(losing, m2, allocateExtraction(losing.simulations, losing.horizon));
    for (let i = 0; i < losing.simulations; i++) expect(m2.breakEvenYears[i]).toBe(-1);
  });
});

describe('zero-boundary sampling', () => {
  it('returns the bound deterministically when the location equals it', () => {
    const rng = new FastPRNG(5);
    const consumed = new FastPRNG(5);
    expect(sampleBoundedParameter(rng, 0, 0.1, 0, 1)).toBe(0);
    expect(sampleBoundedParameter(rng, 1, 0.1, 0, 1)).toBe(1);
    // No randomness consumed at the bounds.
    expect(rng.nextUint32()).toBe(consumed.nextUint32());
  });

  it('preserves truncated-normal behavior in the interior', () => {
    const a = new FastPRNG(11);
    const b = new FastPRNG(11);
    expect(sampleBoundedParameter(a, 0.5, 0.1, 0, 1)).toBe(b.nextTruncatedNormal(0.5, 0.1, 0, 1));
  });

  it('disables each share contribution when configured at zero', () => {
    for (const key of [
      'interventionEffectiveness',
      'adoptionRate',
      'realizationFactor',
      'attributionFraction',
    ] as ParameterKey[]) {
      const config = sanitizeConfig({
        ...DEFAULT_CONFIG,
        simulations: 10,
        horizon: 5,
        [key]: 0,
      });
      const matrices = allocateMatrices(config.simulations, config.horizon);
      runSync(config, matrices, allocateExtraction(config.simulations, config.horizon));
      for (let i = 0; i < matrices.savings.length; i++) expect(matrices.savings[i]).toBe(0);
    }
  });

  it('pins the upper bound so a share of 1 is always sampled as 1', () => {
    // All six truncated-normal parameters at bounds → fully deterministic
    // sampling; only the two log-normal multipliers are random.
    const seed = 31;
    const horizon = 6;
    const config: ScenarioConfig = {
      ...DEFAULT_CONFIG,
      simulations: 1,
      horizon,
      seed,
      aiGrowthRate: 0,
      efficiencyImprovement: 0,
      interventionEffectiveness: 1,
      adoptionRate: 1,
      realizationFactor: 1,
      attributionFraction: 1,
    };
    const matrices = allocateMatrices(1, horizon);
    runSync(config, matrices, allocateExtraction(1, horizon));

    const rng = new FastPRNG(seed);
    const sWater = config.startingWaterUse * Math.exp(rng.nextNormal(0, 0.12));
    const sPot = config.savingsPotential * Math.exp(rng.nextNormal(0, 0.16));
    for (let t = 1; t <= horizon; t++) {
      relativeExpect(matrices.burden[t - 1], sWater * t); // r = 1, no growth, no efficiency
      relativeExpect(matrices.savings[t - 1], sumAnnual(sPot, t), 1e-8);
    }
  });

  it('keeps log-normal parameters at exactly zero when the base is zero', () => {
    const config = sanitizeConfig({
      ...DEFAULT_CONFIG,
      startingWaterUse: 0,
      savingsPotential: 0,
      simulations: 10,
      horizon: 5,
    });
    const matrices = allocateMatrices(config.simulations, config.horizon);
    runSync(config, matrices, allocateExtraction(config.simulations, config.horizon));
    for (let i = 0; i < matrices.burden.length; i++) expect(matrices.burden[i]).toBe(0);
    for (let i = 0; i < matrices.savings.length; i++) expect(matrices.savings[i]).toBe(0);
  });
});

/** Sum of annual savings terms with k = sPot and all shares pinned at 1. */
function sumAnnual(sPot: number, t: number): number {
  let total = 0;
  for (let u = 1; u <= t; u++) total += annualSavings(sPot, u);
  return total;
}

describe('reproducibility and chunking', () => {
  const config = sanitizeConfig({ ...DEFAULT_CONFIG, simulations: 500, horizon: 20, seed: 777 });

  it('reproduces identical results for the same seed and configuration', () => {
    const a = allocateMatrices(config.simulations, config.horizon);
    const b = allocateMatrices(config.simulations, config.horizon);
    runSync(config, a, allocateExtraction(config.simulations, config.horizon));
    runSync(config, b, allocateExtraction(config.simulations, config.horizon));
    expect(a.burden).toEqual(b.burden);
    expect(a.savings).toEqual(b.savings);
    expect(a.balance).toEqual(b.balance);
    expect(a.breakEvenYears).toEqual(b.breakEvenYears);
  });

  it('produces different ensembles for different seeds', () => {
    const a = allocateMatrices(config.simulations, config.horizon);
    const b = allocateMatrices(config.simulations, config.horizon);
    runSync(config, a, allocateExtraction(config.simulations, config.horizon));
    runSync(
      { ...config, seed: config.seed + 1 },
      b,
      allocateExtraction(config.simulations, config.horizon),
    );
    expect(a.balance).not.toEqual(b.balance);
  });

  it('produces identical results when chunked', () => {
    const single = allocateMatrices(config.simulations, config.horizon);
    runSync(config, single, allocateExtraction(config.simulations, config.horizon));

    const chunked = allocateMatrices(config.simulations, config.horizon);
    const rng = new FastPRNG(config.seed);
    simulateChunk(config, chunked, rng, 0, 250);
    simulateChunk(config, chunked, rng, 250, 500);
    expect(single.burden).toEqual(chunked.burden);
    expect(single.savings).toEqual(chunked.savings);
    expect(single.balance).toEqual(chunked.balance);
    expect(single.breakEvenYears).toEqual(chunked.breakEvenYears);
  });
});

describe('summary statistics', () => {
  it('counts break-even years and never-reached paths consistently', () => {
    const config = sanitizeConfig({ ...DEFAULT_CONFIG, simulations: 250, horizon: 50, seed: 4242 });
    const matrices = allocateMatrices(config.simulations, config.horizon);
    const buffers = allocateExtraction(config.simulations, config.horizon);
    const summary = summarizeMatrices(runSync(config, matrices, buffers), config, buffers);

    const counted = Array.from(summary.breakEvenYearCounts).reduce((a, b) => a + b, 0);
    expect(counted + summary.neverCount).toBe(config.simulations);
    expect(summary.neverCount / config.simulations).toBeGreaterThanOrEqual(0);
    expect(summary.neverCount / config.simulations).toBeLessThanOrEqual(1);
  });

  it('reports nondecreasing cumulative probabilities', () => {
    const config = sanitizeConfig({ ...DEFAULT_CONFIG, simulations: 250, horizon: 50, seed: 99 });
    const matrices = allocateMatrices(config.simulations, config.horizon);
    const buffers = allocateExtraction(config.simulations, config.horizon);
    const summary = summarizeMatrices(runSync(config, matrices, buffers), config, buffers);
    const { y10, y20, y30, y50 } = summary.probabilities;
    expect(y10).toBeLessThanOrEqual(y20);
    expect(y20).toBeLessThanOrEqual(y30);
    expect(y30).toBeLessThanOrEqual(y50);
    expect(y50).toBeCloseTo(1 - summary.neverCount / config.simulations, 12);
  });

  it('keeps a fractional median crossing year (even reached count)', () => {
    const horizon = 3;
    const matrices = allocateMatrices(2, horizon);
    // Path 1 crosses at year 2, path 2 crosses at year 3.
    matrices.balance.set([-1, 1, 1, -1, -1, 1]);
    matrices.breakEvenYears.set([2, 3]);
    const buffers = allocateExtraction(2, horizon);
    const summary = summarizeMatrices(matrices, sanitizeConfig({ ...DEFAULT_CONFIG, simulations: 2, horizon }), buffers);
    expect(summary.medianBreakEvenYear).toBe(2.5);
  });
});

describe('mean capped crossing year metric', () => {
  const horizon = 10;

  const matricesWithYears = (years: number[]) => {
    const matrices = allocateMatrices(years.length, horizon);
    matrices.breakEvenYears.set(years);
    return matrices;
  };

  it('caps non-crossing paths at the horizon', () => {
    const matrices = matricesWithYears([3, 5, -1, 10]);
    const buffers = allocateExtraction(4, horizon);
    expect(computeMetric(matrices, 'meanCappedYear', buffers)).toBeCloseTo((3 + 5 + 10 + 10) / 4, 12);
  });

  it('equals the horizon when no path crosses', () => {
    const matrices = matricesWithYears([-1, -1, -1]);
    const buffers = allocateExtraction(3, horizon);
    expect(computeMetric(matrices, 'meanCappedYear', buffers)).toBe(horizon);
  });

  it('equals the mean crossing year when every path crosses', () => {
    const matrices = matricesWithYears([1, 2, 6]);
    const buffers = allocateExtraction(3, horizon);
    expect(computeMetric(matrices, 'meanCappedYear', buffers)).toBeCloseTo(3, 12);
  });

  it('never exceeds the horizon (no artificial event years)', () => {
    const matrices = matricesWithYears([-1, -1, -1, -1, -1]);
    const buffers = allocateExtraction(5, horizon);
    expect(computeMetric(matrices, 'meanCappedYear', buffers)).toBeLessThanOrEqual(horizon);
  });
});

describe('sensitivity probes', () => {
  it('keeps every probe within PARAMETER_LIMITS', () => {
    const bases = [0, 0.5, 1];
    for (const key of PARAMETER_KEYS) {
      const { min, max } = PARAMETER_LIMITS[key];
      const span = max - min;
      for (const fraction of bases) {
        const base = min + span * fraction;
        const { low, high } = sensitivityProbeValues(key, base);
        expect(low).toBeGreaterThanOrEqual(min);
        expect(low).toBeLessThanOrEqual(max);
        expect(high).toBeGreaterThanOrEqual(min);
        expect(high).toBeGreaterThanOrEqual(low);
        expect(high).toBeLessThanOrEqual(max);
      }
    }
  });

  it('gives zero-valued parameters a meaningful nonzero probe', () => {
    for (const key of PARAMETER_KEYS) {
      const { low, high } = sensitivityProbeValues(key, PARAMETER_LIMITS[key].min);
      expect(high).toBeGreaterThan(low);
    }
  });

  it('respects the slider maximum for growth and efficiency probes', () => {
    expect(sensitivityProbeValues('aiGrowthRate', 0.2).high).toBeLessThanOrEqual(0.2);
    expect(sensitivityProbeValues('efficiencyImprovement', 0.08).high).toBeLessThanOrEqual(0.08);
    expect(sensitivityProbeValues('savingsPotential', 2000).high).toBeLessThanOrEqual(2000);
    expect(sensitivityProbeValues('startingWaterUse', 800).high).toBeLessThanOrEqual(800);
  });

  it('reports probe values within limits end-to-end', () => {
    for (const config of [
      allAtMinimum(),
      allAtMaximum(),
      sanitizeConfig(DEFAULT_CONFIG),
    ]) {
      const { results } = runSensitivityPass({ ...config, simulations: 250 }, 'finalBalance');
      expect(results).toHaveLength(PARAMETER_KEYS.length);
      for (const result of results) {
        const { min, max } = PARAMETER_LIMITS[result.key];
        expect(result.lowValue).toBeGreaterThanOrEqual(min);
        expect(result.lowValue).toBeLessThanOrEqual(max);
        expect(result.highValue).toBeGreaterThanOrEqual(min);
        expect(result.highValue).toBeLessThanOrEqual(max);
        expect(Number.isFinite(result.low)).toBe(true);
        expect(Number.isFinite(result.high)).toBe(true);
        expect(Number.isFinite(result.baseline)).toBe(true);
      }
    }
  });

  it('uses paired deterministic seeds for the low and high runs', () => {
    const config = sanitizeConfig({ ...DEFAULT_CONFIG, simulations: 250, horizon: 20, seed: 555 });
    const metric = 'finalBalance' as const;
    const context = createSensitivityContext(config, metric);
    const index = 4; // adoptionRate
    const result = runSensitivityProbe(context, 'adoptionRate', index, metric);

    // Reproduce both probes manually with the documented shared seed offset.
    const offset = sensitivitySeedOffset(index);
    const { low, high } = sensitivityProbeValues('adoptionRate', config.adoptionRate);
    const matrices = allocateMatrices(context.probeConfig.simulations, context.probeConfig.horizon);
    const buffers = allocateExtraction(context.probeConfig.simulations, context.probeConfig.horizon);
    const expectedLow = computeMetric(
      runSync({ ...context.probeConfig, adoptionRate: low }, matrices, buffers, offset),
      metric,
      buffers,
    );
    const expectedHigh = computeMetric(
      runSync({ ...context.probeConfig, adoptionRate: high }, matrices, buffers, offset),
      metric,
      buffers,
    );
    expect(result.low).toBeCloseTo(expectedLow, 12);
    expect(result.high).toBeCloseTo(expectedHigh, 12);
  });

  it('is deterministic for a fixed configuration and seed', () => {
    const config = sanitizeConfig({ ...DEFAULT_CONFIG, simulations: 250, horizon: 15, seed: 8 });
    const a = runSensitivityPass(config, 'meanCappedYear');
    const b = runSensitivityPass(config, 'meanCappedYear');
    expect(a.results).toEqual(b.results);
  });
});

describe('scenario consistency', () => {
  const extremeConfigs = [
    sanitizeConfig(DEFAULT_CONFIG),
    allAtMinimum(),
    allAtMaximum(),
    sanitizeConfig({ ...DEFAULT_CONFIG, adoptionRate: 1, efficiencyImprovement: 0.08, aiGrowthRate: 0.2 }),
    sanitizeConfig({ ...DEFAULT_CONFIG, realizationFactor: 0, attributionFraction: 0 }),
  ];

  it('keeps every preset within PARAMETER_LIMITS', () => {
    for (const preset of SCENARIO_PRESETS) {
      for (const [key, value] of Object.entries(preset.values)) {
        const { min, max } = PARAMETER_LIMITS[key as ParameterKey];
        expect(value, `${preset.id}.${key}`).toBeGreaterThanOrEqual(min);
        expect(value, `${preset.id}.${key}`).toBeLessThanOrEqual(max);
      }
    }
  });

  it('keeps every derived scenario within PARAMETER_LIMITS for extreme bases', () => {
    for (const config of extremeConfigs) {
      for (const definition of SCENARIO_DEFINITIONS) {
        const derived = definition.derive(config);
        for (const key of PARAMETER_KEYS) {
          const { min, max } = PARAMETER_LIMITS[key];
          expect(derived[key], `${definition.name}.${key}`).toBeGreaterThanOrEqual(min);
          expect(derived[key], `${definition.name}.${key}`).toBeLessThanOrEqual(max);
        }
      }
    }
  });

  it('sanitizes derived scenarios without changing them (UI == worker config)', () => {
    for (const config of extremeConfigs) {
      for (const definition of SCENARIO_DEFINITIONS) {
        const derived = definition.derive(config);
        expect(sanitizeConfig(derived)).toEqual(derived);
      }
    }
  });

  it('runs every scenario at the capped simulation count', () => {
    const config = sanitizeConfig({ ...DEFAULT_CONFIG, simulations: 250, horizon: 12, seed: 21 });
    const outcomes = runScenarioPass(config);
    expect(outcomes).toHaveLength(SCENARIO_DEFINITIONS.length);
    for (const outcome of outcomes) {
      expect(outcome.medianBalance).toHaveLength(config.horizon);
      expect(Number.isFinite(outcome.finalMedianBalance)).toBe(true);
      expect(Number.isFinite(outcome.probability30)).toBe(true);
    }
  });
});

describe('sanitizeConfig', () => {
  it('clamps every field into PARAMETER_LIMITS', () => {
    const wild: ScenarioConfig = sanitizeConfig({
      ...DEFAULT_CONFIG,
      startingWaterUse: 5000,
      aiGrowthRate: 5,
      efficiencyImprovement: -3,
      savingsPotential: -100,
      interventionEffectiveness: 2,
      adoptionRate: -1,
      realizationFactor: Number.NaN,
      attributionFraction: 7,
      horizon: 3,
      simulations: 1,
      seed: 0,
    });
    for (const key of [...PARAMETER_KEYS, 'horizon', 'simulations', 'seed'] as const) {
      const { min, max } = PARAMETER_LIMITS[key];
      expect(wild[key]).toBeGreaterThanOrEqual(min);
      expect(wild[key]).toBeLessThanOrEqual(max);
    }
  });

  it('is idempotent', () => {
    const once = sanitizeConfig({ ...DEFAULT_CONFIG, startingWaterUse: 9999 });
    expect(sanitizeConfig(once)).toEqual(once);
  });
});

function allAtMinimum(): ScenarioConfig {
  const config = { ...DEFAULT_CONFIG };
  for (const key of PARAMETER_KEYS) config[key] = PARAMETER_LIMITS[key].min;
  return sanitizeConfig(config);
}

function allAtMaximum(): ScenarioConfig {
  const config = { ...DEFAULT_CONFIG };
  for (const key of PARAMETER_KEYS) config[key] = PARAMETER_LIMITS[key].max;
  return sanitizeConfig(config);
}
