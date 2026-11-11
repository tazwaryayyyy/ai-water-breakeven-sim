/**
 * CSV / JSON / high-DPI PNG export for the current ensemble.
 *
 * The CSV uses one rectangular tidy schema (record, series, year, statistic,
 * value, unit) covering quantile trajectories, the break-even distribution,
 * scenario results, run metadata, and the model assumptions. JSON snapshots
 * the full model state — including sampling assumptions and calculation
 * definitions — for reproducibility; PNG re-renders the main chart SVG onto
 * an offscreen canvas at 3× scale.
 */

import { useCallback } from 'react';
import { SAMPLING_SPEC } from '@/domain/model/simulation-engine';
import { MODEL_VERSION, type SimulationResult } from '@/domain/model/types';

const PNG_SCALE = 3;

const CSV_HEADER = ['record', 'series', 'year', 'statistic', 'value', 'unit'];

/** RFC 4180 field escaping: quote when needed, double embedded quotes. */
export function csvEscape(field: string | number): string {
  const text = String(field);
  if (/[",\r\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }
  return text;
}

const ASSUMPTION_UNITS: Record<string, string> = {
  startingWaterUse: 'Mm3/yr',
  aiGrowthRate: 'fraction/yr',
  efficiencyImprovement: 'fraction/yr',
  savingsPotential: 'Mm3/yr',
  interventionEffectiveness: 'fraction',
  adoptionRate: 'fraction',
  realizationFactor: 'fraction',
  attributionFraction: 'fraction',
};

function csvRow(fields: Array<string | number>): string {
  return fields.map(csvEscape).join(',');
}

/** Build the rectangular tidy CSV for the current ensemble. */
export function buildEnsembleCSV(result: SimulationResult): string {
  const { summary } = result;
  const { years, burden, savings, balance, breakEvenCounts, config } = summary;
  const rows: Array<Array<string | number>> = [CSV_HEADER];

  rows.push(['meta', 'run', '', 'model_version', MODEL_VERSION, '']);
  rows.push(['meta', 'run', '', 'seed', config.seed, '']);
  rows.push(['meta', 'run', '', 'horizon', config.horizon, 'years']);
  rows.push(['meta', 'run', '', 'simulations', config.simulations, 'paths']);
  rows.push(['meta', 'run', '', 'volumes', 'millions of cubic meters', '']);
  rows.push([
    'meta',
    'run',
    '',
    'full_payload',
    'See the accompanying JSON export for the complete reproducibility payload.',
    '',
  ]);

  for (const [key, value] of Object.entries(ASSUMPTION_UNITS)) {
    rows.push(['assumption', key, '', 'value', config[key as keyof typeof config], value]);
  }

  const quantileStats = ['p5', 'p25', 'median', 'p75', 'p95'] as const;
  const trajectories: Array<['burden' | 'savings' | 'balance', typeof burden]> = [
    ['burden', burden],
    ['savings', savings],
    ['balance', balance],
  ];
  for (let i = 0; i < years.length; i++) {
    for (const [series, percentiles] of trajectories) {
      for (const statistic of quantileStats) {
        rows.push(['trajectory', series, years[i], statistic, percentiles[statistic][i], 'Mm3']);
      }
    }
  }

  for (let year = 1; year <= breakEvenCounts.length; year++) {
    rows.push(['break_even', 'crossing_year', year, 'count', breakEvenCounts[year - 1], 'paths']);
  }
  rows.push(['break_even', 'never', '', 'count', summary.neverCount, 'paths']);

  for (const scenario of result.scenarios) {
    for (let i = 0; i < years.length; i++) {
      rows.push(['scenario', scenario.name, years[i], 'median_balance', scenario.medianBalance[i], 'Mm3']);
    }
    // The engine clamps probability cutoffs to the horizon; label the actual cutoffs.
    const cutoff30 = Math.min(30, config.horizon);
    const cutoff50 = Math.min(50, config.horizon);
    rows.push(['scenario', scenario.name, '', `break_even_by_year_${cutoff30}`, scenario.probability30, 'fraction']);
    if (cutoff50 > cutoff30) {
      rows.push(['scenario', scenario.name, '', `break_even_by_year_${cutoff50}`, scenario.probability50, 'fraction']);
    }
    if (scenario.medianBreakEvenYear !== null) {
      rows.push(['scenario', scenario.name, '', 'median_crossing_year', scenario.medianBreakEvenYear, 'years']);
    }
    rows.push(['scenario', scenario.name, '', 'final_median_balance', scenario.finalMedianBalance, 'Mm3']);
    rows.push(['scenario', scenario.name, '', 'final_median_burden', scenario.finalMedianBurden, 'Mm3']);
  }

  return rows.map(csvRow).join('\n');
}

/** Build the reproducibility JSON payload (serialize with JSON.stringify). */
export function buildReproducibilityPayload(result: SimulationResult) {
  return {
    generatedAt: new Date().toISOString(),
    modelVersion: MODEL_VERSION,
    engine: {
      prng: 'xoshiro128** (SplitMix32 seeded, Math.imul, no BigInt)',
      sustainedBreakEven:
        'backwards scan for the last negative year; balance exactly zero is nonnegative',
      quantiles: 'in-place Float64Array sort, Hyndman-Fan type 7',
      burdenEquation: 'W_b(t) = W0 * (1 + g_ai)^t * (1 - e_water)^t (cumulative sum)',
      savingsEquation:
        'W_s(t) = S0 * 1.022^t * (1 - e^(-0.105*t)) * adoption * effectiveness * realization * attribution (cumulative sum)',
      sensitivity: {
        method: 'one-at-a-time low/high probes with paired deterministic seeds',
        metrics: {
          finalBalance: 'median final-year net balance across paths',
          probability30: 'share of paths with sustained crossing by year min(30, horizon)',
          probability50: 'share of paths with sustained crossing by year min(50, horizon)',
          meanCappedYear: 'mean of min(crossing year, horizon) over all paths',
        },
      },
      sampling: SAMPLING_SPEC,
      samplingNotes: [
        'Parameters are sampled independently; each sampled parameter is held constant along its path.',
        'Log-normal base values are distribution medians; truncated-normal base values are distribution locations, not realized means.',
        'A parameter configured exactly at a sampling bound is drawn deterministically at that bound.',
        'The model has no year-to-year random shocks and no correlation between assumptions.',
      ],
    },
    reproducibility: {
      note: `The same seed and configuration reproduce these results exactly under model version ${MODEL_VERSION}; no reproducibility is implied across future engine versions.`,
    },
    config: result.summary.config,
    summary: result.summary,
    sensitivity: result.sensitivity,
    scenarios: result.scenarios,
    computeMs: result.computeMs,
  };
}

function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function useExport(result: SimulationResult | null) {
  const exportCSV = useCallback(() => {
    if (!result) return;
    downloadBlob(
      new Blob([buildEnsembleCSV(result)], { type: 'text/csv;charset=utf-8' }),
      'ai-water-break-even-ensemble.csv',
    );
  }, [result]);

  const exportJSON = useCallback(() => {
    if (!result) return;
    downloadBlob(
      new Blob([JSON.stringify(buildReproducibilityPayload(result), null, 2)], {
        type: 'application/json',
      }),
      'ai-water-break-even-simulation.json',
    );
  }, [result]);

  const exportPNG = useCallback(
    (selector: string, filename = 'ai-water-break-even-chart.png') => {
      const svg = document.querySelector<SVGSVGElement>(`${selector} svg`);
      if (!svg) return;
      const width = svg.clientWidth || 1200;
      const height = svg.clientHeight || 640;
      const clone = svg.cloneNode(true) as SVGSVGElement;
      clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
      clone.setAttribute('width', String(width));
      clone.setAttribute('height', String(height));
      const serialized = new XMLSerializer().serializeToString(clone);
      const svgUrl = URL.createObjectURL(
        new Blob([serialized], { type: 'image/svg+xml;charset=utf-8' }),
      );
      const image = new Image();
      image.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = width * PNG_SCALE;
        canvas.height = height * PNG_SCALE;
        const context = canvas.getContext('2d');
        if (context) {
          context.scale(PNG_SCALE, PNG_SCALE);
          context.fillStyle = '#fffdf7';
          context.fillRect(0, 0, width, height);
          context.drawImage(image, 0, 0, width, height);
          canvas.toBlob((blob) => {
            if (blob) downloadBlob(blob, filename);
          }, 'image/png');
        }
        URL.revokeObjectURL(svgUrl);
      };
      image.onerror = () => URL.revokeObjectURL(svgUrl);
      image.src = svgUrl;
    },
    [],
  );

  return { exportCSV, exportJSON, exportPNG };
}
