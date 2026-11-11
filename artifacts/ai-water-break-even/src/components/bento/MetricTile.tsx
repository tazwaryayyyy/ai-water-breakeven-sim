import { Tile } from '@/components/ui/Tile';
import {
  buildNarrative,
  formatInt,
  formatPercent,
  formatWater,
  physicalEquivalents,
  formatPools,
  formatLakeShare,
  formatCityYears,
} from '@/lib/format';
import {
  applicableMilestones,
  probabilityAt,
  type SimulationResult,
  type SensitivityResult,
} from '@/domain/model/types';

interface MetricTileProps {
  result: SimulationResult | null;
  sensitivity: SensitivityResult[];
}

function Kpi({
  value,
  label,
  detail,
  accent = false,
}: {
  value: string;
  label: string;
  detail?: string;
  accent?: boolean;
}) {
  return (
    <div className={accent ? 'border-t-2 border-teal pt-3' : 'border-t border-line pt-3'}>
      <div className="font-display text-[44px] leading-[1.05] tracking-tight text-ink tabular-nums xl:text-[56px]">
        {value}
      </div>
      <div className="mt-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-ink-soft">
        {label}
      </div>
      {detail && <div className="mt-1 text-[11px] leading-snug text-muted">{detail}</div>}
    </div>
  );
}

/**
 * Executive hero: sustained break-even probabilities, median crossing,
 * and the physical, human-scale reading of the median position.
 */
export function MetricTile({ result, sensitivity }: MetricTileProps) {
  if (!result) {
    return (
      <Tile
        index="01"
        title="Executive summary"
        note="Waiting for the first ensemble to arrive from the worker."
        className="xl:col-span-8"
      >
        <div className="grid gap-6 md:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <div key={i} className="border-t border-line pt-3">
              <div className="h-14 w-32 animate-pulse bg-ink/[0.06]" />
              <div className="mt-3 h-2.5 w-24 animate-pulse bg-ink/[0.06]" />
            </div>
          ))}
        </div>
      </Tile>
    );
  }

  const { summary } = result;
  const horizon = summary.config.horizon;
  const evalYear = Math.min(30, horizon);
  const balanceAtEval = summary.balance.median[evalYear - 1] ?? 0;
  const equivalents = physicalEquivalents(balanceAtEval);
  const dominant = sensitivity[0]?.label ?? null;
  const narrative = buildNarrative(
    summary.neverProbability,
    summary.medianBreakEvenYear,
    summary.finalMedianBalance,
    dominant,
    horizon,
  );
  // Only milestones within the simulated horizon describe a simulated period.
  const milestones = applicableMilestones(horizon);
  const primaryMilestone = milestones.includes(30) ? 30 : milestones[milestones.length - 1];
  const hasFifty = milestones.includes(50);

  return (
    <Tile
      index="01"
      title="Executive summary"
      note="A path breaks even at year T only if its net balance stays ≥ 0 from T to the horizon."
      className="xl:col-span-8"
      footer={
        <span className="tabular-nums">
          {formatInt(summary.config.simulations)} paths · {horizon}-year horizon ·
          seed {formatInt(summary.config.seed)} · computed in {Math.round(result.computeMs)} ms
        </span>
      }
    >
      <div className="grid gap-x-8 gap-y-6 sm:grid-cols-3">
        <Kpi
          accent
          value={formatPercent(probabilityAt(summary.probabilities, primaryMilestone))}
          label={`Reach break-even by year ${primaryMilestone}`}
          detail={`Share of paths whose sustained crossing lands at or before year ${primaryMilestone}.`}
        />
        <Kpi
          value={
            hasFifty
              ? formatPercent(summary.probabilities.y50)
              : formatPercent(1 - summary.neverProbability)
          }
          label={hasFifty ? 'By year 50' : `By year ${horizon} (horizon end)`}
          detail={`Never crosses within the horizon: ${formatPercent(summary.neverProbability)}.`}
        />
        <Kpi
          value={summary.medianBreakEvenYear !== null ? `Y${summary.medianBreakEvenYear}` : '—'}
          label="Median crossing year"
          detail="Median across paths that reach a sustained crossing."
        />
      </div>

      <div className="mt-6 border-t border-dashed border-line pt-4">
        <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted">
          Physical equivalents · median net position at year {evalYear}
        </p>
        <p className="mt-2 font-display text-[15px] leading-relaxed text-ink">
          {formatWater(balanceAtEval)} M m³ {equivalents.negative ? 'deficit' : 'surplus'} —
          ≈ {formatPools(Math.abs(equivalents.pools))} · {formatLakeShare(equivalents.lakeMeadShare)} ·{' '}
          {formatCityYears(Math.abs(equivalents.cityYears))}
        </p>
      </div>

      <p className="mt-4 max-w-prose font-display text-[14px] italic leading-relaxed text-ink-soft">
        {narrative}
      </p>
    </Tile>
  );
}
