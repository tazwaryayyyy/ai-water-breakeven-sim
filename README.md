# AI Water Break-Even Simulator

A production-grade Monte Carlo research simulator comparing cumulative AI water
burden with water savings credited to AI-enabled interventions. All computation
runs off the main thread in a dedicated Web Worker; the interface is an
editorial, Swiss-inspired research dashboard.

## What it is

Each run samples uncertain assumptions and estimates the share of simulated
paths that reach **sustained break-even** within the selected horizon. A path
breaks even at year T if and only if its cumulative net balance stays ≥ 0 from
T through the end of the horizon — found by scanning backwards for the last
negative year.

The implementation lives in [artifacts/ai-water-break-even](artifacts/ai-water-break-even).
Its defaults are illustrative assumptions, not measured inputs or an empirical
forecast.

## Architecture

```
artifacts/ai-water-break-even/src/
├── domain/
│   ├── math/
│   │   ├── prng.ts                # 32-bit xoshiro128** (SplitMix32 seed, Math.imul, no BigInt)
│   │   └── statistics.ts          # In-place typed-array sorts and quantile extraction
│   ├── model/
│   │   ├── types.ts               # Strict domain + worker wire-protocol types
│   │   └── simulation-engine.ts   # Monte Carlo engine, sensitivity and scenario passes
│   └── worker/
│       ├── simulation.worker.ts   # Worker entry: runs, sensitivity, scenarios
│       └── worker-client.ts       # Typed native postMessage bridge, abortable request IDs
├── components/
│   ├── bento/                     # BentoGrid, MetricTile, ProbabilityTile
│   ├── controls/                  # ParameterScrubber, ScenarioSelector
│   ├── visualization/             # MainBalanceChart, SensitivityTornado, DistributionChart
│   └── ui/                        # Button, Tile primitives
├── hooks/
│   ├── useSimulation.ts           # Worker lifecycle, debounced dispatch, progress
│   └── useExport.ts               # CSV / JSON / high-DPI PNG export
├── App.tsx
└── index.css                      # Tailwind v4 warm editorial theme
```

### Performance properties

- **No main-thread freezing.** Baseline, sensitivity probes (all 8
  parameters), and scenario comparisons run inside the worker over flat
  `Float64Array` / `Int32Array` matrices. The worker yields to its message
  loop between chunks, between sensitivity probes, and between scenarios, so
  a superseding run aborts at the next checkpoint.
- **No RPC library.** The worker bridge uses native `postMessage` with
  Transferable buffers, incrementing request IDs, and a generation-token
  cancellation scheme.
- **Fast PRNG.** xoshiro128** seeded by SplitMix32 — plain 32-bit
  `Math.imul` arithmetic, no BigInt in the hot loop — with Box–Muller
  normals, truncated-normal, and log-normal parameter sampling.
- **Allocation-light summaries.** The per-path loop allocates nothing; year
  slices are sorted in place in a reusable scratch buffer and read at the
  5th/25th/50th/75th/95th percentiles. Summary objects are allocated once
  per run.
- **Responsive scrubbing.** Slider drags coalesce through a 130 ms debounce
  into one worker dispatch per pause; charts render with animations off.

## Path equations

- Burden: `W_b(t) = W₀ · (1 + g_ai)^t · (1 − e_water)^t`
- Savings: `W_s(t) = S₀ · 1.022^t · (adoption · (1 − e^(−0.105 t))) · E_int · R · F_attr`

Both are evaluated by iterative compounding in the inner loop — algebraically
identical, with no `Math.pow` or `Math.exp` per year. Charts and exports show
the *cumulative* sums of these annual terms.

## Sampling assumptions

Each path draws its eight parameters once, independently, and holds them
constant for the whole horizon. There are no year-to-year random shocks, and
no correlation between assumptions is modeled.

| Parameter | Distribution | Spread |
|---|---|---|
| Starting water use | Log-normal multiplier | 0.12 log-space SD |
| AI growth | Truncated normal | 0.012 SD |
| Efficiency improvement | Truncated normal | 0.004 SD |
| Savings potential | Log-normal multiplier | 0.16 log-space SD |
| Intervention effectiveness | Truncated normal | 0.09 SD |
| Adoption rate | Truncated normal | 0.10 SD |
| Realization factor | Truncated normal | 0.09 SD |
| Attribution fraction | Truncated normal | 0.10 SD |

Boundary behavior and interpretation:

- A parameter configured exactly at a sampling bound is drawn
  deterministically at that bound — a configured 0 disables that
  contribution, and a share configured at 1 is always sampled as 1.
- For log-normal parameters the configured base value is the **median** of
  the sampled distribution (the multiplier is log-normal with median 1).
- For truncated-normal parameters the configured value is the distribution
  **location**, not necessarily the realized arithmetic mean after
  truncation.
- Simulated probabilities describe the selected assumptions and
  distributions, not empirically established real-world probabilities.

## What the model counts

The "burden" side is annual **operational AI water use** growing at the AI
growth rate and shrinking with efficiency improvements. The "savings" side is
water that AI-enabled interventions could save, scaled by adoption,
intervention effectiveness, realization, and the fraction credited to AI.
Net balance is cumulative savings minus cumulative burden.

Limitations of this accounting boundary:

- Volumes are aggregated globally with no geography, region, or water-scarcity
  differentiation.
- Water sources (freshwater, seawater, reclaimed, etc.) are not
  distinguished, and burden volumes are not directly comparable to savings
  volumes across sources, locations, or timing.
- The model is not empirically validated: no real-world measurements,
  datasets, or observations inform the defaults, which are illustrative
  assumptions.

## Run locally

From the repository root:

```bash
npx pnpm install --frozen-lockfile
npx pnpm --dir artifacts/ai-water-break-even run dev
```

```bash
npx pnpm --dir artifacts/ai-water-break-even run build
npx pnpm --dir artifacts/ai-water-break-even run typecheck
npx pnpm --dir artifacts/ai-water-break-even run test
```

## Inputs and outputs

Controls cover all eight model parameters plus horizon (10–80 years), run
count (250–20,000), and seed, with one-click presets (Reference Case, High AI
Burden, Aggressive Conservation, Delayed Adoption). Results include percentile
trajectories for burden, savings, and net balance; break-even year
distribution; sensitivity tornado; and Baseline / Accelerated Savings / High
Growth scenario comparison. The app exports full-ensemble CSV, reproducibility
JSON, and a 3× PNG of the main chart.

## Reproducibility

The Monte Carlo generator is seeded. The same configuration, seed, and model
version reproduce the same results exactly; sensitivity probes use
deterministic paired seeds per parameter. No reproducibility is implied
across future versions of the engine — the JSON export records the model
version and full sampling assumptions alongside the configuration.

## Interpreting the statistics

- **Sustained break-even year** — a path crosses at year T only if its net
  balance stays ≥ 0 from T through the horizon.
- **Median crossing year** (executive summary and scenario comparison) — the
  median *among paths that actually reach* a sustained crossing; it is
  undefined when no path crosses.
- **Median trajectory** (main chart) — the year-by-year median of the
  ensemble, whose own sustained crossing is a different statistic from the
  median of individual paths' crossing years.
- **Mean capped crossing year** (sensitivity tornado) — the mean of
  min(crossing year, horizon) over *all* paths, so non-crossing paths are
  capped at the observation horizon rather than assigned an artificial event
  time. Lower means earlier and/or more frequent break-even within the
  horizon.
- Probability milestones (Y10/Y20/Y30/Y50) are displayed only when they fall
  within the selected horizon.

## Scope

This is a scenario-analysis tool. Its inputs are illustrative, and simulated
probabilities describe sampled paths under those assumptions, not real-world
outcomes.

## License

MIT. See [LICENSE](LICENSE).
