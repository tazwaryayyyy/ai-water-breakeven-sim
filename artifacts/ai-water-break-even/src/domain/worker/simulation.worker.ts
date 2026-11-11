/**
 * Dedicated simulation worker. All Monte Carlo passes — baseline,
 * sensitivity probes, and scenario comparison — run here so the main
 * thread never blocks.
 *
 * Cancellation is an event-based generation token: every 'run' message
 * bumps `activeRequestId`; long computations chunk their work and yield to
 * the message loop between chunks, aborting as soon as their token is no
 * longer current. A new run therefore supersedes an in-flight run within
 * one chunk (~2500 paths), with no shared state to corrupt.
 */

import { FastPRNG } from '../math/prng';
import {
  SCENARIO_DEFINITIONS,
  SIMULATION_CHUNK,
  allocateExtraction,
  allocateMatrices,
  createSensitivityContext,
  runScenario,
  runSensitivityProbe,
  sanitizeConfig,
  simulateChunk,
  summarizeMatrices,
  type ScenarioPassOutcome,
} from '../model/simulation-engine';
import { PARAMETER_KEYS } from '../model/types';
import type {
  RunRequestMessage,
  SensitivityResult,
  WorkerInboundMessage,
  WorkerOutboundMessage,
} from '../model/types';

interface WorkerScope {
  postMessage(message: WorkerOutboundMessage, transfer?: Transferable[]): void;
  addEventListener(
    type: 'message',
    listener: (event: MessageEvent<WorkerInboundMessage>) => void,
  ): void;
}

const scope = self as unknown as WorkerScope;

/** Token of the run that currently owns the worker; 0 means idle/cancelled. */
let activeRequestId = 0;

/** Hand control back to the message loop so queued messages can run. */
const yieldToMessages = (): Promise<void> =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, 0);
  });

scope.addEventListener('message', (event: MessageEvent<WorkerInboundMessage>) => {
  const message = event.data;
  if (message.type === 'run') {
    activeRequestId = message.requestId;
    void handleRun(message);
  } else if (message.type === 'cancel' && message.requestId === activeRequestId) {
    activeRequestId = 0;
  }
});

async function handleRun(request: RunRequestMessage): Promise<void> {
  const requestId = request.requestId;
  const isStale = (): boolean => activeRequestId !== requestId;
  const started = performance.now();

  const post = (message: WorkerOutboundMessage, transfer?: Transferable[]): void => {
    scope.postMessage(message, transfer);
  };

  try {
    const config = sanitizeConfig(request.config);

    /* ---------------------------- Baseline pass ---------------------------- */
    const matrices = allocateMatrices(config.simulations, config.horizon);
    const buffers = allocateExtraction(config.simulations, config.horizon);
    const rng = new FastPRNG(config.seed);

    for (let from = 0; from < config.simulations; from += SIMULATION_CHUNK) {
      if (isStale()) return;
      const to = Math.min(from + SIMULATION_CHUNK, config.simulations);
      simulateChunk(config, matrices, rng, from, to);
      post({
        type: 'progress',
        requestId,
        stage: 'baseline',
        completed: to,
        total: config.simulations,
      });
      await yieldToMessages();
    }
    if (isStale()) return;

    const summary = summarizeMatrices(matrices, config, buffers);

    /* -------------------------- Sensitivity pass --------------------------- */
    let sensitivity: SensitivityResult[] = [];
    if (request.includeSensitivity) {
      post({
        type: 'progress',
        requestId,
        stage: 'sensitivity',
        completed: 0,
        total: PARAMETER_KEYS.length,
      });
      await yieldToMessages();
      if (isStale()) return;
      const context = createSensitivityContext(config, request.sensitivityMetric);
      for (let index = 0; index < PARAMETER_KEYS.length; index++) {
        await yieldToMessages();
        if (isStale()) return;
        sensitivity.push(
          runSensitivityProbe(context, PARAMETER_KEYS[index], index, request.sensitivityMetric),
        );
        post({
          type: 'progress',
          requestId,
          stage: 'sensitivity',
          completed: index + 1,
          total: PARAMETER_KEYS.length,
        });
      }
      sensitivity.sort((a, b) => b.influence - a.influence);
    }
    if (isStale()) return;

    /* --------------------------- Scenario pass ----------------------------- */
    let scenarioOutcomes: ScenarioPassOutcome[] = [];
    if (request.includeScenarios) {
      post({ type: 'progress', requestId, stage: 'scenarios', completed: 0, total: SCENARIO_DEFINITIONS.length });
      await yieldToMessages();
      if (isStale()) return;
      for (let index = 0; index < SCENARIO_DEFINITIONS.length; index++) {
        scenarioOutcomes.push(runScenario(config, SCENARIO_DEFINITIONS[index]));
        post({
          type: 'progress',
          requestId,
          stage: 'scenarios',
          completed: index + 1,
          total: SCENARIO_DEFINITIONS.length,
        });
        await yieldToMessages();
        if (isStale()) return;
      }
    }
    if (isStale()) return;

    const payload = {
      config,
      summary,
      sensitivity,
      scenarioOutcomes,
      computeMs: performance.now() - started,
    };
    const transfer: Transferable[] = [
      summary.years.buffer,
      summary.burdenQuantiles.buffer,
      summary.savingsQuantiles.buffer,
      summary.balanceQuantiles.buffer,
      summary.breakEvenYearCounts.buffer,
      ...scenarioOutcomes.map((outcome) => outcome.medianBalance.buffer),
    ];
    post({ type: 'result', requestId, payload }, transfer);
  } catch (error) {
    if (!isStale()) {
      post({
        type: 'error',
        requestId,
        message: error instanceof Error ? error.message : 'Simulation failed',
      });
    }
  }
}
