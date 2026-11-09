/** Execute one accepted Strudel program and return bounded symbolic events.
 * Python enforces a wall deadline and launches this process without credentials.
 */
import { normalize, parseCode, methods, decompose, transpose, arrangementCycles } from './ast.mjs';
const proc = process;
const output = proc.stdout;
console.log = (...v) => console.error(...v);
const VERSION = 'strudel-1.2.6/tunelm-2';
let stage = 'request';

function finite(value, depth = 0) {
  if (depth > 16) throw new Error('Control nesting exceeds limit');
  if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('Non-finite control');
  if (value === null || value === undefined) throw new Error('Null control');
  if (typeof value !== 'object') return;
  for (const entry of Object.values(value)) finite(entry, depth + 1);
}

function event(hap) {
  const whole = hap.whole ?? hap.part;
  const result = { begin: Number(whole.begin), end: Number(whole.end),
    part_begin: Number(hap.part.begin), part_end: Number(hap.part.end),
    onset: hap.hasOnset(), value: hap.value };
  finite(result);
  return result;
}

function validateRequest(request) {
  if (request.schema_version !== 2 || typeof request.code !== 'string' || !request.code.trim()) {
    throw new Error('Expected schema_version=2 and nonempty code');
  }
  const limits = [[request.cycles ?? 16, 64], [request.max_events ?? 20000, 50000]];
  for (const [value, ceiling] of limits) {
    if (!Number.isInteger(value) || value < 1 || value > ceiling) throw new Error('Invalid evaluation limit');
  }
}

async function prepareEngine() {
  const core = await import('@strudel/core');
  const mini = await import('@strudel/mini');
  const tonal = await import('@strudel/tonal');
  const { evaluate } = await import('@strudel/transpiler');
  let cpm = null;
  const scope = {
    setcpm: (v) => { cpm = Number(v); return core.silence; },
    setcps: (v) => { cpm = Number(v) * 60; return core.silence; },
  };
  scope.setCpm = scope.setcpm; scope.setCps = scope.setcps;
  for (const name of ['pianoroll','scope','spiral','punchcard','_pianoroll','_scope']) {
    core.Pattern.prototype[name] = function() { return this; };
  }
  await core.evalScope(core, mini, tonal, scope);
  globalThis.fetch = undefined;
  globalThis.WebSocket = undefined;
  return { evaluate, tempo: () => cpm };
}

function validatePattern(pattern, cpm) {
  if (!pattern || typeof pattern.queryArc !== 'function') throw new Error('Final expression is not a Pattern');
  if (cpm !== null && (!Number.isFinite(cpm) || cpm <= 0 || cpm > 1000)) throw new Error('Invalid tempo');
}

function queryEvents(pattern, cycles, limit) {
  const events = [];
  const seeded = pattern.seed(0);
  for (let cycle = 0; cycle < cycles; cycle++) {
    const haps = seeded.queryArc(cycle, cycle + 1);
    if (events.length + haps.length > limit) throw new Error('Event limit exceeded');
    for (const hap of haps) events.push(event(hap));
  }
  if (!events.some((e) => e.onset && (e.value?.gain ?? 1) > 0)) throw new Error('Pattern has no audible onsets');
  return events;
}

async function execute(request) {
  validateRequest(request);
  const { code } = request;
  stage = 'syntax';
  if (request.operation === 'decompose') return { valid: true, ...decompose(code) };
  if (request.operation === 'transpose') return { valid: true, ...transpose(code, request.key) };
  if (request.operation !== 'verify') throw new Error('Unknown operation');
  const normalized = normalize(code);
  const cycles = Math.max(request.cycles ?? 16, arrangementCycles(normalized));
  if (cycles > 64) throw new Error('Arrangement exceeds the 64-cycle verification limit');
  stage = 'runtime_setup';
  const engine = await prepareEngine();
  stage = 'evaluate';
  const result = await engine.evaluate(normalized);
  validatePattern(result.pattern, engine.tempo());
  stage = 'query_arc';
  const events = queryEvents(result.pattern, cycles, request.max_events ?? 20000);
  return { valid: true, normalized_code: normalized, events, cpm: engine.tempo(),
    methods: methods(parseCode(normalized)), cycles };
}

try {
  let raw = '';
  for await (const chunk of proc.stdin) {
    raw += chunk;
    if (Buffer.byteLength(raw) > 131072) throw new Error('Request exceeds 128 KiB');
  }
  const request = JSON.parse(raw);
  const result = await execute(request);
  output.write(JSON.stringify({ schema_version: 2, request_id: request.request_id,
    runtime_version: VERSION, stage: 'passed', ...result }) + '\n');
} catch (error) {
  output.write(JSON.stringify({ schema_version: 2, valid: false, stage,
    error_code: stage.toUpperCase(), error: String(error.message), runtime_version: VERSION }) + '\n');
  proc.exitCode = 1;
}
