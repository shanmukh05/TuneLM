const blocked = /\b(?:import\s*\(|require\s*\(|process\b|globalThis\b|fetch\s*\(|WebSocket\b|XMLHttpRequest\b|child_process\b|node:|Deno\b|Bun\b)/;

function numeric(value) {
  if (value == null) return null;
  if (typeof value === 'number') return value;
  if (typeof value.valueOf === 'function') {
    const converted = Number(value.valueOf());
    if (Number.isFinite(converted)) return converted;
  }
  return Number(value) || null;
}

function jsonSafe(value, depth = 0) {
  if (depth > 4) return String(value);
  if (value == null || ['string', 'number', 'boolean'].includes(typeof value)) return value;
  if (Array.isArray(value)) return value.map((item) => jsonSafe(item, depth + 1));
  if (typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([key, item]) => !key.startsWith('_') && typeof item !== 'function')
        .map(([key, item]) => [key, jsonSafe(item, depth + 1)]),
    );
  }
  return String(value);
}

function eventToJson(hap) {
  const span = hap.whole || hap.part;
  return {
    begin: numeric(span?.begin),
    end: numeric(span?.end),
    onset: typeof hap.hasOnset === 'function' ? hap.hasOnset() : null,
    value: jsonSafe(hap.value),
  };
}

async function main() {
  let raw = '';
  for await (const chunk of process.stdin) raw += chunk;
  const output = process.stdout;
  const request = JSON.parse(raw);
  const code = String(request.code || '');
  const cycles = Math.min(Math.max(Number(request.cycles || 2), 0.25), 16);
  if (blocked.test(code)) throw new Error('Code contains a blocked capability');

  // Some audio-oriented dependencies log harmless browser warnings on import.
  // Keep stdout machine-readable for the Python bridge.
  console.log = (...args) => console.error(...args);
  const core = await import('@strudel/core');
  const mini = await import('@strudel/mini');
  const tonal = await import('@strudel/tonal');
  const { evaluate } = await import('@strudel/transpiler');
  let requestedCpm = null;
  const tempoScope = {
    setcpm: (value) => { requestedCpm = Number(value); return core.silence; },
    setCpm: (value) => { requestedCpm = Number(value); return core.silence; },
    setcps: (value) => { requestedCpm = Number(value) * 60; return core.silence; },
    setCps: (value) => { requestedCpm = Number(value) * 60; return core.silence; },
  };
  await core.evalScope(core, mini, tonal, tempoScope);
  // Evaluated programs only need the Strudel scope. Remove ambient I/O handles.
  globalThis.fetch = undefined;
  globalThis.WebSocket = undefined;
  const result = await evaluate(code);
  const pattern = result?.pattern;
  if (!pattern || typeof pattern.queryArc !== 'function') {
    throw new Error('The final expression did not produce a Strudel Pattern');
  }
  const events = pattern.queryArc(0, cycles).slice(0, Number(request.max_events || 2048));
  output.write(JSON.stringify({
    valid: true,
    events: events.map(eventToJson),
    runtime: { cpm: requestedCpm },
  }));
}

main().catch((error) => {
  process.stdout.write(JSON.stringify({ valid: false, error: String(error?.stack || error) }));
  process.exitCode = 1;
});
