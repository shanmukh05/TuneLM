import * as core from '@strudel/core';
import { m } from '@strudel/mini';
import * as tonal from '@strudel/tonal';
import { arrangementCycles, normalize, parseCode } from '../verifier/ast.mjs';

const operations = new Set(`
  stack arrange cat slowcat fastcat seq sequence stepcat timeCat polyrhythm silence
  note n s sound chord scale scaleTranspose scaleTrans strans transpose trans voicing
  sine sine2 cosine cosine2 saw saw2 isaw isaw2 tri tri2 square square2 rand perlin
  slow fast rev palindrome early late iter iterBack every firstOf lastOf inside outside
  add sub mul div mod range range2 segment seg degrade degradeBy sometimes sometimesBy
  often rarely almostNever almostAlways never always off jux juxBy echo echoWith
  euclid euclidRot ply struct mask clip legato compress compressSpan zoom linger
  layer superimpose chunk chunkBack fastChunk rib ribbon striate chop slice splice
  loopAt loopat swing swingBy press pressBy brak brand irand choose chooseCycles
  rootNotes voicings pick pickmod pickF pickOut pickS fmap
`.trim().split(/\s+/));
const excludedControls = new Set(['source', 'ir', 'i', 'channels', 'analyze', 'fft', 'orbit', 'duckorbit']);
const tempoNames = new Set(['setcpm', 'setcps', 'setCpm', 'setCps']);
const rests = new Set(['-', '_', '~']);
const soundName = /^[a-z0-9_-]{1,100}$/i;
const drums = /^(?:bd|sd|rim|cp|hh|oh|cr|rd|ht|mt|lt|sh|cb|tb|brk|perc|misc|fx)$/;
const slice = (code, node) => code.slice(node.start, node.end);
const isPattern = value => value instanceof core.Pattern;
const callable = name => operations.has(name) ||
  (core.isControlName(name) && !excludedControls.has(name));

function literal(value, mini = false) {
  if (typeof value === 'string') return mini ? m(value, 0) : value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  throw new Error('Use text or finite numbers in Strudel expressions.');
}

function arithmetic(operator, left, right) {
  if (!Number.isFinite(left) || !Number.isFinite(right)) {
    throw new Error('Arithmetic is supported for numbers. Use Strudel methods for patterns.');
  }
  switch (operator) {
    case '+': return literal(left + right);
    case '-': return literal(left - right);
    case '*': return literal(left * right);
    case '/': return literal(left / right);
    case '%': return literal(left % right);
    default: throw new Error(`Unsupported operator: ${operator}.`);
  }
}

function identify(name, scope) {
  if (scope.has(name)) return scope.get(name);
  if (callable(name)) return tonal[name] ?? core[name];
  throw new Error(`Unsupported Strudel name: ${name}.`);
}

function closure(node, scope) {
  if (node.params.some(param => param.type !== 'Identifier') ||
      (node.body.type === 'BlockStatement' && (node.body.body.length !== 1 ||
        node.body.body[0].type !== 'ReturnStatement'))) {
    throw new Error('Use a single expression in pattern callbacks.');
  }
  const body = node.body.type === 'BlockStatement' ? node.body.body[0].argument : node.body;
  return (...args) => {
    const local = new Map(scope);
    node.params.forEach((param, index) => local.set(param.name, args[index]));
    return evaluate(body, local);
  };
}

function call(node, scope) {
  const args = node.arguments.map(arg => evaluate(arg, scope));
  const callee = node.callee;
  if (callee.type === 'Identifier') {
    const fn = identify(callee.name, scope);
    if (typeof fn === 'function') return fn(...args);
  }
  if (callee.type === 'MemberExpression' && !callee.computed) {
    const name = callee.property.name;
    if (callable(name)) {
      const pattern = evaluate(callee.object, scope);
      if (isPattern(pattern) && typeof pattern[name] === 'function') return pattern[name](...args);
    }
    throw new Error(`Unsupported Strudel method: ${name}`);
  }
  throw new Error('Unsupported Strudel function or method.');
}

function evaluate(node, scope) {
  switch (node?.type) {
    case 'Literal': return literal(node.value, node.raw.startsWith('"'));
    case 'Identifier': return identify(node.name, scope);
    case 'TemplateLiteral':
      if (node.expressions.length === 0) return literal(node.quasis[0].value.raw, true);
      break;
    case 'ArrayExpression': return node.elements.map(item => evaluate(item, scope));
    case 'CallExpression': return call(node, scope);
    case 'ArrowFunctionExpression': return closure(node, scope);
    case 'BinaryExpression':
      return arithmetic(node.operator, evaluate(node.left, scope), evaluate(node.right, scope));
    case 'UnaryExpression':
      if (['+', '-'].includes(node.operator)) return arithmetic(node.operator, 0, evaluate(node.argument, scope));
      break;
  }
  throw new Error(`Unsupported Strudel syntax: ${node?.type ?? 'empty expression'}.`);
}

function eventValue(hap) {
  if (!hap.value || typeof hap.value !== 'object') {
    throw new Error('A playable pattern needs a sound or a note.');
  }
  const entries = Object.entries(hap.value).filter(([key, value]) =>
    !excludedControls.has(key) &&
    ((typeof value === 'number' && Number.isFinite(value)) ||
      (typeof value === 'string' && value.length <= 120)));
  const value = Object.fromEntries(entries);
  if ((value.s !== undefined && !rests.has(value.s) && !soundName.test(value.s)) ||
      (value.bank !== undefined && !soundName.test(value.bank))) {
    throw new Error('This composition references an unsupported sound name.');
  }
  return value;
}

function collect(pattern, cycles) {
  if (!isPattern(pattern)) throw new Error('The last expression must produce a Strudel pattern.');
  const events = [];
  for (let i = 0; i < cycles; i++) {
    const haps = pattern.queryArc(i, i + 1);
    if (haps.length > 1500 || events.length + haps.length > 24000) {
      throw new Error('This composition is too dense for the browser player.');
    }
    for (const hap of haps) {
      if (!hap.hasOnset()) continue;
      const value = eventValue(hap);
      if (rests.has(value.s) || Number(value.gain ?? 1) <= 0) continue;
      const start = Number(hap.whole.begin);
      const duration = Number(hap.whole.end) - start;
      if (Number.isFinite(start) && Number.isFinite(duration) && duration > 0) {
        events.push({ start, duration: Math.min(duration, cycles - start), value });
      }
    }
  }
  return events.sort((a, b) => a.start - b.start);
}

function title(name) {
  const text = name.replace(/^gm_/, '').replace(/([a-z])([A-Z])/g, '$1 $2').replace(/_/g, ' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function byInstrument(events) {
  const groups = new Map();
  for (const event of events) {
    const { bank, s = 'triangle' } = event.value;
    const sound = s.split('_').at(-1);
    const name = drums.test(sound)
      ? `${bank ?? (s === sound ? '' : s.slice(0, -sound.length - 1))} drums`.trim() : s;
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(event);
  }
  return Array.from(groups, ([name, events]) => ({ name: title(name), events }));
}

function resolve(node, declarations) {
  const seen = new Set();
  while (node.type === 'Identifier' && declarations.has(node.name)) {
    if (seen.has(node.name)) throw new Error('Circular voice declaration.');
    seen.add(node.name);
    node = declarations.get(node.name);
  }
  return node;
}

export function compile(code) {
  if (code.length > 24000) throw new Error('This composition exceeds the browser player’s code limit.');
  const normalized = normalize(code);
  const ast = parseCode(normalized);
  const scope = new Map();
  const declarations = new Map();
  let cpm = 30;
  for (const statement of ast.body.slice(0, -1)) {
    if (statement.type === 'VariableDeclaration') {
      for (const { id, init } of statement.declarations) {
        if (id.type !== 'Identifier') throw new Error('Use a name for each musical part.');
        scope.set(id.name, evaluate(init, scope));
        declarations.set(id.name, init);
      }
    } else {
      const { callee, arguments: args } = statement.expression;
      if (!tempoNames.has(callee.name)) throw new Error('Only tempo may be set outside a musical part.');
      cpm = evaluate(args[0], scope) * (callee.name.toLowerCase() === 'setcps' ? 60 : 1);
    }
  }
  if (!Number.isFinite(cpm) || cpm < 4 || cpm > 125) throw new Error('Tempo is outside the player’s range.');
  const expression = ast.body.at(-1).expression;
  const final = resolve(expression, declarations);
  const arrangement = arrangementCycles(normalized);
  if (arrangement > 64) throw new Error('The browser player supports arrangements up to 64 cycles.');
  const cycles = arrangement || 16;
  const events = collect(evaluate(expression, scope), cycles);
  if (!events.length) throw new Error('No audible notes were found.');
  const isStack = final.type === 'CallExpression' && final.callee.name === 'stack';
  const tracks = isStack && final.arguments.length > 1 ? final.arguments.map((node, index) => {
    const events = collect(evaluate(node, scope), cycles);
    const instruments = byInstrument(events);
    return {
      name: node.type === 'Identifier' ? title(node.name) :
        (instruments.length === 1 ? instruments[0].name : `Layer ${index + 1}`),
      code: slice(normalized, resolve(node, declarations)),
      events,
    };
  }).filter(track => track.events.length) : byInstrument(events);
  return { cpm, cycles, arrangement: Boolean(arrangement), events, tracks };
}
