/** Own AST validation, REPL normalization, and conservative stem extraction.
 * This limits the accepted DSL; process limits remain necessary for execution.
 */
import { parse } from 'acorn';
import { simple, full } from 'acorn-walk';

const forbidden = new Set(['process', 'globalThis', 'global', 'window', 'document',
  'fetch', 'require', 'eval', 'Function', 'WebSocket', 'XMLHttpRequest', 'samples',
  'import', 'constructor', '__proto__', 'prototype', 'setTimeout', 'setInterval',
  'Reflect', 'Proxy', 'console', 'Buffer', 'Deno', 'Bun']);
for (const name of ['Date', 'performance', 'crypto', 'random']) forbidden.add(name);
const allowed = new Set(['Program', 'ExpressionStatement', 'VariableDeclaration',
  'VariableDeclarator', 'Identifier', 'Literal', 'CallExpression', 'MemberExpression',
  'ArrayExpression', 'ObjectExpression', 'Property', 'ArrowFunctionExpression',
  'BinaryExpression', 'UnaryExpression', 'ConditionalExpression', 'LogicalExpression',
  'BlockStatement', 'ReturnStatement', 'LabeledStatement', 'TemplateLiteral',
  'TemplateElement', 'SequenceExpression']);
const tempoNames = new Set(['setcpm', 'setcps', 'setCpm', 'setCps']);
const source = (code, node) => code.slice(node.start, node.end);
const isTempo = (node) => node.type === 'ExpressionStatement' &&
  node.expression.type === 'CallExpression' && tempoNames.has(node.expression.callee.name);

function blockedName(name) {
  if (forbidden.has(name)) throw new Error(`Blocked capability: ${name}`);
}

function validateMember(node) {
  if (!node.computed) return blockedName(node.property.name);
  const integerIndex = node.property.type === 'Literal' && Number.isInteger(node.property.value);
  if (!integerIndex) throw new Error('Computed access is limited to integer indices');
}

const validators = {
  Identifier: (node) => blockedName(node.name),
  MemberExpression: validateMember,
  Property: (node) => blockedName(String(node.key.value ?? node.key.name)),
  LabeledStatement: (node) => {
    if (node.label.name !== '$') throw new Error('Only $: track labels are supported');
  },
  VariableDeclaration: (node) => {
    if (node.kind !== 'const') throw new Error('Use immutable const declarations');
  },
};

export function parseCode(code) {
  const ast = parse(code, { ecmaVersion: 2022 });
  full(ast, (node) => {
    if (!allowed.has(node.type)) throw new Error(`Unsupported syntax: ${node.type}`);
    validators[node.type]?.(node);
  });
  return ast;
}

function trackStatements(code, ast) {
  const setup = [];
  const tracks = [];
  for (const node of ast.body) {
    if (node.type === 'VariableDeclaration' || isTempo(node)) {
      if (tracks.length) throw new Error('Setup must precede all $: tracks');
      setup.push(source(code, node));
      continue;
    }
    if (node.type !== 'LabeledStatement' || node.body.type !== 'ExpressionStatement') {
      throw new Error('Do not mix $: tracks and unlabelled final patterns');
    }
    tracks.push(source(code, node.body.expression));
  }
  return `${setup.join('\n')}\nstack(${tracks.join(',\n')})`;
}

export function normalize(code) {
  const ast = parseCode(code);
  if (ast.body.some((n) => n.type === 'LabeledStatement')) return trackStatements(code, ast);
  if (ast.body.at(-1)?.type !== 'ExpressionStatement') {
    throw new Error('End the program with a musical pattern expression');
  }
  for (const node of ast.body.slice(0, -1)) {
    if (node.type !== 'VariableDeclaration' && !isTempo(node)) {
      throw new Error('Only const and tempo setup may precede the final pattern');
    }
  }
  return code;
}

function resolveVoice(node, declarations, seen = new Set()) {
  if (node.type !== 'Identifier' || !declarations.has(node.name)) return node;
  if (seen.has(node.name)) throw new Error('Circular voice declaration');
  seen.add(node.name);
  return resolveVoice(declarations.get(node.name), declarations, seen);
}

function requiredSetup(node, setup, declarations, code) {
  const needed = new Set();
  const groups = new Map(setup.filter((s) => s.type === 'VariableDeclaration')
    .flatMap((s) => s.declarations.map((d) => [d.id.name, s])));
  function visit(expression) {
    full(expression, (entry) => {
      if (entry.type !== 'Identifier' || !declarations.has(entry.name) || needed.has(entry.name)) return;
      needed.add(entry.name);
      visit(groups.get(entry.name));
    });
  }
  visit(node);
  return setup.filter((statement) => statement.type === 'VariableDeclaration' &&
    statement.declarations.some((d) => needed.has(d.id.name)))
    .map((statement) => source(code, statement)).join('\n');
}

export function decompose(code) {
  const normalized = normalize(code);
  const ast = parseCode(normalized);
  const setup = ast.body.slice(0, -1);
  const declarations = new Map();
  for (const n of setup.filter((n) => n.type === 'VariableDeclaration')) {
    for (const d of n.declarations) declarations.set(d.id.name, d.init);
  }
  const final = resolveVoice(ast.body.at(-1).expression, declarations);
  const directStack = final.type === 'CallExpression' && final.callee.name === 'stack';
  const expressions = directStack ? final.arguments : [final];
  const scales = new Set();
  simple(ast, { CallExpression(node) {
    if ((node.callee.property?.name ?? node.callee.name) !== 'scale') return;
    let arg = node.arguments[0];
    if (arg?.type === 'Identifier') arg = declarations.get(arg.name);
    if (arg?.type !== 'Literal' || typeof arg.value !== 'string') return;
    const match = arg.value.match(/^([A-Ga-g][#b]?)(?:-?\d+)?:([a-z]+)$/);
    if (match) scales.add(`${match[1][0].toUpperCase()}${match[1].slice(1)}:${match[2]}`);
  } });
  return {
    normalized_code: normalized,
    decomposable: directStack || !usesMethod(normalized, 'arrange'),
    setup: setup.map((n) => source(normalized, n)).join('\n'),
    stems: expressions.map((n) => {
      const expression = resolveVoice(n, declarations);
      return { code: source(normalized, expression),
        setup: requiredSetup(expression, setup, declarations, normalized) };
    }),
    methods: methods(ast),
    keys: [...scales].sort(),
  };
}

export function methods(ast) {
  const names = new Set();
  simple(ast, { CallExpression(node) {
    const callee = node.callee;
    if (callee.type === 'Identifier') names.add(callee.name);
    if (callee.type === 'MemberExpression' && !callee.computed) names.add(callee.property.name);
  } });
  return [...names].sort();
}

export function arrangementCycles(code) {
  const ast = parseCode(code);
  const declarations = new Map();
  for (const statement of ast.body.filter((s) => s.type === 'VariableDeclaration')) {
    for (const item of statement.declarations) declarations.set(item.id.name, item.init);
  }
  const span = arrangementSpan(ast.body.at(-1).expression, declarations);
  if (!span && methods(ast).includes('arrange')) {
    throw new Error('Cannot determine arrangement duration; use direct const/stack/arrange expressions');
  }
  return Math.ceil(span);
}

function arrangementSpan(expression, declarations) {
  const node = resolveVoice(expression, declarations);
  if (node.type !== 'CallExpression') return 0;
  if (node.callee.name === 'arrange') {
    return node.arguments.reduce((sum, arg) => {
      const duration = arg.type === 'ArrayExpression' ? arg.elements[0] : null;
      if (duration?.type !== 'Literal' || typeof duration.value !== 'number' ||
          !Number.isFinite(duration.value) || duration.value <= 0) {
        throw new Error('arrange durations must be positive numeric literals');
      }
      return sum + duration.value;
    }, 0);
  }
  if (node.callee.name === 'stack') {
    return Math.max(0, ...node.arguments.map((arg) => arrangementSpan(arg, declarations)));
  }
  if (node.callee.type !== 'MemberExpression') return 0;
  const span = arrangementSpan(node.callee.object, declarations);
  return scaledSpan(node, span);
}

function scaledSpan(node, span) {
  const method = node.callee.property.name;
  if (!span || !['slow','fast'].includes(method)) return span;
  const factor = node.arguments[0];
  if (factor?.type !== 'Literal' || typeof factor.value !== 'number' || factor.value <= 0) {
    throw new Error('Use literal time factors around an arrangement');
  }
  return method === 'slow' ? span * factor.value : span / factor.value;
}

function usesMethod(code, name) { return methods(parseCode(code)).includes(name); }

export function transpose(code, key) {
  if (!/^[A-G](?:#|b)?:(?:major|minor|dorian|mixolydian|phrygian|lydian|locrian)$/.test(key)) {
    throw new Error('Unsupported target key');
  }
  const ast = parseCode(code);
  const edits = [];
  simple(ast, { CallExpression(node) {
    const name = node.callee.property?.name ?? node.callee.name;
    if (name === 'note' || name === 'chord') throw new Error('Absolute pitch transposition unsupported');
    if (name !== 'scale') return;
    const arg = node.arguments[0];
    if (arg?.type !== 'Literal' || typeof arg.value !== 'string') {
      throw new Error('Scale must be a literal for transposition');
    }
    const match = arg.value.match(/^[A-Ga-g](?:#|b)?(-?\d+)?:[a-z]+$/);
    if (!match) throw new Error('Unsupported scale expression');
    const [root, mode] = key.split(':');
    edits.push({start:arg.start, end:arg.end, value:JSON.stringify(`${root}${match[1] ?? '3'}:${mode}`)});
  } });
  let output = code;
  for (const e of edits.sort((a,b) => b.start-a.start)) {
    output = output.slice(0,e.start)+e.value+output.slice(e.end);
  }
  const resultAst = parseCode(output);
  const final = resultAst.body.at(-1);
  if (final?.type !== 'ExpressionStatement') throw new Error('Missing final expression');
  return { code: output, transposed: edits.length > 0,
    expression: source(output, final.expression),
    setup: resultAst.body.slice(0,-1).map((n) => source(output,n)).join('\n') };
}
