/** Read literal examples or independent Strudel programs from source files.
 * Acorn decodes literals without importing or executing downloaded source code.
 * The Python collector owns downloads, licensing, provenance, and ingestion.
 */
import { readFileSync } from 'node:fs';
import { parse, tokenizer } from 'acorn';

const fields = new Set(['id', 'name', 'description', 'category', 'engine', 'code', 'tags']);

function literal(tokens, index) {
  const token = tokens[index];
  if (token.type.label === 'string') return token.value;
  if (token.type.label === '`' && tokens[index + 2]?.type.label === '`') {
    return tokens[index + 1].value;
  }
  if (token.type.label === '[') {
    const values = [];
    for (let cursor = index + 1; tokens[cursor]?.type.label !== ']'; cursor++) {
      if (tokens[cursor]?.type.label === ',') continue;
      if (tokens[cursor]?.type.label !== 'string') return undefined;
      values.push(tokens[cursor].value);
    }
    return values;
  }
  return undefined;
}

function setField(record, tokens, index) {
  const name = tokens[index].value;
  if (!record || !fields.has(name) || tokens[index + 1]?.type.label !== ':') return;
  record[name] = literal(tokens, index + 2);
}

function extract(source) {
  const tokens = [...tokenizer(source, { ecmaVersion: 2022, locations: true })];
  const stack = [], records = [];
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index];
    if (token.type.label === '{' || token.type.label === '${') {
      stack.push({ line: token.loc.start.line });
      continue;
    }
    if (token.type.label === '}') {
      const row = stack.pop();
      if (row?.engine === 'strudel' && typeof row.code === 'string') records.push(row);
      continue;
    }
    setField(stack.at(-1), tokens, index);
  }
  if (!records.length) throw new Error('No static Strudel example records found');
  return records;
}

const roots = new Set(['s', 'sound', 'n', 'note', 'stack', 'cat', 'slowcat', 'fastcat', 'arrange']);
const tempos = new Set(['setcpm', 'setcps', 'setCpm', 'setCps']);

function patternExpression(node) {
  if (node.type !== 'CallExpression') return false;
  if (node.callee.type === 'Identifier') return roots.has(node.callee.name);
  if (node.callee.type === 'MemberExpression') return patternExpression(node.callee.object);
  return false;
}

function setupStatement(node) {
  if (node.type === 'VariableDeclaration') return true;
  if (node.type !== 'ExpressionStatement') return false;
  return node.expression.type === 'CallExpression' && tempos.has(node.expression.callee.name);
}

function tracks(source, body) {
  const statements = [];
  let active = 0;
  for (const node of body) {
    if (setupStatement(node)) {
      statements.push(source.slice(node.start, node.end));
      continue;
    }
    if (node.type !== 'LabeledStatement' || node.body.type !== 'ExpressionStatement') return null;
    if (node.label.name.startsWith('_')) continue;
    statements.push('$: ' + source.slice(node.body.expression.start, node.body.expression.end));
    active++;
  }
  if (!active) return null;
  return [{ code: statements.join('\n'), line: 1,
    normalization: 'named tracks canonicalized to $:; underscore-muted tracks omitted' }];
}

function programs(source) {
  const original = [{ code: source, line: 1 }];
  let ast;
  try {
    ast = parse(source, { ecmaVersion: 2022, locations: true, sourceType: 'module' });
  } catch {
    return original;
  }
  if (ast.body.some((node) => node.type === 'LabeledStatement')) return tracks(source, ast.body) ?? original;
  const setup = [], records = [];
  for (const node of ast.body) {
    const code = source.slice(node.start, node.end);
    if (setupStatement(node)) {
      setup.push(code);
      continue;
    }
    if (node.type !== 'ExpressionStatement' || !patternExpression(node.expression)) return original;
    records.push({ code: [...setup, code].join('\n'), line: node.loc.start.line });
  }
  return records.length ? records : original;
}

function templates(source) {
  const ast = parse(source, { ecmaVersion: 2022, sourceType: 'module', locations: true });
  const records = [];
  for (const entry of ast.body) {
    if (entry.type !== 'ExportNamedDeclaration' || entry.declaration?.type !== 'VariableDeclaration') continue;
    for (const declaration of entry.declaration.declarations) {
      const value = declaration.init;
      if (value?.type !== 'TemplateLiteral' || value.expressions.length) continue;
      records.push({ id: declaration.id.name, code: value.quasis[0].value.cooked,
        description: declaration.id.name + ' drum pattern', line: entry.loc.start.line,
        tags: ['drums', 'licensed-rhythm'] });
    }
  }
  if (!records.length) throw new Error('No static exported program templates found');
  return records;
}

try {
  const source = readFileSync(process.argv[2], 'utf8');
  const extractors = { programs, module_objects: extract, module_templates: templates };
  const extractSource = extractors[process.argv[3] ?? 'module_objects'];
  if (!extractSource) throw new Error('Unknown extraction format');
  const records = extractSource(source);
  process.stdout.write(JSON.stringify(records) + '\n');
} catch (error) {
  console.error(String(error.message));
  process.exitCode = 1;
}
