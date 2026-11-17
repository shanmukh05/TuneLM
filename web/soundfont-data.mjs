import { parse } from 'acorn';

const decodedZones = new WeakMap();

function read(node) {
  switch (node.type) {
    case 'Literal':
      if (['number', 'string', 'boolean'].includes(typeof node.value) || node.value === null) {
        return node.value;
      }
      break;
    case 'UnaryExpression':
      if (node.operator === '-' && typeof node.argument.value === 'number') return -node.argument.value;
      break;
    case 'ArrayExpression': return node.elements.map(read);
    case 'ObjectExpression':
      return Object.fromEntries(node.properties.map(prop => {
        if (prop.type !== 'Property' || prop.kind !== 'init' || prop.computed || prop.shorthand) {
          throw new Error('Unexpected soundfont property.');
        }
        return [prop.key.name ?? prop.key.value, read(prop.value)];
      }));
  }
  throw new Error('Unexpected value in Strudel soundfont.');
}

export function readSoundfont(source) {
  const ast = parse(source, { ecmaVersion: 2022 });
  const tone = ast.body.filter(node => node.type === 'VariableDeclaration')
    .flatMap(node => node.declarations).find(node => node.id.name?.startsWith('_tone_'));
  if (tone?.init?.type !== 'ObjectExpression') throw new Error('Invalid Strudel soundfont.');
  const { zones } = read(tone.init);
  if (!Array.isArray(zones) || !zones.length) throw new Error('The Strudel soundfont contains no samples.');
  return zones;
}

export function nearestZone(zones, pitch) {
  if (!Number.isFinite(pitch)) return undefined;
  let best;
  let distance = Infinity;
  for (const zone of zones) {
    const gap = Math.max(zone.keyRangeLow - pitch, pitch - zone.keyRangeHigh - 1, 0);
    if (gap === 0) return zone;
    if (gap < distance) {
      best = zone;
      distance = gap;
    }
  }
  return best;
}

async function decodeZone(zone, context) {
  if (zone.file) {
    const bytes = Uint8Array.from(atob(zone.file), character => character.charCodeAt(0));
    return context.decodeAudioData(bytes.buffer);
  }
  if (!zone.sample) throw new Error('The soundfont zone contains no audio.');

  const bytes = Uint8Array.from(atob(zone.sample), character => character.charCodeAt(0));
  const pcm = new DataView(bytes.buffer);
  const buffer = context.createBuffer(1, Math.floor(bytes.length / 2), zone.sampleRate);
  const samples = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i++) samples[i] = pcm.getInt16(2 * i, true) / 32768;
  return buffer;
}

export function decodeSoundfontZone(zone, context) {
  if (!decodedZones.has(zone)) {
    decodedZones.set(zone, decodeZone(zone, context).catch(error => {
      decodedZones.delete(zone);
      throw error;
    }));
  }
  return decodedZones.get(zone);
}
