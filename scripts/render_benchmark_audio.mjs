/** Render bounded verifier events with the pinned Strudel Web Audio engine. */
import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import * as webAudio from 'node-web-audio-api';

const SAMPLE_RATE = 48000;
const RAW = 'https://raw.githubusercontent.com/';
const DOUGH = `${RAW}felixroos/dough-samples/9eacfc86ec4393e68a463ff52b01c19cfaa77f38/`;
const DRUMS = `${RAW}ritchse/tidal-drum-machines/6577395a4d05031728ced2ec3e5637fa89d8be48/machines/`;
const UZU = `${RAW}tidalcycles/uzu-drumkit/2f3e05c70ab4d73ad053a1467adec89bd27377a0/`;
const FONTS = `${RAW}felixroos/webaudiofontdata/23ca907d4370a04fd89ca483a92915e4d6159ab9/sound`;

function cacheAssets(directory) {
  const fetchRemote = globalThis.fetch;
  const pending = new Map();
  globalThis.fetch = async (input) => {
    const url = new URL(input);
    const key = url.href;
    if (!key.startsWith(RAW)) throw new Error(`Unexpected Strudel asset origin: ${url.origin}`);
    if (!pending.has(key)) pending.set(key, fetchAsset(key, directory, fetchRemote));
    return new Response(await pending.get(key));
  };
}

async function fetchAsset(url, directory, fetchRemote) {
  const filename = createHash('sha256').update(url).digest('hex');
  const path = join(directory, filename);
  try {
    return await readFile(path);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  const response = await fetchRemote(url, { redirect: 'error', signal: AbortSignal.timeout(20000) });
  if (!response.ok) throw new Error(`Strudel audio asset returned HTTP ${response.status}: ${url}`);
  const data = Buffer.from(await response.arrayBuffer());
  await writeFile(`${path}.${process.pid}.tmp`, data);
  await rename(`${path}.${process.pid}.tmp`, path);
  return data;
}

async function initWorklets(context, audio, polyphony) {
  const worklet = context.audioWorklet;
  const addModule = worklet.addModule.bind(worklet);
  let failure;
  worklet.addModule = async (url) => {
    // The Strudel bundle embeds its worklets as data URLs; the Node API accepts blobs.
    const prefix = 'data:text/javascript;base64,';
    if (!url.startsWith(prefix)) return addModule(url);
    const source = Buffer.from(url.slice(prefix.length), 'base64');
    const blob = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
    try {
      await addModule(blob);
    } catch (error) {
      failure = error;
    } finally {
      URL.revokeObjectURL(blob);
    }
  };
  context.resume = async () => {};
  globalThis.window = {};
  try {
    await audio.initAudio({ maxPolyphony: polyphony });
    if (failure) throw failure;
  } finally {
    delete globalThis.window;
    delete context.resume;
  }
}

async function registerAssets(audio, events) {
  audio.registerSynthSounds();
  const names = new Set(events.map(({ value }) => {
    const sound = value.s ?? 'triangle';
    return value.bank ? `${value.bank}_${sound}` : sound;
  }));
  const samples = async (url, base) => {
    const map = await fetch(url).then(response => response.json());
    await audio.samples(map, base);
  };
  if (names.has('piano')) await samples(`${DOUGH}piano.json`, `${DOUGH}piano/`);
  if ([...names].some(name => name.startsWith('gm_'))) {
    const { registerSoundfonts, setSoundfontUrl } = await import('@strudel/soundfonts/fontloader.mjs');
    setSoundfontUrl(FONTS);
    registerSoundfonts();
  }
  const missing = [...names].filter(name => !['-', '~', '_'].includes(name) && !audio.getSound(name));
  if (missing.some(name => name.includes('_'))) await samples(`${DOUGH}tidal-drum-machines.json`, DRUMS);
  if (missing.some(name => !name.includes('_'))) await samples(`${UZU}strudel.json`, UZU);
}

function encodeWav(audio) {
  const frames = audio.length;
  const size = frames * 4;
  const wav = Buffer.alloc(44 + size);
  wav.write('RIFF', 0);
  wav.writeUInt32LE(36 + size, 4);
  wav.write('WAVEfmt ', 8);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(2, 22);
  wav.writeUInt32LE(SAMPLE_RATE, 24);
  wav.writeUInt32LE(SAMPLE_RATE * 4, 28);
  wav.writeUInt16LE(4, 32);
  wav.writeUInt16LE(16, 34);
  wav.write('data', 36);
  wav.writeUInt32LE(size, 40);
  let peak = 0;
  for (let channel = 0; channel < 2; channel++) {
    const data = audio.getChannelData(channel);
    for (let frame = 0; frame < frames; frame++) {
      const value = data[frame];
      if (!Number.isFinite(value)) throw new Error('Strudel produced non-finite audio');
      peak = Math.max(peak, Math.abs(value));
      const pcm = Math.round(Math.max(-1, Math.min(1, value)) * 32767);
      wav.writeInt16LE(pcm, 44 + 4 * frame + 2 * channel);
    }
  }
  if (peak < 1 / 32767) throw new Error('Strudel rendered silence');
  return wav;
}

async function main() {
  const [output, cache] = process.argv.slice(2);
  const { cpm, seconds, events } = JSON.parse(await readFile('/dev/stdin', 'utf8'));
  const cps = cpm / 60;
  const onsets = events.filter(event => event.onset && event.begin >= 0 && event.begin < seconds * cps);
  onsets.sort((a, b) => a.begin - b.begin);
  if (!onsets.length) throw new Error('No verified onsets in the render window');
  await mkdir(cache, { recursive: true });
  cacheAssets(cache);
  Object.assign(globalThis, webAudio);
  const context = new webAudio.OfflineAudioContext(2, Math.round(seconds * SAMPLE_RATE), SAMPLE_RATE);
  globalThis.AudioContext = function () { return context; };
  // Superdough adds its reverb methods to AudioContext.prototype.
  globalThis.AudioContext.prototype = Object.getPrototypeOf(context);
  const preparations = [];
  globalThis.OfflineAudioContext = class extends webAudio.OfflineAudioContext {
    startRendering() {
      const complete = new Promise(resolve => this.addEventListener('complete', resolve, { once: true }));
      const ready = Promise.all([super.startRendering(), complete]);
      ready.catch(() => {});
      preparations.push(ready);
      return ready;
    }
  };
  const audio = await import('superdough');
  await initWorklets(context, audio, onsets.length);
  await registerAssets(audio, onsets);
  // The reverb generator renders its impulse response in a separate context.
  globalThis.window = {};
  for (const { value, begin, end } of onsets) {
    const sound = [value.bank, value.s ?? 'triangle'].filter(Boolean).join('_');
    try {
      await audio.superdough(value, begin / cps, (end - begin) / cps, cps, begin);
    } catch (error) {
      throw new Error(`At ${(begin / cps).toFixed(3)}s, ${sound} note ${value.note ?? '(none)'}: ${error.message}`);
    }
  }
  await Promise.all(preparations);
  await writeFile(output, encodeWav(await context.startRendering()));
}

try {
  await main();
} catch (error) {
  console.error(error.message);
  // The offline context owns worklet threads even when scheduling fails.
  process.exit(1);
}
