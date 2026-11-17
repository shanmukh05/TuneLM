import * as audio from 'superdough';
import { finishReverbs } from 'superdough/reverbGen.mjs';
import { getFontBufferSource, registerSoundfonts, setSoundfontUrl } from '@strudel/soundfonts/fontloader.mjs';
import gm from '@strudel/soundfonts/gm.mjs';
import { fetchAudioAsset } from './audio-assets.mjs';

const raw = 'https://raw.githubusercontent.com/';
const dough = `${raw}felixroos/dough-samples/9eacfc86ec4393e68a463ff52b01c19cfaa77f38/`;
const drums = `${raw}ritchse/tidal-drum-machines/6577395a4d05031728ced2ec3e5637fa89d8be48/machines/`;
const uzu = `${raw}tidalcycles/uzu-drumkit/2f3e05c70ab4d73ad053a1467adec89bd27377a0/`;
const fonts = `${raw}felixroos/webaudiofontdata/23ca907d4370a04fd89ca483a92915e4d6159ab9/sound`;
const maps = new Map();
const rendered = new Map();
const pending = new WeakMap();
const cacheBudget = 96 * 1024 * 1024;
const renderJobs = [];
let cachedBytes = 0;
let rendering = false;
let latestMix;
let registered = false;
let outputContext;
let active;

export class UnplayablePatternError extends Error {}

function assetError(label, error) {
  return new UnplayablePatternError(
    `Could not load ${label}. ${error.message} Use built-in sounds such as sine, triangle, or sbd if samples are unavailable.`,
    { cause: error },
  );
}

function loadMap(url, base, label) {
  if (!maps.has(url)) {
    maps.set(url, fetchAudioAsset(url).then(async response => {
      await audio.samples(await response.json(), base);
    }).catch(error => {
      maps.delete(url);
      throw assetError(label, error);
    }));
  }
  return maps.get(url);
}

function soundName({ bank, s = 'triangle' }) {
  return bank ? `${bank}_${s}` : s;
}

async function loadSounds(events, context) {
  const names = new Set(events.map(event => soundName(event.value)));
  const missing = [...names].filter(name => !audio.getSound(name));
  await Promise.all([
    missing.includes('piano') && loadMap(`${dough}piano.json`, `${dough}piano/`, 'the Strudel piano library'),
    missing.some(name => name !== 'piano' && !name.includes('_')) && loadMap(`${uzu}strudel.json`, uzu, 'the Strudel sample library'),
    missing.some(name => name.includes('_')) && loadMap(`${dough}tidal-drum-machines.json`, drums, 'the Strudel drum machine library'),
  ]);

  const warm = new Map();
  for (const { value } of events) {
    const name = soundName(value);
    const sound = audio.getSound(name);
    if (!sound) throw new UnplayablePatternError(`The Strudel sound “${name}” is unavailable.`);
    if (Object.hasOwn(gm, name)) {
      const bank = gm[name];
      const index = ((Math.round(Number(value.n) || 0) % bank.length) + bank.length) % bank.length;
      const preset = bank[index];
      const key = `${preset}:${value.freq ?? ''}:${value.note ?? ''}`;
      if (!warm.has(key)) warm.set(key, { name, load: () => getFontBufferSource(preset, value, context) });
    } else if (sound.data?.samples) {
      const controls = { ...value, s: name };
      const key = audio.getSampleInfo(controls, sound.data.samples).url;
      if (!warm.has(key)) warm.set(key, { name, load: () => audio.getSampleBuffer(controls, sound.data.samples) });
    }
  }
  const jobs = [...warm.values()];
  for (let i = 0; i < jobs.length; i += 12) {
    await Promise.all(jobs.slice(i, i + 12).map(async ({ name, load }) => {
      try {
        await load();
      } catch (error) {
        throw assetError(`the Strudel sound “${name}”`, error);
      }
    }));
  }
}

async function loadWorklet(context) {
  let timeout;
  try {
    await Promise.race([
      context.audioWorklet.addModule(new URL('./audio-worklets.js', import.meta.url)),
      new Promise((_, reject) => {
        timeout = setTimeout(() => reject(new Error('The audio worklet took too long to load.')), 15000);
      }),
    ]);
  } catch (error) {
    throw new Error('The browser audio engine could not load. Reload the page and try again.', { cause: error });
  } finally {
    clearTimeout(timeout);
  }
}

export const maxExportSeconds = 240;

function leaveHeadroom(buffer) {
  const channels = [buffer.getChannelData(0), buffer.getChannelData(1)];
  let peak = 0;
  for (const data of channels) {
    for (const sample of data) peak = Math.max(peak, Math.abs(sample));
  }
  if (!Number.isFinite(peak)) throw new UnplayablePatternError('The pattern produced invalid audio samples.');
  if (peak > 0.98) {
    const scale = 0.98 / peak;
    for (const data of channels) {
      for (let i = 0; i < data.length; i++) data[i] *= scale;
    }
  }
  return peak;
}

function runNextRender() {
  if (rendering || !renderJobs.length) return;
  rendering = true;
  const job = renderJobs.shift();
  job.run().then(job.resolve, job.reject).finally(() => {
    rendering = false;
    runNextRender();
  });
}

function prioritize(job) {
  const index = renderJobs.indexOf(job);
  if (index > 0) renderJobs.unshift(renderJobs.splice(index, 1)[0]);
}

function queueRender(run, urgent) {
  const job = { run };
  job.result = new Promise((resolve, reject) => Object.assign(job, { resolve, reject }));
  if (urgent) renderJobs.unshift(job);
  else renderJobs.push(job);
  runNextRender();
  return job;
}

function renderAudio(score, events, rate, seconds, urgent = false) {
  return queueRender(async () => {
    if (!registered) {
      audio.registerSynthSounds();
      setSoundfontUrl(fonts);
      registerSoundfonts();
      registered = true;
    }
    const cps = score.cpm / 60;
    const frames = Math.ceil(seconds * rate);
    const padding = Math.ceil(0.05 * rate);
    const context = new OfflineAudioContext(2, frames + padding, rate);
    // Superdough installs these generic effects on the live AudioContext prototype.
    for (const method of ['adjustLength', 'createReverb', 'createFeedbackDelay', 'createVowelFilter']) {
      context[method] = AudioContext.prototype[method];
    }
    const selected = events.filter(event => event.start / cps < seconds);
    audio.setRenderingContext(context);
    try {
      await loadWorklet(context);
      await loadSounds(selected, context);
      audio.setMaxPolyphony(Math.max(selected.length, 1));
      for (const [index, event] of selected.entries()) {
        const value = { ...event.value };
        if (value.roomsize !== undefined) value.roomsize = Math.min(8, Math.max(0.01, Number(value.roomsize) || 2));
        if (value.roomfade !== undefined) value.roomfade = Math.min(8, Math.max(0, Number(value.roomfade) || 0));
        const at = event.start / cps;
        try {
          await audio.superdough(value, at + padding / rate, Math.min(event.duration / cps, seconds - at), cps, event.start);
        } catch (error) {
          throw new UnplayablePatternError(`Could not render “${soundName(value)}”: ${error.message}`);
        }
        if (index % 128 === 127) await new Promise(resolve => setTimeout(resolve, 0));
      }
      await finishReverbs();
      const buffer = await context.startRendering();
      const peak = leaveHeadroom(buffer);
      return { buffer, padding, frames, rate, peak };
    } finally {
      await finishReverbs().catch(() => {});
      audio.releaseAudioGraph();
      audio.setRenderingContext(undefined);
    }
  }, urgent);
}

function previewRate(seconds) {
  // Very long arrangements stay within roughly 96 MB of stereo PCM.
  return Math.min(44100, Math.floor(12_000_000 / seconds));
}

export async function prepareMix(score) {
  latestMix = score.events;
  const clip = await previewAudio(score, score.events, true);
  if (clip.peak < 1 / 32768) throw new UnplayablePatternError('The pattern rendered silent audio.');
  return clip;
}

export function previewAudio(score, events, urgent = false) {
  const previous = rendered.get(events);
  if (previous) {
    rendered.delete(events);
    rendered.set(events, previous);
    return Promise.resolve(previous);
  }
  const underway = pending.get(events);
  if (underway) {
    if (urgent) prioritize(underway.job);
    return underway.result;
  }

  const seconds = score.cycles * 60 / score.cpm;
  const job = renderAudio(score, events, previewRate(seconds), seconds, urgent);
  const result = job.result.then(clip => {
    const size = clip.buffer.length * 8;
    if (size <= cacheBudget) {
      while (cachedBytes + size > cacheBudget) {
        const evict = [...rendered].find(([key]) => key !== latestMix);
        if (!evict) return clip;
        const [key, oldest] = evict;
        cachedBytes -= oldest.buffer.length * 8;
        rendered.delete(key);
      }
      rendered.set(events, clip);
      cachedBytes += size;
    }
    return clip;
  }).finally(() => pending.delete(events));
  pending.set(events, { job, result });
  return result;
}

export function stop() {
  if (!active) return;
  const playback = active;
  active = undefined;
  cancelAnimationFrame(playback.frame);
  fadeOut(playback.source);
}

export async function warmPlayback() {
  outputContext ??= new AudioContext({ latencyHint: 'interactive' });
  if (outputContext.state !== 'running') await outputContext.resume();
}

function fadeOut(playback) {
  if (!playback) return;
  const now = outputContext.currentTime;
  playback.gain.gain.cancelScheduledValues(now);
  playback.gain.gain.setTargetAtTime(0, now, 0.002);
  playback.source.stop(now + 0.01);
}

function startAt(session) {
  const { clip } = session;
  const position = Math.min(Math.max(0, session.position), (clip.frames - 1) / clip.rate);
  const source = outputContext.createBufferSource();
  const gain = outputContext.createGain();
  const now = outputContext.currentTime;
  source.buffer = clip.buffer;
  source.loop = true;
  source.loopStart = clip.padding / clip.rate;
  source.loopEnd = (clip.padding + clip.frames) / clip.rate;
  source.connect(gain).connect(outputContext.destination);
  source.onended = () => {
    source.disconnect();
    gain.disconnect();
  };
  gain.gain.setValueAtTime(0, now);
  gain.gain.linearRampToValueAtTime(1, now + 0.008);
  source.start(now, source.loopStart + position);
  fadeOut(session.source);
  session.source = { source, gain };
  session.position = position;
  session.startedAt = now;
}

export function seek(seconds) {
  if (!active) return;
  active.position = seconds;
  if (active.clip) startAt(active);
}

export async function play(score, events, seconds, { ready, progress, failed }) {
  stop();
  const session = { position: seconds };
  active = session;
  try {
    const [clip] = await Promise.all([previewAudio(score, events, true), warmPlayback()]);
    if (active !== session) return;
    session.clip = clip;
    startAt(session);

    const duration = clip.frames / clip.rate;
    const animate = () => {
      const elapsed = Math.max(0, outputContext.currentTime - session.startedAt);
      progress((session.position + elapsed) % duration);
      session.frame = requestAnimationFrame(animate);
    };
    ready(clip);
    animate();
  } catch (error) {
    if (active !== session) return;
    stop();
    failed(error);
  }
}

export async function makeWav(score, events, { rate, channels, repeats, seconds }) {
  const duration = score.cycles * 60 / score.cpm;
  if (seconds * repeats > maxExportSeconds) throw new Error('Choose an export length of four minutes or less.');
  const clip = rate === previewRate(duration) && seconds === duration
    ? await previewAudio(score, events, true) : await renderAudio(score, events, rate, seconds, true).result;
  const size = clip.frames * channels * 2 * repeats;
  const bytes = new Uint8Array(44 + size);
  const header = new DataView(bytes.buffer);
  const text = new TextEncoder();
  bytes.set(text.encode('RIFF'), 0);
  header.setUint32(4, size + 36, true);
  bytes.set(text.encode('WAVEfmt '), 8);
  header.setUint32(16, 16, true);
  header.setUint16(20, 1, true);
  header.setUint16(22, channels, true);
  header.setUint32(24, rate, true);
  header.setUint32(28, rate * channels * 2, true);
  header.setUint16(32, channels * 2, true);
  header.setUint16(34, 16, true);
  bytes.set(text.encode('data'), 36);
  header.setUint32(40, size, true);

  const [left, right] = [clip.buffer.getChannelData(0), clip.buffer.getChannelData(1)];
  const pcm = value => Math.round(Math.max(-1, Math.min(1, value || 0)) * (value < 0 ? 32768 : 32767));
  let offset = 44;
  for (let i = clip.padding; i < clip.padding + clip.frames; i++) {
    if (channels === 1) {
      header.setInt16(offset, pcm((left[i] + right[i]) / 2), true);
      offset += 2;
    } else {
      header.setInt16(offset, pcm(left[i]), true);
      header.setInt16(offset + 2, pcm(right[i]), true);
      offset += 4;
    }
  }
  const pass = bytes.subarray(44, offset);
  for (let repeat = 1; repeat < repeats; repeat++) bytes.set(pass, 44 + repeat * pass.length);
  return new Blob([bytes], { type: 'audio/wav' });
}
