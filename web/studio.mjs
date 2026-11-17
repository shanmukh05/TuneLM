import './studio.css';
import MarkdownIt from 'markdown-it';
import { makeWav, maxExportSeconds, play, prepareMix, previewAudio, seek, stop, UnplayablePatternError, warmPlayback } from './player.mjs';
import { musicFacts } from './music-facts.mjs';

const notesMarkdown = new MarkdownIt({ html: false, breaks: true }).disable('image');

const icons = {
  plus: '<path d="M12 5v14M5 12h14"/>',
  moon: '<path d="M20.5 15.4A8.6 8.6 0 0 1 8.6 3.5a8.6 8.6 0 1 0 11.9 11.9Z"/>',
  note: '<path d="M9 18V5l11-2v13M9 9l11-2"/><ellipse cx="6.5" cy="18" rx="2.5" ry="2"/><ellipse cx="17.5" cy="16" rx="2.5" ry="2"/>',
  activity: '<path d="M2 12h4l3-6 4 12 3-6h6"/>',
  arrow: '<path d="M4 12h15m-6-6 6 6-6 6"/>',
  wave: '<path d="M3 10v4m4-7v10m5-14v18m5-14v10m4-7v4"/>',
  code: '<path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-15-2 18"/>',
  play: '<path fill="currentColor" stroke="none" d="M7 4.8a.8.8 0 0 1 1.2-.7l11 7.2a.8.8 0 0 1 0 1.4l-11 7.2a.8.8 0 0 1-1.2-.7z"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="1.3" fill="currentColor" stroke="none"/>',
  repeat: '<path d="m17 2 4 4-4 4M3 11V9a3 3 0 0 1 3-3h15M7 22l-4-4 4-4m14-3v4a3 3 0 0 1-3 3H3"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
  download: '<path d="M12 3v12m-4-4 4 4 4-4M4 16v4h16v-4"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  user: '<circle cx="12" cy="8" r="3"/><path d="M5 20v-1a5 5 0 0 1 5-5h4a5 5 0 0 1 5 5v1"/>',
};
const colors = [
  ['#577045', '#edf1e5'], ['#956048', '#f5e9df'], ['#756188', '#efe9f4'],
  ['#4b7280', '#e7eef0'], ['#89703b', '#f2eddf'], ['#7d5771', '#f4e8ef'],
];

const conversation = document.querySelector('#conversation');
const form = document.querySelector('#composer');
const input = document.querySelector('#prompt');
const submit = form.querySelector('button');
const template = document.querySelector('#reply-template');
const downloadDialog = document.querySelector('#download-dialog');
const downloadForm = document.querySelector('#download-form');
const downloadStatus = downloadForm.querySelector('.download-status');
const downloadConfirm = downloadForm.querySelector('.download-confirm');
let download;
const endpoint = '/tunelm/api/compose';
const maxRepairs = 2;
let request;
let playing;
let nextId = 0;
let nextFact = Math.floor(Math.random() * musicFacts.length);
const history = [];

function icon(node, name) {
  node.innerHTML = `<span class="icon" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${icons[name]}</svg></span>`;
}
function installIcons(root) {
  for (const node of root.querySelectorAll('[data-icon]')) icon(node, node.dataset.icon);
}
function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined) node.textContent = value;
  return node;
}
function makeIcon(name, className) {
  const node = element('span', className);
  icon(node, name);
  return node;
}
function beginTurn(prompt) {
  const turn = element('div', 'exchange');
  const user = element('div', 'user-turn');
  user.append(element('p', 'user-bubble', prompt), makeIcon('user', 'user-icon'));
  turn.append(user);
  conversation.append(turn);
  return turn;
}
function assistantTurn(turn) {
  const reply = element('article', 'reply');
  const brand = element('div', 'reply-brand');
  brand.append(makeIcon('activity', 'mini-brand'), element('strong', '', 'TuneLM'));
  reply.append(brand);
  turn.append(reply);
  return reply;
}
function compactTitle(prompt) {
  const words = prompt.trim().split(/\s+/);
  return words.length > 8 ? `${words.slice(0, 8).join(' ')}…` : prompt;
}
function time(seconds) {
  const rounded = Math.floor(seconds);
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, '0')}`;
}
function splitResponse(raw) {
  const thought = /^\s*<think>([\s\S]*?)<\/think>/.exec(raw);
  const content = thought ? raw.slice(thought[0].length) : raw;
  const opening = /^\s*```(?:strudel|javascript|js)[ \t]*\r?\n/.exec(content);
  const program = opening ? content.slice(opening[0].length) : '';
  const closing = /\r?\n```\s*$/.exec(program);
  const attemptedCode = (closing ? program.slice(0, closing.index) : program).trim();
  return {
    thinking: thought?.[1].trim() || '',
    code: closing && !attemptedCode.includes('```') ? attemptedCode : '',
    attemptedCode,
  };
}
function highlight(node, code) {
  const pattern = /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|(\b(?:const|return|let|function)\b)|(\b\d+(?:\.\d+)?\b)/g;
  const classes = ['comment', 'string', 'keyword', 'number'];
  let offset = 0;
  for (const match of code.matchAll(pattern)) {
    node.append(document.createTextNode(code.slice(offset, match.index)));
    const type = classes[match.slice(1).findIndex(Boolean)];
    node.append(element('span', `syntax-${type}`, match[0]));
    offset = match.index + match[0].length;
  }
  node.append(document.createTextNode(code.slice(offset)));
}
function bindTabs(reply, id, selected = 0) {
  const tabs = Array.from(reply.querySelectorAll('[role=tab]'));
  const panels = Array.from(reply.querySelectorAll('[role=tabpanel]'));
  function activate(index) {
    tabs.forEach((tab, i) => {
      tab.setAttribute('aria-selected', i === index);
      tab.tabIndex = i === index ? 0 : -1;
      panels[i].hidden = i !== index;
      panels[i].tabIndex = i === index ? 0 : -1;
    });
  }
  for (const [index, tab] of tabs.entries()) {
    tab.id = `tab-${id}-${index}`;
    panels[index].id = `panel-${id}-${index}`;
    tab.setAttribute('aria-controls', panels[index].id);
    panels[index].setAttribute('aria-labelledby', tab.id);
    tab.addEventListener('click', () => activate(index));
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : 1 - index;
      activate(next);
      tabs[next].focus();
    });
  }
  activate(selected);
}
function readPattern(code, signal) {
  signal?.throwIfAborted();
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./pattern-worker.js', import.meta.url), { type: 'module' });
    const finish = (error, score) => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      worker.terminate();
      if (error) reject(error);
      else resolve(score);
    };
    const abort = () => finish(signal.reason);
    const timer = setTimeout(() => finish(new UnplayablePatternError('This pattern took too long to compile.')), 6500);
    signal?.addEventListener('abort', abort, { once: true });
    worker.onmessage = ({ data }) => finish(data.error ? new UnplayablePatternError(data.error) : null, data.score);
    worker.onerror = () => finish(new Error('The Strudel pattern player could not start.'));
    worker.postMessage(code);
  });
}
function stopPlaying() {
  if (!playing) return;
  const previous = playing;
  playing = undefined;
  stop();
  previous.reset();
}

function bindOverviewExample(example) {
  const button = example.querySelector('.example-play');
  const codeBlock = example.querySelector('code');
  const errorNote = example.querySelector('.example-error');
  const code = codeBlock.textContent;
  let score;

  codeBlock.replaceChildren();
  highlight(codeBlock, code);
  function setButton(label, symbol) {
    icon(button.firstElementChild, symbol);
    button.lastElementChild.textContent = label;
  }
  button.addEventListener('click', async () => {
    const wasActive = playing?.example === example;
    stopPlaying();
    if (wasActive) return;
    const session = { example, reset: () => setButton('Play code', 'play') };
    playing = session;
    errorNote.hidden = true;
    setButton('Cancel render', 'stop');
    function failed(error) {
      if (playing !== session) return;
      stopPlaying();
      errorNote.hidden = false;
      errorNote.textContent = error.message;
    }
    try {
      const [, pattern] = await Promise.all([warmPlayback(), score ?? readPattern(code)]);
      if (playing !== session) return;
      score = pattern;
      await play(score, score.events, 0, {
        ready: () => setButton('Stop example', 'stop'),
        progress: () => {},
        failed,
      });
    } catch (error) {
      failed(error);
    }
  });
  button.hidden = false;
}

function paintWaveform(canvas, clip, color) {
  const ctx = canvas.getContext('2d');
  canvas.width = 800;
  canvas.height = 96;
  const [left, right] = [clip.buffer.getChannelData(0), clip.buffer.getChannelData(1)];
  const { width, height } = canvas;
  const low = new Float32Array(width);
  const high = new Float32Array(width);
  let peak = 0;
  for (let x = 0; x < width; x++) {
    const from = clip.padding + Math.floor(x * clip.frames / width);
    const to = Math.min(left.length, clip.padding + Math.ceil((x + 1) * clip.frames / width));
    let min = 0;
    let max = 0;
    for (let i = from; i < to; i++) {
      min = Math.min(min, left[i] || 0, right[i] || 0);
      max = Math.max(max, left[i] || 0, right[i] || 0);
    }
    low[x] = min;
    high[x] = max;
    peak = Math.max(peak, -min, max);
  }
  const mid = height / 2;
  const scale = mid * 0.8 / Math.max(peak, 0.001);
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.17;
  ctx.fillRect(0, mid, width, 1.6);
  ctx.globalAlpha = 0.88;
  for (let x = 0; x < width; x++) {
    ctx.fillRect(x, mid - high[x] * scale, 1, (high[x] - low[x]) * scale);
  }
}

function addChoice(select, value, label) {
  select.add(new Option(label, value));
}

function showDownload(reply, score, selection) {
  const title = reply.querySelector('.track-title').textContent;
  const duration = score.cycles * 60 / score.cpm;
  download = { score, title, duration };
  downloadForm.reset();
  downloadConfirm.disabled = false;
  const track = downloadForm.elements.namedItem('track');
  const length = downloadForm.elements.namedItem('length');
  track.replaceChildren();
  addChoice(track, -1, 'Full mix');
  score.tracks.forEach((part, i) => addChoice(track, i, part.name));
  track.value = selection ?? -1;
  length.replaceChildren();
  for (const repeats of [1, 2, 4]) {
    const seconds = duration * repeats;
    if (seconds <= maxExportSeconds) {
      addChoice(length, repeats, `${repeats === 1 ? 'One pass' : `${repeats} passes`} · ${time(seconds)}`);
    }
  }
  for (const seconds of [30, 60]) {
    if (seconds < duration) addChoice(length, `first-${seconds}`, `First ${seconds} seconds`);
  }
  updateDownloadSize();
  downloadDialog.showModal();
}

function downloadSettings(duration) {
  const { track, rate, channels } = downloadForm.elements;
  const length = downloadForm.elements.namedItem('length');
  const excerpt = length.value.startsWith('first-');
  return {
    track: Number(track.value),
    rate: Number(rate.value),
    channels: Number(channels.value),
    repeats: excerpt ? 1 : Number(length.value),
    seconds: excerpt ? Number(length.value.slice(6)) : duration,
  };
}

function updateDownloadSize() {
  if (!download) return;
  const { seconds, rate, channels, repeats } = downloadSettings(download.duration);
  const megabytes = seconds * rate * channels * 2 * repeats / 1_000_000;
  downloadStatus.classList.remove('error');
  downloadStatus.textContent = `Approx. ${megabytes.toFixed(1)} MB`;
}

downloadForm.addEventListener('change', () => {
  if (!downloadConfirm.disabled) updateDownloadSize();
});
downloadForm.querySelector('.dialog-close').addEventListener('click', () => downloadDialog.close());
downloadDialog.addEventListener('close', () => {
  if (!downloadDialog.open) download = undefined;
});
downloadForm.addEventListener('submit', async event => {
  event.preventDefault();
  const job = download;
  if (!job || downloadConfirm.disabled) return;
  const settings = downloadSettings(job.duration);
  const stem = job.score.tracks[settings.track];
  downloadConfirm.disabled = true;
  downloadStatus.classList.remove('error');
  downloadStatus.textContent = 'Preparing the WAV file…';
  try {
    const file = await makeWav(job.score, stem?.events ?? job.score.events, settings);
    if (download !== job || !downloadDialog.open) return;
    const name = `${job.title} ${stem?.name ?? 'mix'}`.toLowerCase()
      .normalize('NFKD').replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    const url = URL.createObjectURL(file);
    const link = element('a');
    link.href = url;
    link.download = `${name || 'tunelm-audio'}.wav`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    downloadDialog.close();
  } catch (error) {
    if (download !== job || !downloadDialog.open) return;
    downloadStatus.classList.add('error');
    downloadStatus.textContent = error.message;
  } finally {
    if (download === job) downloadConfirm.disabled = false;
  }
});

function drawTimeline(reply, score, mixClip) {
  const timeline = reply.querySelector('.timeline');
  const ruler = reply.querySelector('.ruler');
  const lanes = reply.querySelector('.lanes');
  const scrubber = reply.querySelector('.timeline-seek');
  const playhead = reply.querySelector('.playhead');
  const transport = reply.querySelector('.transport');
  const mix = transport.querySelector('.play-mix');
  const clock = transport.querySelector('.time');
  const playbackError = reply.querySelector('.playback-error');
  const duration = score.cycles * 60 / score.cpm;
  let position = 0;
  let drag;
  const ticks = Math.max(1, Math.min(4, Math.floor(duration)));
  for (let index = 0; index < ticks; index++) {
    const tick = element('span', '', time(duration * index / ticks));
    tick.style.left = `${index / ticks * 100}%`;
    ruler.append(tick);
  }
  ruler.append(element('span', 'ruler-end', time(duration)));
  scrubber.setAttribute('aria-valuemax', String(duration));

  function setPosition(seconds) {
    position = Math.min(duration, Math.max(0, seconds));
    playhead.style.left = `${position / duration * 100}%`;
    const stamp = time(position);
    const label = `${stamp} / ${time(duration)}`;
    if (clock.textContent !== label) clock.textContent = label;
    const value = String(Math.min(duration, Math.round(position * 10) / 10));
    if (scrubber.getAttribute('aria-valuenow') !== value) {
      scrubber.setAttribute('aria-valuenow', value);
    }
    const spoken = `${stamp} of ${time(duration)}`;
    if (scrubber.getAttribute('aria-valuetext') !== spoken) {
      scrubber.setAttribute('aria-valuetext', spoken);
    }
  }

  function moveTo(seconds) {
    setPosition(seconds);
    if (playing?.reply === reply) seek(position);
  }

  function pointerTime(event) {
    const { left, width } = scrubber.getBoundingClientRect();
    return ((event.clientX - left) / width) * duration;
  }

  scrubber.addEventListener('pointerdown', event => {
    if (drag || !event.isPrimary || event.button !== 0) return;
    if (event.pointerType === 'mouse') scrubber.focus({ preventScroll: true });
    drag = { id: event.pointerId, previous: position };
    scrubber.setPointerCapture(event.pointerId);
    scrubber.classList.add('dragging');
    setPosition(pointerTime(event));
  });
  scrubber.addEventListener('pointermove', event => {
    if (drag?.id === event.pointerId) setPosition(pointerTime(event));
  });
  scrubber.addEventListener('pointerup', event => {
    if (drag?.id !== event.pointerId) return;
    drag = undefined;
    scrubber.classList.remove('dragging');
    moveTo(pointerTime(event));
  });
  scrubber.addEventListener('lostpointercapture', event => {
    if (drag?.id !== event.pointerId) return;
    setPosition(drag.previous);
    drag = undefined;
    scrubber.classList.remove('dragging');
  });
  scrubber.addEventListener('keydown', event => {
    const step = Math.min(1, duration / 10);
    const positions = {
      ArrowLeft: position - step,
      ArrowDown: position - step,
      ArrowRight: position + step,
      ArrowUp: position + step,
      PageDown: position - 5 * step,
      PageUp: position + 5 * step,
      Home: 0,
      End: duration,
    };
    if (!Object.hasOwn(positions, event.key)) return;
    event.preventDefault();
    moveTo(positions[event.key]);
  });
  const laneButtons = [];
  const laneNodes = [];
  const waveforms = [];
  function addLane(name, mark, tint, button, events) {
    const lane = element('div', button ? 'lane' : 'lane mix-lane');
    lane.style.setProperty('--mark', mark);
    lane.style.setProperty('--lane', tint);
    lane.style.setProperty('--step', `${100 / score.cycles}%`);
    const label = element('div', 'lane-label');
    const text = element('span', 'lane-name', name);
    text.title = name;
    label.append(element('span', 'lane-dot'), text);
    if (button) label.append(button);
    const cells = element('div', 'lane-cells');
    const canvas = element('canvas');
    canvas.setAttribute('aria-label', `${name} audio waveform`);
    canvas.setAttribute('role', 'img');
    canvas.hidden = true;
    const status = element('span', 'wave-status', 'Rendering waveform…');
    cells.append(canvas, status);
    lane.append(label, cells);
    lanes.append(lane);
    const waveform = { canvas, status, events, mark, rendered: false };
    waveforms.push(waveform);
    return lane;
  }

  function updateWaveform(waveform, clip) {
    if (waveform.rendered) return;
    paintWaveform(waveform.canvas, clip, waveform.mark);
    waveform.rendered = true;
    waveform.canvas.hidden = false;
    waveform.status.hidden = true;
  }

  async function prepareWaveforms() {
    for (const waveform of waveforms.slice(1)) {
      if (!reply.isConnected) return;
      if (waveform.rendered) continue;
      try {
        const clip = await previewAudio(score, waveform.events);
        updateWaveform(waveform, clip);
      } catch {
        if (!waveform.rendered) waveform.status.textContent = 'Waveform unavailable';
      }
    }
  }

  const mixLane = addLane('Full mix', '#79442f', '#f2e6dc', null, score.events);
  function setTransport(index, loading = false) {
    const mixPlaying = index === -1;
    icon(mix.firstElementChild, mixPlaying ? 'stop' : 'play');
    mix.lastElementChild.textContent = mixPlaying ? (loading ? 'Cancel' : 'Stop mix') : 'Play mix';
    mix.setAttribute('aria-pressed', String(mixPlaying));
    mixLane.classList.toggle('inactive', index >= 0);
    laneButtons.forEach((button, i) => {
      const selected = index === i;
      const action = selected ? (loading ? 'Cancel' : 'Stop') : 'Solo';
      const label = `${action} ${score.tracks[i].name}`;
      icon(button, selected ? 'stop' : 'play');
      button.title = label;
      button.setAttribute('aria-label', label);
      button.setAttribute('aria-pressed', String(selected));
      laneNodes[i].classList.toggle('inactive', index >= 0 && !selected);
    });
  }
  function toggle(index) {
    const wasActive = playing?.reply === reply && playing.index === index;
    stopPlaying();
    if (wasActive) return;
    const reset = () => setTransport(null);
    playing = { reply, index, reset };
    playbackError.hidden = true;
    setTransport(index, true);
    play(score, index === -1 ? score.events : score.tracks[index].events, position, {
      ready: clip => {
        updateWaveform(waveforms[index + 1], clip);
        setTransport(index);
      },
      progress: seconds => {
        if (!drag) setPosition(seconds);
      },
      failed: error => {
        playing = undefined;
        reset();
        playbackError.textContent = error.message;
        playbackError.hidden = false;
      },
    });
  }
  score.tracks.forEach((track, index) => {
    const [mark, tint] = colors[index % colors.length];
    const button = element('button', 'lane-play');
    button.type = 'button';
    button.addEventListener('click', () => toggle(index));
    const lane = addLane(track.name, mark, tint, button, track.events);
    laneButtons.push(button);
    laneNodes.push(lane);
  });
  setPosition(0);
  mix.addEventListener('click', () => toggle(-1));
  transport.querySelector('.download-button').addEventListener('click', () => {
    showDownload(reply, score, playing?.reply === reply ? playing.index : -1);
  });
  setTransport(null);
  timeline.hidden = false;
  transport.hidden = false;
  updateWaveform(waveforms[0], mixClip);
  void prepareWaveforms();
}
function showReply(turn, { thinking, code, score, clip }, title) {
  const reply = template.content.firstElementChild.cloneNode(true);
  installIcons(reply);
  reply.querySelector('.track-title').textContent = title;
  reply.querySelector('.thinking').hidden = !thinking;
  const notes = reply.querySelector('.notes-body');
  notes.innerHTML = notesMarkdown.render(thinking);
  const preview = notes.querySelector('p, li') ?? notes.firstElementChild;
  reply.querySelector('.notes-preview').textContent = preview?.textContent.trim() ?? '';
  const codeBlock = reply.querySelector('.code-panel code');
  highlight(codeBlock, code);
  const copy = reply.querySelector('.copy-button');
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(code);
      copy.lastChild.textContent = ' Copied';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(codeBlock);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      copy.lastChild.textContent = ' Code selected';
    }
  });
  bindTabs(reply, ++nextId);
  reply.querySelector('.track-meta').textContent = [
    `${Math.round(score.cpm * 4)} BPM`, `${score.tracks.length} ${score.tracks.length === 1 ? 'layer' : 'layers'}`,
    `${score.cycles} cycles${score.arrangement ? '' : ' · loop preview'}`,
  ].join(' · ');
  turn.append(reply);
  drawTimeline(reply, score, clip);
}
function showPending(turn) {
  const message = assistantTurn(turn);
  const pending = element('div', 'pending-card');
  const content = element('div', 'pending-copy');
  const status = element('p', 'pending-status');
  status.setAttribute('role', 'status');
  const wave = element('span', 'loading-wave');
  wave.setAttribute('aria-hidden', 'true');
  for (let i = 0; i < 7; i++) {
    const bar = element('i');
    bar.style.setProperty('--bar', i);
    wave.append(bar);
  }
  const stage = document.createTextNode('Composing your music… This may take a few minutes.');
  status.append(wave, stage);
  const text = element('a', 'fact-copy');
  text.target = '_blank';
  text.rel = 'noreferrer';
  content.append(status, text);
  const art = element('div', 'fact-art');
  art.setAttribute('aria-hidden', 'true');
  pending.append(content, art);
  message.append(pending);
  const rotate = () => {
    if (pending.contains(document.activeElement)) return;
    const { copy, source, paper, mark, sheet, size, position } = musicFacts[nextFact++ % musicFacts.length];
    text.textContent = copy;
    text.href = source;
    pending.style.setProperty('--fact-paper', paper);
    pending.style.setProperty('--fact-mark', mark);
    art.style.backgroundImage = `url("${sheet}")`;
    art.style.backgroundSize = size;
    art.style.backgroundPosition = position;
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
      text.animate([{ opacity: 0, transform: 'translateY(5px)' }, { opacity: 1, transform: 'none' }], 350);
      art.animate([{ opacity: 0.25 }, { opacity: 1 }], 500);
    }
  };
  rotate();
  const interval = setInterval(rotate, 8500);
  return {
    setStage: label => { stage.textContent = label; },
    remove: () => { clearInterval(interval); message.remove(); },
  };
}
async function json(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error ||
    (response.status === 429 ? 'Too many compositions. Wait a moment and try again.' : 'The music service is unavailable.'));
  return data;
}
function wait(ms, signal) {
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); reject(signal.reason); };
    const timer = setTimeout(() => {
      signal.removeEventListener('abort', abort);
      resolve();
    }, ms);
    signal.addEventListener('abort', abort, { once: true });
    if (signal.aborted) abort();
  });
}
async function compose(input, signal, deadline) {
  if (Date.now() >= deadline) throw new Error('This composition timed out. Please try again.');
  const started = await fetch(endpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
    signal,
  });
  const { job } = await json(started);
  if (typeof job !== 'string') throw new Error('The music service could not start this composition.');
  const url = `${endpoint}?${new URLSearchParams({ job })}`;
  let attempts = 0;
  while (Date.now() < deadline) {
    await wait(++attempts > 15 ? 5000 : 2500, signal);
    const result = await fetch(url, { signal, cache: 'no-store' });
    if (result.status === 202) continue;
    const data = await json(result);
    if (typeof data.completion !== 'string') throw new Error('The music service returned an empty response.');
    return splitResponse(data.completion);
  }
  throw new Error('This composition timed out. Please try again.');
}
async function composePlayable(prompt, history, signal, setStage) {
  const deadline = Date.now() + 16 * 60 * 1000;
  let input = { prompt, history };
  for (let attempt = 0; ; attempt++) {
    const { thinking, code, attemptedCode } = await compose(input, signal, deadline);
    try {
      setStage('Checking Strudel code…');
      if (!code) throw new UnplayablePatternError('Expected one complete, nonempty Strudel code block.');
      const score = await readPattern(code, signal);
      signal.throwIfAborted();
      setStage('Checking playback…');
      const clip = await prepareMix(score);
      signal.throwIfAborted();
      return { thinking, code, score, clip };
    } catch (error) {
      signal.throwIfAborted();
      if (!(error instanceof UnplayablePatternError)) throw error;
      if (attempt === maxRepairs) {
        throw new Error(`TuneLM couldn’t produce a playable composition after ${maxRepairs} repair attempts. Try composing again.`);
      }
      const diagnostic = error.message.replace(/\s+/g, ' ').slice(0, 500);
      const retryGuidance = attempt
        ? ' The earlier repair failed. Simplify the failing part if needed, while keeping the requested feel.'
        : '';
      input = {
        prompt: `Repair the preceding Strudel attempt for the same listener request. Preserve its musical intent.${retryGuidance}\nThe TuneLM browser player reported: ${diagnostic}\nReturn composition notes and one complete Strudel program in a fenced code block.`,
        history: [...history.slice(-5), {
          prompt,
          code: attemptedCode.slice(0, 24000) || '// The previous attempt did not include Strudel code.',
        }],
      };
      setStage(`Fixing the composition (${attempt + 1}/${maxRepairs})…`);
    }
  }
}
function showError(turn, error) {
  const note = element('p', 'error-box', error.message);
  note.setAttribute('role', 'alert');
  assistantTurn(turn).append(note);
}
function scrollToLatest(turn) {
  turn.scrollIntoView({ block: 'nearest' });
}
function newSession() {
  request?.abort();
  request = undefined;
  history.length = 0;
  submit.disabled = false;
  stopPlaying();
  conversation.replaceChildren();
  input.value = '';
  window.scrollTo(0, 0);
  input.focus({ preventScroll: true });
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  const prompt = input.value.trim();
  if (!prompt || request) return;
  void warmPlayback().catch(() => {});
  stopPlaying();
  const title = compactTitle(prompt);
  const activeRequest = new AbortController();
  request = activeRequest;
  const turn = beginTurn(prompt);
  const pending = showPending(turn);
  input.value = '';
  submit.disabled = true;
  scrollToLatest(turn);
  try {
    const result = await composePlayable(prompt, history, activeRequest.signal, pending.setStage);
    activeRequest.signal.throwIfAborted();
    showReply(turn, result, title);
    history.push({ prompt, code: result.code });
    if (history.length > 6) history.shift();
  } catch (error) {
    if (!activeRequest.signal.aborted) showError(turn, error);
  } finally {
    pending.remove();
    if (request === activeRequest) {
      request = undefined;
      submit.disabled = false;
      scrollToLatest(turn);
    }
  }
});
input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});
for (const button of document.querySelectorAll('[data-prompt]')) {
  button.addEventListener('click', () => {
    input.value = button.dataset.prompt;
    input.focus();
  });
}
const newChat = document.querySelector('#new-session');
newChat.addEventListener('click', newSession);
const pageTabs = [...document.querySelectorAll('.page-tabs button')];
let page;

function updatePage() {
  const next = location.hash === '#try' ? 'try' : 'overview';
  if (next === page) return;
  const focusedPanel = document.activeElement?.closest('.page-panel');
  if (page) stopPlaying();
  const activeTab = pageTabs.find(tab => tab.dataset.view === next);
  for (const tab of pageTabs) {
    const selected = tab === activeTab;
    tab.setAttribute('aria-selected', String(selected));
    tab.tabIndex = selected ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !selected;
  }
  newChat.hidden = next !== 'try';
  if (page) window.scrollTo(0, 0);
  if (focusedPanel?.hidden) activeTab.focus({ preventScroll: true });
  page = next;
}

pageTabs.forEach((tab, index) => {
  tab.addEventListener('click', () => { location.hash = tab.dataset.view; });
  tab.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const destination = pageTabs[event.key === 'Home' ? 0 : event.key === 'End' ? 1 : 1 - index];
    destination.focus();
    location.hash = destination.dataset.view;
  });
});
window.addEventListener('hashchange', updatePage);
updatePage();
installIcons(document);
bindOverviewExample(document.querySelector('.overview-example'));
