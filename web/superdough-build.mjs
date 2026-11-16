import { readFile } from 'node:fs/promises';
import { basename, resolve } from 'node:path';

function replace(source, original, replacement) {
  if (!source.includes(original)) throw new Error('Review TuneLM’s adapter for the updated Superdough source.');
  return source.replace(original, replacement);
}

// Adapt the pinned browser engine for independent stereo OfflineAudioContexts.
export function offlineAudio(root) {
  return {
    name: 'superdough-offline-audio',
    setup(build) {
      build.onResolve({ filter: /^superdough$/ }, () => ({
        path: resolve(root, 'node_modules/superdough/index.mjs'),
      }));
      build.onResolve({ filter: /worklets\.mjs\?audioworklet$/ }, () => ({
        path: 'audio-worklets', namespace: 'tunelm-audio',
      }));
      build.onLoad({ filter: /.*/, namespace: 'tunelm-audio' }, () => ({
        contents: "export default new URL('./audio-worklets.js', import.meta.url).href;",
      }));
      build.onLoad({
        filter: /[/\\]superdough[/\\](audioContext|superdough|superdoughoutput|reverbGen)\.mjs$/,
      }, async ({ path }) => {
        const source = await readFile(path, 'utf8');
        switch (basename(path)) {
          case 'audioContext.mjs':
            return { contents: source + '\nexport function setRenderingContext(context) { audioContext = context; }' };
          case 'superdough.mjs':
            return { contents: source + `
              export function releaseAudioGraph() {
                controller?.output.disconnect();
                controller = undefined;
                activeSoundSources.clear();
              }
            ` };
          case 'superdoughoutput.mjs':
            return { contents: replace(source,
              '    this.audioContext.destination.channelCount = maxChannelCount;',
              '    if (!(audioContext instanceof OfflineAudioContext)) audioContext.destination.channelCount = maxChannelCount;') };
          case 'reverbGen.mjs':
            return { contents: replace(source, `  context.oncomplete = function (event) {
    callback(event.renderedBuffer);
  };
  context.startRendering();

  window.filterNode = filter;`, '  pending.add(context.startRendering().then(callback));') + `
              const pending = new Set();
              export async function finishReverbs() {
                const renders = [...pending];
                pending.clear();
                await Promise.all(renders);
              }
            ` };
        }
      });
    },
  };
}
