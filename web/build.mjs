import { cp, mkdir, readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { offlineAudio } from './superdough-build.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const output = resolve(root, 'dist/tunelm');
await mkdir(`${output}/assets`, { recursive: true });

await Promise.all([
  cp(`${root}/web/index.html`, `${output}/index.html`),
  cp(`${root}/web/tunelm-mark.svg`, `${output}/assets/tunelm-mark.svg`),
  build({
    absWorkingDir: root,
    entryPoints: {
      studio: 'web/studio.mjs',
      'pattern-worker': 'web/pattern-worker.mjs',
      'audio-worklets': 'node_modules/superdough/worklets.mjs',
    },
    outdir: `${output}/assets`,
    bundle: true,
    minify: true,
    format: 'esm',
    platform: 'browser',
    target: 'es2022',
    loader: { '.jpg': 'file' },
    legalComments: 'eof',
    plugins: [offlineAudio(root), {
      name: 'strudel-soundfont-data',
      setup(build) {
        build.onLoad({ filter: /@strudel[/\\]soundfonts[/\\]fontloader\.mjs$/ }, async ({ path }) => {
          const source = await readFile(path, 'utf8');
          const upstream = "let [_, data] = preset.split('={');\n    return eval('{' + data);";
          const findZone = `return preset.find((zone) => {
    return zone.keyRangeLow <= pitch && zone.keyRangeHigh + 1 >= pitch;
  });`;
          if (!source.includes(upstream) || !source.includes(findZone)) {
            throw new Error('Review the updated Strudel soundfont loader.');
          }
          return {
            contents: `import { readSoundfont, nearestZone } from ${JSON.stringify(resolve(root, 'web/soundfont-data.mjs'))};\n` +
              source.replace(upstream, 'return readSoundfont(preset);')
                .replace(findZone, 'return nearestZone(preset, pitch);')
                .replace("from '@strudel/webaudio'", "from 'superdough'"),
          };
        });
      },
    }],
  }),
]);
