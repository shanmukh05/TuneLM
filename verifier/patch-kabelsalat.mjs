/** Select the published ESM entry for the pinned Strudel audio dependency.
 * Fail closed if package layout changes rather than silently patching unknown code.
 */
import { readFile, writeFile } from 'node:fs/promises';
const path = new URL('../node_modules/@kabelsalat/web/package.json', import.meta.url);
const pkg = JSON.parse(await readFile(path, 'utf8'));
if (pkg.version !== '0.4.1') throw new Error('Review @kabelsalat/web patch for new version');
pkg.main = 'dist/index.mjs';
await writeFile(path, `${JSON.stringify(pkg, null, 2)}\n`);
