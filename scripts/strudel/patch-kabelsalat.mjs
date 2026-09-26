// @kabelsalat/web 0.4.1 declares ESM but points Node at its UMD build. Strudel
// imports a named export from it, so select the package's published ESM build.
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const packagePath = resolve('node_modules/@kabelsalat/web/package.json');
const pkg = JSON.parse(await readFile(packagePath, 'utf8'));
if (pkg.version !== '0.4.1') {
  throw new Error(`Review the Kabelsalat patch for version ${pkg.version}`);
}
if (pkg.main !== 'dist/index.mjs') {
  pkg.main = 'dist/index.mjs';
  await writeFile(packagePath, `${JSON.stringify(pkg, null, 2)}\n`);
  console.log('Patched @kabelsalat/web Node ESM entry point.');
}
