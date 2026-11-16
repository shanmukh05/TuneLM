import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { modalHttpTransport } from '../api/compose.mjs';

const script = fileURLToPath(new URL('./modal-preview.py', import.meta.url));
const venv = fileURLToPath(new URL(
  process.platform === 'win32' ? '../.venv/Scripts/python.exe' : '../.venv/bin/python',
  import.meta.url,
));
const python = existsSync(venv) ? venv : process.platform === 'win32' ? 'python' : 'python3';
const signature = randomBytes(32);

function setupError(message) {
  return Object.assign(new Error(message), { localSetup: true });
}

function sdkRequest(method, job, input) {
  return new Promise((resolve, reject) => {
    const child = spawn(python, [script, method], {
      stdio: ['pipe', 'pipe', 'ignore'],
      timeout: 25000,
    });
    let output = '';
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', chunk => { output += chunk; });
    child.stdin.on('error', reject);
    child.on('error', error => reject(error.code === 'ENOENT'
      ? setupError('Python is needed for local Modal inference. Run uv sync --extra modal.')
      : error));
    child.on('close', code => {
      if (code !== 0) {
        reject(new Error('The local Modal client could not complete the request.'));
        return;
      }
      try {
        const { status, body } = JSON.parse(output);
        if (status === 503) throw setupError(body.error);
        resolve(Response.json(body, { status }));
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(JSON.stringify({ job, ...input }));
  });
}

export function previewTransport() {
  if (['TUNELM_MODAL_URL', 'TUNELM_MODAL_KEY', 'TUNELM_MODAL_SECRET'].some(key => process.env[key])) {
    return modalHttpTransport();
  }
  return { local: true, secret: signature, request: sdkRequest };
}
