import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { handleCompose } from '../api/compose.mjs';
import { previewTransport } from './modal-preview.mjs';

const port = Number(process.env.PORT || 4173);
const types = { html: 'text/html', js: 'text/javascript', css: 'text/css', jpg: 'image/jpeg', svg: 'image/svg+xml' };
const localHosts = new Set([`127.0.0.1:${port}`, `localhost:${port}`]);
createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  if (url.pathname === '/tunelm/api/compose') {
    if (!localHosts.has(req.headers.host) ||
        (req.headers.origin && req.headers.origin !== `http://${req.headers.host}`)) {
      res.writeHead(403).end();
      return;
    }
    try {
      if (req.method === 'POST') {
        if (req.headers['content-type']?.split(';', 1)[0] !== 'application/json') {
          res.writeHead(415).end();
          return;
        }
        const chunks = [];
        let size = 0;
        for await (const chunk of req) {
          size += chunk.length;
          if (size > 1024 * 1024) throw new Error('Request too large');
          chunks.push(chunk);
        }
        req.body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      }
      req.query = Object.fromEntries(url.searchParams);
      res.status = status => { res.statusCode = status; return res; };
      res.json = body => {
        res.setHeader('Content-Type', 'application/json');
        res.end(JSON.stringify(body));
      };
      return await handleCompose(req, res, previewTransport);
    } catch {
      res.writeHead(400).end();
      return;
    }
  }
  const path = ['/', '/tunelm', '/tunelm/'].includes(url.pathname)
    ? '/tunelm/index.html' : url.pathname;
  if (!/^\/tunelm\/(?:index\.html|assets\/[\w-]+\.(?:js|css|jpg|svg))$/.test(path)) {
    res.writeHead(404).end();
    return;
  }
  try {
    const source = await readFile(new URL(`../dist${path}`, import.meta.url));
    const headers = { 'Content-Type': types[path.split('.').at(-1)] };
    if (path.endsWith('/pattern-worker.js')) {
      headers['Content-Security-Policy'] = "default-src 'none'; script-src 'self'; connect-src 'none'";
    }
    res.writeHead(200, headers).end(source);
  } catch {
    res.writeHead(404).end();
  }
}).listen(port, '127.0.0.1', () => console.log(`TuneLM at http://127.0.0.1:${port}/tunelm`));
