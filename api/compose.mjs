import { createHmac, timingSafeEqual } from 'node:crypto';

const jobLifetime = 20 * 60 * 1000;
const modalJobId = /^fc-[a-z0-9]+$/i;

function sign(payload, secret) {
  return createHmac('sha256', secret).update(`tunelm-job:${payload}`).digest();
}

function issueJob(id, secret) {
  const payload = `${id}.${Date.now() + jobLifetime}`;
  return `${payload}.${sign(payload, secret).toString('base64url')}`;
}

function readJob(token, secret) {
  if (typeof token !== 'string' || token.length > 160) return null;
  const [id, expires, mac, extra] = token.split('.');
  if (extra !== undefined || !modalJobId.test(id) || !/^\d{13}$/.test(expires) ||
      Number(expires) < Date.now() || !/^[\w-]{43}$/.test(mac)) return null;
  const signature = Buffer.from(mac, 'base64url');
  return timingSafeEqual(signature, sign(`${id}.${expires}`, secret)) ? id : null;
}

export function modalHttpTransport() {
  const { TUNELM_MODAL_URL: base, TUNELM_MODAL_KEY: key, TUNELM_MODAL_SECRET: secret } = process.env;
  if (!base || !key || !secret) return null;
  return {
    secret,
    request: (method, job, input) => fetch(`${base.replace(/\/$/, '')}/jobs${job ? `/${job}` : ''}`, {
      method,
      headers: { 'Content-Type': 'application/json', 'Modal-Key': key, 'Modal-Secret': secret },
      body: method === 'POST' ? JSON.stringify(input) : undefined,
      signal: AbortSignal.timeout(20000),
      redirect: 'error',
    }),
  };
}

function validTurn(turn) {
  return typeof turn?.prompt === 'string' && turn.prompt.trim() &&
    turn.prompt.length <= 1000 && typeof turn.code === 'string' &&
    turn.code.trim() && turn.code.length <= 24000;
}

export async function handleCompose(req, res, getTransport) {
  res.setHeader('Cache-Control', 'no-store');
  const start = req.method === 'POST';
  if (!start && req.method !== 'GET') {
    res.setHeader('Allow', 'GET, POST');
    return res.status(405).json({ error: 'Method not allowed.' });
  }

  let input;
  if (start) {
    let body;
    try {
      body = req.body;
    } catch {
      return res.status(400).json({ error: 'The composition request must contain valid JSON.' });
    }
    const prompt = typeof body?.prompt === 'string' ? body.prompt.trim() : '';
    if (!prompt || prompt.length > 1000) {
      return res.status(400).json({ error: 'Enter a prompt of 1–1,000 characters.' });
    }
    const history = body.history ?? [];
    if (!Array.isArray(history) || history.length > 6 || !history.every(validTurn)) {
      return res.status(400).json({ error: 'The previous compositions are too large or could not be read.' });
    }
    input = { prompt, history: history.map(({ prompt, code }) => ({ prompt, code })) };
  }
  const transport = getTransport();
  if (!transport) {
    return res.status(503).json({ error: 'Live composition is unavailable.' });
  }
  const { secret } = transport;
  const job = start ? null : readJob(req.query?.job, secret);
  if (!start && !job) {
    return res.status(400).json({ error: 'This composition link is invalid or has expired.' });
  }

  try {
    const response = await transport.request(req.method, job, input);
    if (response.status === 404 && !start) {
      return res.status(404).json({ error: 'The composition job has expired.' });
    }
    if (response.status === 504 && !start) {
      return res.status(504).json({ error: 'The composition timed out. Please try again.' });
    }
    if (!response.ok) throw new Error(`Modal returned HTTP ${response.status}`);
    if (!start && response.status === 202) return res.status(202).json({ status: 'pending' });

    const result = await response.json();
    if (start && typeof result.job_id === 'string' && modalJobId.test(result.job_id)) {
      return res.status(202).json({ job: issueJob(result.job_id, secret) });
    }
    if (!start && typeof result.completion === 'string') {
      return res.status(200).json({ completion: result.completion });
    }
    throw new Error('Unexpected Modal response');
  } catch (error) {
    console.error('TuneLM inference request failed', error.message);
    if (transport.local && error.localSetup) {
      return res.status(503).json({ error: error.message });
    }
    return res.status(502).json({ error: 'The music service could not complete this request.' });
  }
}

export default function compose(req, res) {
  return handleCompose(req, res, modalHttpTransport);
}
