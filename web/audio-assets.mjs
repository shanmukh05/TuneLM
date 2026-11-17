export async function fetchAudioAsset(url) {
  const response = await fetch(url, { signal: AbortSignal.timeout(15000) });
  if (!response.ok) throw new Error(`The sound library returned HTTP ${response.status}.`);
  return response;
}
