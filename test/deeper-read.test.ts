import { describe, it, expect, vi } from 'vitest';
import { requestDeeperRead } from '../src/lib/deeper-read';

const okBody = {
  feeling: { id: 'wary', label: 'Wary', emoji: '🫣', blurb: 'blurb', cue: 'cue' },
  runner_up: { id: 'startled', label: 'Startled' },
  confidence: 0.6,
  reliable: true,
  evidence: ['one', 'two'],
  eyes: { pupil_dilation: 0.42, usable: true },
};

const fakeFetch = (status: number, body: unknown) =>
  vi.fn(async () => ({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  })) as unknown as typeof fetch;

const blob = () => new Blob(['fake'], { type: 'image/png' });

describe('requestDeeperRead', () => {
  it('returns the reading on success', async () => {
    const result = await requestDeeperRead(blob(), 'https://api.example', fakeFetch(200, okBody));
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.reading.feeling.id).toBe('wary');
      expect(result.reading.runnerUpLabel).toBe('Startled');
      expect(result.reading.confidence).toBe(0.6);
      expect(result.reading.reliable).toBe(true);
      expect(result.reading.evidence).toEqual(['one', 'two']);
      expect(result.reading.eyes).toEqual({ pupilDilation: 0.42, usable: true });
    }
  });

  it('defaults eyes to unusable when the server omits them', async () => {
    const { eyes, ...bodyWithoutEyes } = okBody;
    const result = await requestDeeperRead(blob(), 'https://api.example', fakeFetch(200, bodyWithoutEyes));
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.reading.eyes).toEqual({ pupilDilation: 0.5, usable: false });
  });

  it('posts the photo as multipart form data to /analyze', async () => {
    const fetchImpl = fakeFetch(200, okBody);
    await requestDeeperRead(blob(), 'https://api.example', fetchImpl);
    const [url, init] = (fetchImpl as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(url).toBe('https://api.example/analyze');
    expect(init.method).toBe('POST');
    expect(init.body).toBeInstanceOf(FormData);
  });

  it('reports no-cat-found on 422', async () => {
    const result = await requestDeeperRead(blob(), 'https://api.example', fakeFetch(422, { detail: 'no cat face found in this photo' }));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toBe('no-cat');
  });

  it('reports rate-limited on 429 and carries the server message', async () => {
    const result = await requestDeeperRead(
      blob(), 'https://api.example',
      fakeFetch(429, { detail: 'You have hit the per-visitor limit for this hour - please slow down.' }),
    );
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toBe('rate-limited');
      expect(result.message).toContain('per-visitor limit');
    }
  });

  it('reports too-large on 413', async () => {
    const result = await requestDeeperRead(blob(), 'https://api.example', fakeFetch(413, { detail: 'image too large' }));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toBe('too-large');
  });

  it('reports bad-image on 400', async () => {
    const result = await requestDeeperRead(blob(), 'https://api.example', fakeFetch(400, { detail: 'upload must be an image' }));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toBe('bad-image');
  });

  it('reports unknown on an unexpected status', async () => {
    const result = await requestDeeperRead(blob(), 'https://api.example', fakeFetch(500, {}));
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toBe('unknown');
  });

  it('reports network on a fetch rejection', async () => {
    const failing = vi.fn(async () => { throw new TypeError('Failed to fetch'); }) as unknown as typeof fetch;
    const result = await requestDeeperRead(blob(), 'https://api.example', failing);
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason).toBe('network');
  });
});
