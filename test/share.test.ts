import { describe, it, expect, vi } from 'vitest';
import { shareSticker, supportsFileShare } from '../src/lib/share';

const blob = () => new Blob(['fake-png-bytes'], { type: 'image/png' });

const fakeNav = (over: Record<string, unknown> = {}) => ({
  canShare: () => true,
  share: vi.fn(async () => undefined),
  ...over,
}) as unknown as Navigator;

describe('supportsFileShare', () => {
  it('is true when the browser can share files', () => {
    expect(supportsFileShare(fakeNav())).toBe(true);
  });

  it('is false when canShare is missing', () => {
    expect(supportsFileShare({ share: () => {} } as unknown as Navigator)).toBe(false);
  });

  it('is false when share is missing', () => {
    expect(supportsFileShare({ canShare: () => true } as unknown as Navigator)).toBe(false);
  });

  it('is false when the browser refuses file payloads', () => {
    expect(supportsFileShare(fakeNav({ canShare: () => false }))).toBe(false);
  });
});

describe('shareSticker', () => {
  it('opens the native sharesheet when the browser allows it', async () => {
    const nav = fakeNav();
    const save = vi.fn();
    const result = await shareSticker({
      blob: blob(), filename: 'cat.png', title: 'Sleepy', text: 'My cat is sleepy.',
      navigator: nav, saveFallback: save,
    });
    expect(result).toBe('shared');
    expect(save).not.toHaveBeenCalled();
    const payload = (nav.share as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(payload.files).toHaveLength(1);
    expect(payload.files[0].name).toBe('cat.png');
    expect(payload.files[0].type).toBe('image/png');
    expect(payload.title).toBe('Sleepy');
    expect(payload.text).toBe('My cat is sleepy.');
  });

  it('saves the image when the browser cannot share files', async () => {
    const save = vi.fn();
    const result = await shareSticker({
      blob: blob(), filename: 'cat.png', title: 'T', text: 'X',
      navigator: fakeNav({ canShare: () => false }), saveFallback: save,
    });
    expect(result).toBe('saved');
    expect(save).toHaveBeenCalledOnce();
    expect(save.mock.calls[0][1]).toBe('cat.png');
  });

  it('reports a cancel when the user closes the sharesheet', async () => {
    const abort = Object.assign(new Error('cancelled'), { name: 'AbortError' });
    const save = vi.fn();
    const result = await shareSticker({
      blob: blob(), filename: 'cat.png', title: 'T', text: 'X',
      navigator: fakeNav({ share: vi.fn(async () => { throw abort; }) }), saveFallback: save,
    });
    expect(result).toBe('cancelled');
    expect(save).not.toHaveBeenCalled();
  });

  it('saves the image when the sharesheet fails for any other reason', async () => {
    const save = vi.fn();
    const result = await shareSticker({
      blob: blob(), filename: 'cat.png', title: 'T', text: 'X',
      navigator: fakeNav({ share: vi.fn(async () => { throw new Error('no transport'); }) }),
      saveFallback: save,
    });
    expect(result).toBe('saved');
    expect(save).toHaveBeenCalledOnce();
  });
});
