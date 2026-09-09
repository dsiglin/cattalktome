import { describe, it, expect } from 'vitest';
import { GENZ_STICKERS, pickRandomGenZSticker } from '../src/lib/genz-stickers';

describe('GENZ_STICKERS', () => {
  it('has a real catalogue of image stickers', () => {
    expect(GENZ_STICKERS.length).toBeGreaterThanOrEqual(10);
    for (const sticker of GENZ_STICKERS) {
      expect(sticker.file.length).toBeGreaterThan(0);
      expect(sticker.file.endsWith('.png')).toBe(true);
      expect(sticker.alt.length).toBeGreaterThan(0);
    }
  });

  it('has no duplicate files', () => {
    const files = GENZ_STICKERS.map((s) => s.file);
    expect(new Set(files).size).toBe(files.length);
  });
});

describe('pickRandomGenZSticker', () => {
  it('always returns a sticker from the catalogue', () => {
    for (let i = 0; i < 20; i++) {
      const pick = pickRandomGenZSticker();
      expect(GENZ_STICKERS).toContain(pick);
    }
  });

  it('picks the first entry when the injected rng returns 0', () => {
    expect(pickRandomGenZSticker(() => 0)).toBe(GENZ_STICKERS[0]);
  });

  it('picks the last entry when the injected rng returns just under 1', () => {
    expect(pickRandomGenZSticker(() => 0.9999)).toBe(GENZ_STICKERS[GENZ_STICKERS.length - 1]);
  });

  it('varies its pick as the injected rng varies', () => {
    const picks = new Set(
      Array.from({ length: GENZ_STICKERS.length }, (_, i) => pickRandomGenZSticker(() => i / GENZ_STICKERS.length)),
    );
    expect(picks.size).toBeGreaterThan(1);
  });
});
