import { describe, it, expect } from 'vitest';
import { GENZ_STICKERS, pickRandomGenZSticker } from '../src/lib/genz-stickers';

describe('GENZ_STICKERS', () => {
  it('has a real catalogue of short, non-empty phrases', () => {
    expect(GENZ_STICKERS.length).toBeGreaterThanOrEqual(10);
    for (const phrase of GENZ_STICKERS) {
      expect(phrase.length).toBeGreaterThan(2);
      expect(phrase.length).toBeLessThanOrEqual(28);
    }
  });

  it('has no duplicate phrases', () => {
    expect(new Set(GENZ_STICKERS).size).toBe(GENZ_STICKERS.length);
  });
});

describe('pickRandomGenZSticker', () => {
  it('always returns a phrase from the catalogue', () => {
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
