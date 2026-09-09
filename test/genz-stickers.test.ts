import { describe, it, expect } from 'vitest';
import {
  GENZ_STICKERS,
  STICKERS_BY_FEELING,
  pickRandomGenZSticker,
  pickStickerFor,
  stickersFor,
} from '../src/lib/genz-stickers';

// Must match the ids in server/app/feelings.py.
const FEELING_IDS = ['curious', 'focused', 'frightened', 'cautious', 'irritated', 'trusting', 'unimpressed'];

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

describe('STICKERS_BY_FEELING', () => {
  it('has a pool for every feeling the server can return', () => {
    for (const id of FEELING_IDS) {
      expect(STICKERS_BY_FEELING[id], id).toBeDefined();
    }
  });

  it('every pool has at least two stickers so the pick still varies', () => {
    for (const [id, pool] of Object.entries(STICKERS_BY_FEELING)) {
      expect(pool.length, id).toBeGreaterThanOrEqual(2);
    }
  });

  it('every pool names only real catalogue files', () => {
    const files = new Set(GENZ_STICKERS.map((s) => s.file));
    for (const pool of Object.values(STICKERS_BY_FEELING)) {
      for (const file of pool) expect(files.has(file), file).toBe(true);
    }
  });

  it('uses every sticker in the catalogue somewhere', () => {
    const used = new Set(Object.values(STICKERS_BY_FEELING).flat());
    for (const sticker of GENZ_STICKERS) expect(used.has(sticker.file), sticker.file).toBe(true);
  });

  it('never puts an angry or screaming cat on a calm reading', () => {
    for (const id of ['trusting', 'curious']) {
      expect(STICKERS_BY_FEELING[id]).not.toContain('angry.png');
      expect(STICKERS_BY_FEELING[id]).not.toContain('screaming.png');
    }
  });

  it('never puts a hearts-and-smiles cat on an angry or frightened reading', () => {
    for (const id of ['irritated', 'frightened']) {
      expect(STICKERS_BY_FEELING[id]).not.toContain('love.png');
      expect(STICKERS_BY_FEELING[id]).not.toContain('smile.png');
      expect(STICKERS_BY_FEELING[id]).not.toContain('celebration.png');
    }
  });
});

describe('pickStickerFor', () => {
  it('only ever picks from the feeling pool', () => {
    for (const id of FEELING_IDS) {
      const allowed = new Set(STICKERS_BY_FEELING[id]);
      for (let i = 0; i < 20; i++) {
        expect(allowed.has(pickStickerFor(id).file), id).toBe(true);
      }
    }
  });

  it('is deterministic given an rng', () => {
    expect(pickStickerFor('trusting', () => 0).file).toBe(STICKERS_BY_FEELING.trusting[0]);
    const last = STICKERS_BY_FEELING.trusting.length - 1;
    expect(pickStickerFor('trusting', () => 0.9999).file).toBe(STICKERS_BY_FEELING.trusting[last]);
  });

  it('falls back to the whole catalogue for an unknown feeling id', () => {
    expect(stickersFor('not-a-feeling')).toBe(GENZ_STICKERS);
    expect(GENZ_STICKERS).toContain(pickStickerFor('not-a-feeling'));
  });
});

describe('pickRandomGenZSticker', () => {
  it('always returns a sticker from the catalogue', () => {
    for (let i = 0; i < 20; i++) {
      expect(GENZ_STICKERS).toContain(pickRandomGenZSticker());
    }
  });

  it('picks the first and last entries at the rng extremes', () => {
    expect(pickRandomGenZSticker(() => 0)).toBe(GENZ_STICKERS[0]);
    expect(pickRandomGenZSticker(() => 0.9999)).toBe(GENZ_STICKERS[GENZ_STICKERS.length - 1]);
  });
});
