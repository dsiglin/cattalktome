import { describe, it, expect } from 'vitest';
import { layoutSticker, estimateTextWidth } from '../src/lib/sticker';

describe('layoutSticker', () => {
  it('scales the sticker with the short side of the photo', () => {
    const small = layoutSticker(400, 400, 'Sleepy 😴');
    const large = layoutSticker(1600, 1600, 'Sleepy 😴');
    expect(large.fontSize).toBeGreaterThan(small.fontSize * 3);
  });

  it('keeps the sticker inside the photo', () => {
    for (const [w, h] of [[800, 600], [600, 800], [1200, 1200], [1920, 480]]) {
      const l = layoutSticker(w, h, 'Plotting something 😼');
      expect(l.box.x, `${w}x${h} left`).toBeGreaterThanOrEqual(0);
      expect(l.box.y, `${w}x${h} top`).toBeGreaterThanOrEqual(0);
      expect(l.box.x + l.box.width, `${w}x${h} right`).toBeLessThanOrEqual(w);
      expect(l.box.y + l.box.height, `${w}x${h} bottom`).toBeLessThanOrEqual(h);
    }
  });

  it('makes a longer label a wider sticker', () => {
    const short = layoutSticker(1000, 1000, 'Wary 🫣');
    const long = layoutSticker(1000, 1000, 'Plotting something 😼');
    expect(long.box.width).toBeGreaterThan(short.box.width);
  });

  it('shrinks the font rather than overflowing on a very long label', () => {
    const normal = layoutSticker(1000, 1000, 'Wary 🫣');
    const huge = layoutSticker(1000, 1000, 'Absolutely furious about the vacuum cleaner right now');
    expect(huge.fontSize).toBeLessThan(normal.fontSize);
    expect(huge.box.width).toBeLessThanOrEqual(1000);
  });

  it('tilts the sticker a little, and always the same way for the same label', () => {
    const a = layoutSticker(1000, 1000, 'Curious 👀');
    const b = layoutSticker(1000, 1000, 'Curious 👀');
    const c = layoutSticker(1000, 1000, 'Sleepy 😴');
    expect(a.rotation).toBe(b.rotation);
    expect(Math.abs(a.rotation)).toBeGreaterThan(0);
    expect(Math.abs(a.rotation)).toBeLessThan(0.08);
    expect(a.rotation).not.toBe(c.rotation);
  });

  it('uses the caller measurer when one is given', () => {
    const wide = layoutSticker(1000, 1000, 'Wary 🫣', () => 900);
    const narrow = layoutSticker(1000, 1000, 'Wary 🫣', () => 100);
    expect(wide.box.width).toBeGreaterThan(narrow.box.width);
  });

  it('rejects an empty label', () => {
    expect(() => layoutSticker(1000, 1000, '   ')).toThrow();
  });

  it('rejects a photo with no size', () => {
    expect(() => layoutSticker(0, 500, 'Wary')).toThrow();
  });
});

describe('estimateTextWidth', () => {
  it('grows with the label length and the font size', () => {
    expect(estimateTextWidth('aaaa', 20)).toBeGreaterThan(estimateTextWidth('aa', 20));
    expect(estimateTextWidth('aa', 40)).toBeGreaterThan(estimateTextWidth('aa', 20));
  });

  it('counts an emoji as wider than a letter', () => {
    expect(estimateTextWidth('😴', 20)).toBeGreaterThan(estimateTextWidth('a', 20));
  });
});
