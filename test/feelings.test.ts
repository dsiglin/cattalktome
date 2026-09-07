import { describe, it, expect } from 'vitest';
import { FEELINGS, readFeeling } from '../src/lib/feelings';
import type { ImageFeatures } from '../src/lib/features';

const base: ImageFeatures = {
  brightness: 0.5,
  contrast: 0.5,
  saturation: 0.4,
  warmth: 0.1,
  sharpness: 0.4,
  darkRatio: 0.2,
  brightRatio: 0.1,
  centreFocus: 0.5,
  aspect: 1,
};
const f = (over: Partial<ImageFeatures>): ImageFeatures => ({ ...base, ...over });

describe('the feeling catalogue', () => {
  it('holds at least eight feelings', () => {
    expect(FEELINGS.length).toBeGreaterThanOrEqual(8);
  });

  it('gives every feeling a unique id', () => {
    const ids = FEELINGS.map((x) => x.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('gives every feeling a sticker label, an emoji, a blurb and a real cue', () => {
    for (const feeling of FEELINGS) {
      expect(feeling.label.length, feeling.id).toBeGreaterThan(0);
      expect(feeling.label.length, `${feeling.id} label fits a sticker`).toBeLessThanOrEqual(24);
      expect(feeling.emoji.length, feeling.id).toBeGreaterThan(0);
      expect(feeling.blurb.length, feeling.id).toBeGreaterThan(10);
      expect(feeling.cue.length, feeling.id).toBeGreaterThan(10);
    }
  });
});

describe('readFeeling', () => {
  it('returns the same reading for the same features', () => {
    const a = readFeeling(f({}));
    const b = readFeeling(f({}));
    expect(a.feeling.id).toBe(b.feeling.id);
    expect(a.confidence).toBe(b.confidence);
  });

  it('returns a feeling that exists in the catalogue', () => {
    const reading = readFeeling(f({}));
    expect(FEELINGS.map((x) => x.id)).toContain(reading.feeling.id);
  });

  it('returns a runner-up that differs from the winner', () => {
    const reading = readFeeling(f({}));
    expect(reading.runnerUp.id).not.toBe(reading.feeling.id);
  });

  it('keeps confidence honest, between 0.3 and 0.9', () => {
    const reading = readFeeling(f({ brightness: 0.95, warmth: 0.6, contrast: 0.05, sharpness: 0.05 }));
    expect(reading.confidence).toBeGreaterThanOrEqual(0.3);
    expect(reading.confidence).toBeLessThanOrEqual(0.9);
  });

  it('gives at least two pieces of evidence', () => {
    const reading = readFeeling(f({}));
    expect(reading.evidence.length).toBeGreaterThanOrEqual(2);
    for (const line of reading.evidence) expect(line.length).toBeGreaterThan(5);
  });

  it('reads warm, bright, flat and soft light as sun-drunk', () => {
    const reading = readFeeling(
      f({ brightness: 0.85, warmth: 0.5, contrast: 0.12, sharpness: 0.1, brightRatio: 0.3, darkRatio: 0.02 }),
    );
    expect(reading.feeling.id).toBe('sun-drunk');
  });

  it('reads dim, flat and soft light as sleepy', () => {
    const reading = readFeeling(
      f({ brightness: 0.3, contrast: 0.15, sharpness: 0.06, brightRatio: 0.01, darkRatio: 0.35, warmth: 0.05 }),
    );
    expect(reading.feeling.id).toBe('sleepy');
  });

  it('reads a crisp, centred, high-contrast frame as locked on', () => {
    const reading = readFeeling(
      f({ sharpness: 0.85, centreFocus: 0.85, contrast: 0.8, brightness: 0.5, darkRatio: 0.25, brightRatio: 0.05 }),
    );
    expect(reading.feeling.id).toBe('locked-on');
  });

  it('reads a dark, cool, deep-shadow frame as wary', () => {
    const reading = readFeeling(
      f({ brightness: 0.2, contrast: 0.75, warmth: -0.5, darkRatio: 0.6, sharpness: 0.35, centreFocus: 0.4, brightRatio: 0.02 }),
    );
    expect(reading.feeling.id).toBe('wary');
  });

  it('can reach every feeling in the catalogue', () => {
    const seen = new Set<string>();
    const axis = [0, 0.25, 0.5, 0.75, 1];
    const warmths = [-0.6, -0.2, 0.2, 0.6];
    const aspects = [0.6, 1, 1.6];
    for (const brightness of axis)
      for (const contrast of axis)
        for (const sharpness of axis)
          for (const centreFocus of axis)
            for (const saturation of axis)
              for (const darkRatio of axis)
                for (const brightRatio of [0, 0.3, 0.6])
                  for (const warmth of warmths)
                    for (const aspect of aspects)
                      seen.add(
                        readFeeling({
                          brightness, contrast, sharpness, centreFocus,
                          saturation, darkRatio, brightRatio, warmth, aspect,
                        }).feeling.id,
                      );
    const missing = FEELINGS.map((x) => x.id).filter((id) => !seen.has(id));
    expect(missing, `unreachable feelings: ${missing.join(', ')}`).toEqual([]);
  });
});
