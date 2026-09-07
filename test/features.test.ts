import { describe, it, expect } from 'vitest';
import { extractFeatures } from '../src/lib/features';
import { makeImage, solid } from './helpers';

describe('extractFeatures', () => {
  it('reads an all-black image as dark, flat and unlit', () => {
    const f = extractFeatures(solid(32, 32, 0, 0, 0));
    expect(f.brightness).toBeCloseTo(0, 3);
    expect(f.contrast).toBeCloseTo(0, 3);
    expect(f.darkRatio).toBeCloseTo(1, 3);
    expect(f.brightRatio).toBeCloseTo(0, 3);
    expect(f.sharpness).toBeCloseTo(0, 3);
  });

  it('reads an all-white image as bright and fully lit', () => {
    const f = extractFeatures(solid(32, 32, 255, 255, 255));
    expect(f.brightness).toBeCloseTo(1, 3);
    expect(f.brightRatio).toBeCloseTo(1, 3);
    expect(f.darkRatio).toBeCloseTo(0, 3);
  });

  it('reads a half-black half-white image as mid-bright and high contrast', () => {
    const img = makeImage(32, 32, (_x, y) => (y < 16 ? [0, 0, 0] : [255, 255, 255]));
    const f = extractFeatures(img);
    expect(f.brightness).toBeCloseTo(0.5, 2);
    expect(f.contrast).toBeGreaterThan(0.9);
    expect(f.darkRatio).toBeCloseTo(0.5, 2);
    expect(f.brightRatio).toBeCloseTo(0.5, 2);
  });

  it('reads red as warm and blue as cool', () => {
    expect(extractFeatures(solid(16, 16, 255, 0, 0)).warmth).toBeGreaterThan(0.9);
    expect(extractFeatures(solid(16, 16, 0, 0, 255)).warmth).toBeLessThan(-0.9);
    expect(extractFeatures(solid(16, 16, 128, 128, 128)).warmth).toBeCloseTo(0, 2);
  });

  it('reads grey as unsaturated and pure red as fully saturated', () => {
    expect(extractFeatures(solid(16, 16, 128, 128, 128)).saturation).toBeCloseTo(0, 3);
    expect(extractFeatures(solid(16, 16, 255, 0, 0)).saturation).toBeCloseTo(1, 3);
  });

  it('reads a checkerboard as sharper than a flat field', () => {
    const check = makeImage(32, 32, (x, y) => ((x + y) % 2 === 0 ? [255, 255, 255] : [0, 0, 0]));
    const flat = solid(32, 32, 128, 128, 128);
    expect(extractFeatures(check).sharpness).toBeGreaterThan(0.5);
    expect(extractFeatures(flat).sharpness).toBeCloseTo(0, 3);
  });

  it('reads detail in the centre as centre focus', () => {
    const centre = makeImage(60, 60, (x, y) => {
      const inMiddle = x >= 20 && x < 40 && y >= 20 && y < 40;
      return inMiddle && (x + y) % 2 === 0 ? [255, 255, 255] : [0, 0, 0];
    });
    const edges = makeImage(60, 60, (x, y) => {
      const inMiddle = x >= 20 && x < 40 && y >= 20 && y < 40;
      return !inMiddle && (x + y) % 2 === 0 ? [255, 255, 255] : [0, 0, 0];
    });
    expect(extractFeatures(centre).centreFocus).toBeGreaterThan(extractFeatures(edges).centreFocus);
  });

  it('reports the aspect ratio of the frame', () => {
    expect(extractFeatures(solid(40, 20, 10, 10, 10)).aspect).toBeCloseTo(2, 3);
    expect(extractFeatures(solid(20, 40, 10, 10, 10)).aspect).toBeCloseTo(0.5, 3);
  });

  it('keeps every feature finite and inside its declared range', () => {
    let seed = 7;
    const rand = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
    const noise = makeImage(48, 48, () => [rand() * 255, rand() * 255, rand() * 255]);
    const f = extractFeatures(noise);
    for (const [key, value] of Object.entries(f)) {
      expect(Number.isFinite(value), `${key} is finite`).toBe(true);
    }
    for (const key of ['brightness', 'contrast', 'saturation', 'sharpness', 'darkRatio', 'brightRatio', 'centreFocus'] as const) {
      expect(f[key], key).toBeGreaterThanOrEqual(0);
      expect(f[key], key).toBeLessThanOrEqual(1);
    }
    expect(f.warmth).toBeGreaterThanOrEqual(-1);
    expect(f.warmth).toBeLessThanOrEqual(1);
  });

  it('rejects an empty image', () => {
    expect(() => extractFeatures({ width: 0, height: 0, data: new Uint8ClampedArray(0) })).toThrow();
  });
});
