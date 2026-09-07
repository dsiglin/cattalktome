import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { decode } from 'jpeg-js';
import { analysePhoto, downscale } from '../src/lib/analyse';
import { extractFeatures } from '../src/lib/features';
import { FEELINGS } from '../src/lib/feelings';
import { layoutSticker } from '../src/lib/sticker';
import { makeImage } from './helpers';

const realCat = () => {
  const raw = readFileSync(new URL('./fixtures/cat.jpg', import.meta.url));
  const { width, height, data } = decode(raw, { useTArray: true, formatAsRGBA: true });
  return { width, height, data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.length) };
};

describe('downscale', () => {
  it('caps the long side and keeps the aspect ratio', () => {
    const img = makeImage(800, 400, () => [10, 20, 30]);
    const small = downscale(img, 200);
    expect(Math.max(small.width, small.height)).toBe(200);
    expect(small.width / small.height).toBeCloseTo(2, 1);
  });

  it('leaves an already small photo alone', () => {
    const img = makeImage(100, 50, () => [10, 20, 30]);
    const same = downscale(img, 200);
    expect(same.width).toBe(100);
    expect(same.height).toBe(50);
  });

  it('keeps the colours it samples', () => {
    const img = makeImage(400, 400, () => [200, 100, 50]);
    const small = downscale(img, 64);
    expect(small.data[0]).toBe(200);
    expect(small.data[1]).toBe(100);
    expect(small.data[2]).toBe(50);
    expect(small.data[3]).toBe(255);
  });
});

describe('the whole pipeline, on a real cat photograph', () => {
  const cat = realCat();

  it('decodes the fixture', () => {
    expect(cat.width).toBeGreaterThan(200);
    expect(cat.height).toBeGreaterThan(200);
  });

  it('produces a reading from the catalogue', () => {
    const reading = analysePhoto(cat);
    // Print the real result so a human can sanity-check it.
    console.log('\n  Real cat photo reads as:', reading.feeling.emoji, reading.feeling.label,
      `(confidence ${reading.confidence})`);
    console.log('  Runner-up:', reading.runnerUp.label);
    console.log('  Evidence:', reading.evidence.join(' '), '\n');
    expect(FEELINGS.map((x) => x.id)).toContain(reading.feeling.id);
    expect(reading.confidence).toBeGreaterThanOrEqual(0.3);
    expect(reading.confidence).toBeLessThanOrEqual(0.9);
    expect(reading.evidence.length).toBeGreaterThanOrEqual(2);
  });

  it('reads the same photo the same way every time', () => {
    const a = analysePhoto(cat);
    const b = analysePhoto(cat);
    expect(a.feeling.id).toBe(b.feeling.id);
    expect(a.confidence).toBe(b.confidence);
  });

  it('measures sane features from a real photograph', () => {
    const f = extractFeatures(downscale(cat, 160));
    expect(f.brightness).toBeGreaterThan(0.03);
    expect(f.brightness).toBeLessThan(0.97);
    expect(f.contrast).toBeGreaterThan(0);
    expect(f.sharpness).toBeGreaterThan(0);
    expect(f.sharpness).toBeLessThan(0.9);
    expect(f.aspect).toBeCloseTo(cat.width / cat.height, 1);
  });

  it('lays a readable sticker onto the real photo size', () => {
    const reading = analysePhoto(cat);
    const label = `${reading.feeling.label} ${reading.feeling.emoji}`;
    const l = layoutSticker(cat.width, cat.height, label);
    expect(l.fontSize).toBeGreaterThan(12);
    expect(l.box.x + l.box.width).toBeLessThanOrEqual(cat.width);
    expect(l.box.y + l.box.height).toBeLessThanOrEqual(cat.height);
  });

  it('reads a different photo differently', () => {
    const flat = makeImage(300, 300, () => [128, 128, 128]);
    expect(analysePhoto(flat).feeling.id).not.toBe(analysePhoto(cat).feeling.id);
  });
});
