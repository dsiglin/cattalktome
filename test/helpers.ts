import type { RgbaImage } from '../src/lib/features';

/** Build a synthetic image. `fn` returns [r,g,b,a] for each pixel. */
export function makeImage(
  width: number,
  height: number,
  fn: (x: number, y: number) => [number, number, number, number?],
): RgbaImage {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b, a] = fn(x, y);
      const i = (y * width + x) * 4;
      data[i] = r;
      data[i + 1] = g;
      data[i + 2] = b;
      data[i + 3] = a ?? 255;
    }
  }
  return { width, height, data };
}

export const solid = (w: number, h: number, r: number, g: number, b: number): RgbaImage =>
  makeImage(w, h, () => [r, g, b]);
