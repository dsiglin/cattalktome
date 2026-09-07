/**
 * Measurable properties of a photo.
 *
 * These are honest, reproducible measurements of pixels. They are not a
 * diagnosis. `feelings.ts` maps them onto documented feline cue families.
 */
export interface RgbaImage {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

export interface ImageFeatures {
  /** Mean luminance, 0 = black, 1 = white. Stands in for scene light. */
  brightness: number;
  /** Spread of luminance, 0 = flat, 1 = hard light. Stands in for shadow depth. */
  contrast: number;
  /** Mean colour purity, 0 = grey, 1 = vivid. */
  saturation: number;
  /** Red minus blue, -1 = cool, 1 = warm. Stands in for sunlight against shade. */
  warmth: number;
  /** Edge energy, 0 = soft or blurred, 1 = crisp. Stands in for stillness against motion. */
  sharpness: number;
  /** Share of very dark pixels. Stands in for wide pupils and deep shade. */
  darkRatio: number;
  /** Share of very bright pixels. Stands in for eye shine and direct sun. */
  brightRatio: number;
  /** Detail in the middle third against detail overall. Stands in for a face that fills the frame. */
  centreFocus: number;
  /** Frame width divided by frame height. Stands in for a sprawl against a sit. */
  aspect: number;
}

const clamp01 = (n: number) => Math.min(1, Math.max(0, n));

/** Rec. 709 luminance, scaled to 0..1. */
const luma = (r: number, g: number, b: number) =>
  (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;

/** Mean absolute Laplacian over the interior of a luminance plane. */
function edgeEnergy(
  plane: Float32Array,
  width: number,
  height: number,
  x0 = 0,
  y0 = 0,
  x1 = width,
  y1 = height,
): number {
  const left = Math.max(1, x0);
  const top = Math.max(1, y0);
  const right = Math.min(width - 1, x1);
  const bottom = Math.min(height - 1, y1);
  let total = 0;
  let count = 0;
  for (let y = top; y < bottom; y++) {
    for (let x = left; x < right; x++) {
      const i = y * width + x;
      const lap =
        4 * plane[i] - plane[i - 1] - plane[i + 1] - plane[i - width] - plane[i + width];
      total += Math.abs(lap);
      count++;
    }
  }
  return count === 0 ? 0 : total / count;
}

export function extractFeatures(image: RgbaImage): ImageFeatures {
  const { width, height, data } = image;
  if (width <= 0 || height <= 0 || data.length < width * height * 4) {
    throw new Error('extractFeatures needs a non-empty RGBA image');
  }

  const pixels = width * height;
  const plane = new Float32Array(pixels);
  let sumLuma = 0;
  let sumSquares = 0;
  let sumSaturation = 0;
  let sumWarmth = 0;
  let dark = 0;
  let bright = 0;

  for (let p = 0; p < pixels; p++) {
    const i = p * 4;
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const l = luma(r, g, b);
    plane[p] = l;
    sumLuma += l;
    sumSquares += l * l;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    sumSaturation += max === 0 ? 0 : (max - min) / max;
    sumWarmth += (r - b) / 255;

    if (l < 0.2) dark++;
    if (l > 0.85) bright++;
  }

  const brightness = sumLuma / pixels;
  const variance = Math.max(0, sumSquares / pixels - brightness * brightness);

  // A maximum-contrast image (half black, half white) has a standard
  // deviation of 0.5, so doubling maps the real range onto 0..1.
  const contrast = clamp01(Math.sqrt(variance) * 2);

  // A checkerboard produces a mean Laplacian of 4. Dividing by 4 normalises
  // it. Real photographs sit far lower, so a gain of 3 spreads them usefully.
  const rawSharpness = edgeEnergy(plane, width, height);
  const sharpness = clamp01((rawSharpness / 4) * 3);

  const thirdX = Math.floor(width / 3);
  const thirdY = Math.floor(height / 3);
  const centreEnergy = edgeEnergy(plane, width, height, thirdX, thirdY, width - thirdX, height - thirdY);
  const focusRatio = rawSharpness === 0 ? 0 : centreEnergy / rawSharpness;
  const centreFocus = clamp01(focusRatio / (focusRatio + 1));

  return {
    brightness: clamp01(brightness),
    contrast,
    saturation: clamp01(sumSaturation / pixels),
    warmth: Math.min(1, Math.max(-1, sumWarmth / pixels)),
    sharpness,
    darkRatio: dark / pixels,
    brightRatio: bright / pixels,
    centreFocus,
    aspect: width / height,
  };
}
