import { extractFeatures, type RgbaImage, type ImageFeatures } from './features';
import { readFeeling, type Reading } from './feelings';

/** The working size for analysis. Small enough to stay fast, large enough to keep detail. */
export const ANALYSIS_SIZE = 160;

/**
 * Shrink a photo by nearest-neighbour sampling.
 *
 * Nearest neighbour keeps the original pixel values, so colour measurements
 * stay true. Averaging would soften edges and lower the sharpness reading.
 */
export function downscale(image: RgbaImage, maxSide: number): RgbaImage {
  const { width, height, data } = image;
  const longest = Math.max(width, height);
  if (longest <= maxSide) return image;

  const scale = maxSide / longest;
  const outWidth = Math.max(1, Math.round(width * scale));
  const outHeight = Math.max(1, Math.round(height * scale));
  const out = new Uint8ClampedArray(outWidth * outHeight * 4);

  for (let y = 0; y < outHeight; y++) {
    const sourceY = Math.min(height - 1, Math.floor((y * height) / outHeight));
    for (let x = 0; x < outWidth; x++) {
      const sourceX = Math.min(width - 1, Math.floor((x * width) / outWidth));
      const from = (sourceY * width + sourceX) * 4;
      const to = (y * outWidth + x) * 4;
      out[to] = data[from];
      out[to + 1] = data[from + 1];
      out[to + 2] = data[from + 2];
      out[to + 3] = data[from + 3];
    }
  }
  return { width: outWidth, height: outHeight, data: out };
}

export interface Analysis extends Reading {
  features: ImageFeatures;
}

/** Measure a photo, then read a feeling from the measurements. */
export function analysePhoto(image: RgbaImage, size = ANALYSIS_SIZE): Analysis {
  const features = extractFeatures(downscale(image, size));
  return { ...readFeeling(features), features };
}
