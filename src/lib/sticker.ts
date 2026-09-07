export interface StickerLayout {
  fontSize: number;
  padX: number;
  padY: number;
  box: { x: number; y: number; width: number; height: number };
  rotation: number;
  radius: number;
}

/** Measure a label's width in pixels at a given font size. */
export type Measurer = (text: string, fontSize: number) => number;

const LETTER = 0.56;
const SPACE = 0.3;
const EMOJI = 1.15;

/** Estimate a bold sans-serif label width without a canvas. */
export function estimateTextWidth(text: string, fontSize: number): number {
  let units = 0;
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    if (ch === ' ') units += SPACE;
    else if (cp > 0x2000) units += EMOJI;
    else units += LETTER;
  }
  return units * fontSize;
}

/** A small, stable tilt derived from the label. The same label always tilts the same way. */
function tiltFor(text: string): number {
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  const sign = hash & 1 ? 1 : -1;
  const magnitude = 0.018 + ((hash >>> 1) % 1000) / 1000 * 0.04;
  return sign * magnitude;
}

export function layoutSticker(
  imageWidth: number,
  imageHeight: number,
  text: string,
  measure: Measurer = estimateTextWidth,
): StickerLayout {
  if (imageWidth <= 0 || imageHeight <= 0) throw new Error('layoutSticker needs a sized photo');
  const label = text.trim();
  if (label.length === 0) throw new Error('layoutSticker needs a label');

  const short = Math.min(imageWidth, imageHeight);
  const maxBoxWidth = imageWidth * 0.86;
  const minFontSize = short * 0.022;

  let fontSize = short * 0.085;
  let padX = fontSize * 0.62;
  // Shrink the font until the sticker fits, or until it reaches the floor.
  while (measure(label, fontSize) + 2 * padX > maxBoxWidth && fontSize > minFontSize) {
    fontSize *= 0.95;
    padX = fontSize * 0.62;
  }
  const padY = fontSize * 0.42;

  const width = Math.min(maxBoxWidth, measure(label, fontSize) + 2 * padX);
  const height = fontSize + 2 * padY;

  const margin = short * 0.06;
  const x = Math.max(0, Math.min(imageWidth - width, (imageWidth - width) / 2));
  const y = Math.max(0, Math.min(imageHeight - height, imageHeight - margin - height));

  return {
    fontSize,
    padX,
    padY,
    box: { x, y, width, height },
    rotation: tiltFor(label),
    radius: height * 0.36,
  };
}

export interface StickerTheme {
  fill: string;
  text: string;
  edge: string;
}

export const COZY_STICKER: StickerTheme = {
  fill: '#FFF6E9',
  text: '#43301F',
  edge: '#E8A33D',
};

/** Draw the sticker onto a canvas context. The caller draws the photo first. */
export function drawSticker(
  ctx: CanvasRenderingContext2D,
  layout: StickerLayout,
  label: string,
  theme: StickerTheme = COZY_STICKER,
): void {
  const { box, fontSize, rotation, radius } = layout;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(rotation);

  // A soft drop shadow lifts the sticker off the photo.
  ctx.shadowColor = 'rgba(58, 36, 18, 0.35)';
  ctx.shadowBlur = fontSize * 0.5;
  ctx.shadowOffsetY = fontSize * 0.16;

  ctx.beginPath();
  ctx.roundRect(-box.width / 2, -box.height / 2, box.width, box.height, radius);
  ctx.fillStyle = theme.fill;
  ctx.fill();

  ctx.shadowColor = 'transparent';
  ctx.lineWidth = Math.max(2, fontSize * 0.09);
  ctx.strokeStyle = theme.edge;
  ctx.stroke();

  ctx.fillStyle = theme.text;
  ctx.font = `700 ${fontSize}px "Nunito", "Quicksand", system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, 0, fontSize * 0.04, box.width - 2 * layout.padX);

  ctx.restore();
}
