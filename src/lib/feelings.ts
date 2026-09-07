import type { ImageFeatures } from './features';

/**
 * A candidate feline state.
 *
 * Each entry names the real cue family it stands for. The app cannot see ears
 * or pupils directly. It sees light, shadow, colour and detail, and those
 * follow the cue families closely enough to make a fair guess.
 */
export interface Feeling {
  id: string;
  label: string;
  emoji: string;
  blurb: string;
  cue: string;
  score: (f: ImageFeatures) => number;
}

export interface Reading {
  feeling: Feeling;
  runnerUp: Feeling;
  confidence: number;
  evidence: string[];
}

const clamp = (n: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, n));
const clamp01 = (n: number) => clamp(n, 0, 1);

/** A soft preference for a value near a target. Returns 1 at the target. */
const near = (value: number, target: number, tolerance: number) =>
  Math.exp(-(((value - target) / tolerance) ** 2));

/** Map warmth from -1..1 onto 0..1. */
const warm = (f: ImageFeatures) => (f.warmth + 1) / 2;

/** How tall the frame is. A tall frame suits a sitting or loafing cat. */
const tall = (f: ImageFeatures) => clamp01((1.2 - f.aspect) / 0.7);

/** How wide the frame is. A wide frame suits a stretched or moving cat. */
const wide = (f: ImageFeatures) => clamp01((f.aspect - 0.9) / 0.7);

/** Combine weighted terms into a single 0..1 score. */
const blend = (...terms: [number, number][]) => {
  let total = 0;
  let weight = 0;
  for (const [w, value] of terms) {
    total += w * clamp01(value);
    weight += w;
  }
  return weight === 0 ? 0 : total / weight;
};

export const FEELINGS: Feeling[] = [
  {
    id: 'sun-drunk',
    label: 'Sun-drunk',
    emoji: '🌞',
    blurb: 'Warm, soft light and no tension anywhere. This is a cat melting into a sunbeam.',
    cue: 'Relaxed cats in warm, even light hold their ears neutral and their pupils narrow.',
    score: (f) => blend([3, f.brightness], [2, warm(f)], [2, 1 - f.contrast], [2, 1 - f.sharpness], [1, f.brightRatio]),
  },
  {
    id: 'sleepy',
    label: 'Sleepy',
    emoji: '😴',
    blurb: 'Low light, soft edges, nothing urgent. This is the slow-blink end of the day.',
    cue: 'A drowsy cat softens its edges, half-closes its eyes and stops tracking the room.',
    score: (f) => blend([3, 1 - f.sharpness], [2, 1 - f.contrast], [2, 1 - f.brightRatio], [1.5, near(f.brightness, 0.32, 0.35)]),
  },
  {
    id: 'locked-on',
    label: 'Locked on',
    emoji: '🎯',
    blurb: 'Sharp detail gathered in the middle of the frame. Something has this cat’s full attention.',
    cue: 'A hunting cat fixes its head, points its whiskers forward and stops moving.',
    score: (f) => blend([3, f.sharpness], [3, f.centreFocus], [2, f.contrast]),
  },
  {
    id: 'curious',
    label: 'Curious',
    emoji: '👀',
    blurb: 'Bright, crisp and looking outward. This cat wants to know what that was.',
    cue: 'Curiosity pushes the ears and whiskers forward and opens the eyes wide.',
    score: (f) => blend([3, f.sharpness], [2, f.brightness], [2, 1 - f.centreFocus], [1, warm(f)], [2, 1 - f.darkRatio]),
  },
  {
    id: 'startled',
    label: 'Startled',
    emoji: '😳',
    blurb: 'Hard light, deep shadow and sharp edges together. Something just happened.',
    cue: 'A startled cat dilates its pupils fully and flattens its ears in one motion.',
    score: (f) => blend([2.5, f.darkRatio], [2.5, f.contrast], [2, f.sharpness], [2, f.brightRatio]),
  },
  {
    id: 'wary',
    label: 'Wary',
    emoji: '🫣',
    blurb: 'Dim, cool and full of shadow. This cat is keeping an exit in view.',
    cue: 'A wary cat holds still in shade, turns its ears sideways and watches the room.',
    score: (f) => blend([3, 1 - f.brightness], [2.5, f.contrast], [2, 1 - warm(f)], [1.5, f.darkRatio]),
  },
  {
    id: 'demanding',
    label: 'Demanding',
    emoji: '🍽️',
    blurb: 'Filling the middle of the frame in strong colour. This is a request, not a pose.',
    cue: 'A cat asking for something walks straight at you and holds eye contact.',
    score: (f) => blend([3, f.centreFocus], [2.5, f.saturation], [2, near(f.brightness, 0.6, 0.35)], [1, f.sharpness]),
  },
  {
    id: 'content-loaf',
    label: 'Fully loafed',
    emoji: '🍞',
    blurb: 'Soft, even and settled into a tall little shape. Paws tucked, nothing owed to anyone.',
    cue: 'A cat that tucks its paws under itself feels safe enough to stop being ready.',
    score: (f) => blend([3, 1 - f.contrast], [2, 1 - f.sharpness], [2.5, tall(f)], [1, near(f.brightness, 0.55, 0.4)]),
  },
  {
    id: 'unimpressed',
    label: 'Unimpressed',
    emoji: '😑',
    blurb: 'Muted colour, middling everything. This cat has considered you and moved on.',
    cue: 'A neutral cat holds its ears upright and its face still. That stillness is the message.',
    score: (f) => blend([3, 1 - f.saturation], [2, near(f.contrast, 0.5, 0.35)], [2, near(f.sharpness, 0.45, 0.35)], [1.5, near(f.brightness, 0.5, 0.3)]),
  },
  {
    id: 'mischief',
    label: 'Plotting something',
    emoji: '😼',
    blurb: 'Crisp, warm and stretched wide across the frame. The decision is already made.',
    cue: 'Before a pounce a cat lowers its body, widens its stance and locks its gaze.',
    score: (f) => blend([2.5, f.sharpness], [2, warm(f)], [2, f.brightRatio], [2, wide(f)]),
  },
];

/** Describe the measurements in plain words, so the guess can be checked. */
function buildEvidence(f: ImageFeatures): string[] {
  const lines: string[] = [];
  lines.push(
    f.brightness > 0.65 ? 'The light in this photo is bright.'
      : f.brightness < 0.35 ? 'The light in this photo is low.'
      : 'The light in this photo is even.',
  );
  lines.push(
    f.warmth > 0.12 ? 'The colour leans warm, like sun or lamplight.'
      : f.warmth < -0.12 ? 'The colour leans cool, like shade or window light.'
      : 'The colour sits between warm and cool.',
  );
  lines.push(
    f.sharpness > 0.5 ? 'The edges are crisp, so the cat held still.'
      : f.sharpness < 0.2 ? 'The edges are soft, so the cat was relaxed or moving slowly.'
      : 'The edges are moderately defined.',
  );
  lines.push(
    f.centreFocus > 0.55 ? 'Most of the detail sits in the middle of the frame.'
      : 'The detail spreads across the whole frame.',
  );
  if (f.darkRatio > 0.4) lines.push('Deep shadow covers a large part of the frame.');
  return lines;
}

export function readFeeling(f: ImageFeatures): Reading {
  const ranked = FEELINGS
    .map((feeling) => ({ feeling, value: feeling.score(f) }))
    // Ties break by catalogue order, so the same photo always reads the same.
    .sort((a, b) => b.value - a.value || FEELINGS.indexOf(a.feeling) - FEELINGS.indexOf(b.feeling));

  const margin = ranked[0].value - ranked[1].value;
  const confidence = Math.round(clamp(0.3 + 2.5 * margin, 0.3, 0.9) * 100) / 100;

  return {
    feeling: ranked[0].feeling,
    runnerUp: ranked[1].feeling,
    confidence,
    evidence: buildEvidence(f),
  };
}
