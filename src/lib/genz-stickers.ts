/**
 * Decorative bonus stickers - real artwork, licensed for this use.
 *
 * Files live in public/stickers/, downloaded from Flaticon's free cat
 * sticker collection (https://www.flaticon.com/free-stickers/cat) under
 * an account with permission to use them. Flaticon's free license
 * requires attribution, which is why the footer credits Flaticon - see
 * index.html.
 *
 * One is picked at random and stuck on the photo alongside the actual
 * feeling, the way a laptop lid collects stickers that have nothing to
 * do with each other.
 */
export interface GenZSticker {
  file: string;
  alt: string;
}

export const GENZ_STICKERS: GenZSticker[] = [
  { file: 'smile.png', alt: 'A happy orange cat smiling' },
  { file: 'love.png', alt: 'A cat making a finger heart' },
  { file: 'arrogant.png', alt: 'A smug, self-satisfied cat' },
  { file: 'scratch.png', alt: 'A cat stretching up to scratch' },
  { file: 'reading.png', alt: 'A cat reading a book' },
  { file: 'cute.png', alt: 'A wide-eyed, curious cat' },
  { file: 'angry.png', alt: 'An angry cat' },
  { file: 'sad.png', alt: 'A sad cat' },
  { file: 'tired.png', alt: 'An exhausted cat with X eyes' },
  { file: 'thinking.png', alt: 'A cat deep in thought' },
  { file: 'celebration.png', alt: 'A cat celebrating' },
  { file: 'sleep.png', alt: 'A cat sleeping on a crescent moon' },
  { file: 'black-cat.png', alt: 'A mysterious black cat under the stars' },
  { file: 'idea.png', alt: 'A nerdy cat with glasses and a lightbulb idea' },
  { file: 'screaming.png', alt: 'A startled, screaming cat' },
  { file: 'shy.png', alt: 'A shy cat hiding its face' },
];

/** Pick one sticker at random. Pass an rng for deterministic tests. */
export function pickRandomGenZSticker(rng: () => number = Math.random): GenZSticker {
  const index = Math.min(GENZ_STICKERS.length - 1, Math.floor(rng() * GENZ_STICKERS.length));
  return GENZ_STICKERS[index];
}
