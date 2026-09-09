/**
 * Decorative bonus stickers - real artwork, licensed for this use.
 *
 * Files live in public/stickers/, downloaded from Flaticon's free cat
 * sticker collection (https://www.flaticon.com/free-stickers/cat) under
 * an account with permission to use them. Flaticon's free license
 * requires attribution, which is why the footer credits Flaticon - see
 * index.html.
 *
 * One is picked to match the feeling that was read and stuck on the photo
 * alongside the feeling label. It used to be fully random, which put a
 * hissing cat sticker on a photo of a cat that was plainly calm - the
 * sticker contradicted the reading. Now each feeling has its own small
 * pool, and the pick is random only within that pool.
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

/**
 * Which stickers suit which feeling. Keys are the server's feeling ids
 * (server/app/feelings.py). Every sticker in the catalogue appears in at
 * least one pool, and every pool has at least two entries so the pick
 * still varies.
 */
export const STICKERS_BY_FEELING: Record<string, string[]> = {
  curious: ['cute.png', 'idea.png', 'thinking.png', 'scratch.png'],
  focused: ['thinking.png', 'idea.png', 'reading.png'],
  frightened: ['screaming.png', 'shy.png', 'sad.png'],
  cautious: ['shy.png', 'thinking.png', 'black-cat.png', 'sad.png'],
  irritated: ['angry.png', 'arrogant.png', 'tired.png'],
  trusting: ['love.png', 'smile.png', 'celebration.png', 'sleep.png'],
  unimpressed: ['arrogant.png', 'tired.png', 'reading.png', 'black-cat.png'],
};

function byFile(file: string): GenZSticker {
  const found = GENZ_STICKERS.find((s) => s.file === file);
  if (!found) throw new Error(`sticker pool names a file not in the catalogue: ${file}`);
  return found;
}

/** The stickers that suit a feeling. Unknown feeling ids fall back to the
 * whole catalogue rather than failing - the bonus sticker is decoration. */
export function stickersFor(feelingId: string): GenZSticker[] {
  const pool = STICKERS_BY_FEELING[feelingId];
  return pool ? pool.map(byFile) : GENZ_STICKERS;
}

/** Pick one sticker that suits the feeling. Pass an rng for deterministic tests. */
export function pickStickerFor(feelingId: string, rng: () => number = Math.random): GenZSticker {
  const pool = stickersFor(feelingId);
  const index = Math.min(pool.length - 1, Math.floor(rng() * pool.length));
  return pool[index];
}

/** Pick one sticker at random from the whole catalogue. Pass an rng for deterministic tests. */
export function pickRandomGenZSticker(rng: () => number = Math.random): GenZSticker {
  const index = Math.min(GENZ_STICKERS.length - 1, Math.floor(rng() * GENZ_STICKERS.length));
  return GENZ_STICKERS[index];
}
