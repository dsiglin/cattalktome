/**
 * Decorative sticker phrases - internet-cat-culture flavour, purely for
 * fun. These carry no meaning about the reading; one is picked at random
 * and stuck on the photo alongside the actual feeling, the way a laptop
 * lid collects stickers that have nothing to do with each other.
 *
 * Original phrasing - no third-party artwork. Flaticon's cat stickers are
 * licensed stock assets (attribution required even on the free tier, no
 * plain hotlink URL suitable for a public site), so this is a from-scratch
 * set in the app's own voice instead.
 */
export const GENZ_STICKERS: string[] = [
  'no cap 🐾',
  'certified banger 🔥',
  "it's giving feline 💅",
  'living rent free 👑',
  'core memory unlocked ✨',
  'the audacity 😤',
  "he's built different 💪",
  'understood the assignment ✅',
  'main character energy 🌟',
  'vibe check: passed ✅',
  'not the drama 💀',
  'we love to see it 🫶',
  'certified cutie 🥹',
  'absolutely sending me 😭',
  'the range 📈',
];

/** Pick one phrase at random. Pass an rng for deterministic tests. */
export function pickRandomGenZSticker(rng: () => number = Math.random): string {
  const index = Math.min(GENZ_STICKERS.length - 1, Math.floor(rng() * GENZ_STICKERS.length));
  return GENZ_STICKERS[index];
}
