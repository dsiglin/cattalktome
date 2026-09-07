# Cat, Talk To Me 🐈

Upload a photo of your cat. The app tells you what it thinks your cat feels,
puts that feeling on the photo as a sticker, and hands it to the native
sharesheet.

The app is a static site. It runs entirely in the browser. Your photo never
leaves your device.

## How the reading works

The app cannot see ears, pupils or whiskers directly. It measures things it
*can* see, then matches those measurements to how cats actually carry moods:

| Measurement | Real cue it stands for |
|---|---|
| Brightness and warmth | Sunlight against shade. Relaxed cats settle into warm, even light. |
| Contrast and dark share | Deep shadow and wide pupils. Hard light reads as alarm. |
| Sharpness | Stillness against motion. A hunting cat freezes. A drowsy one softens. |
| Centre focus | A face that fills the frame. Cats ask for things up close. |
| Frame shape | A tall sit or loaf, against a wide sprawl. |

The app shows its confidence, names its second guess, and lists the evidence.
It says plainly that this is a considered guess, not a diagnosis.

MobileNet then confirms whether the photo really holds a cat. That step is
optional. If it fails, the app says nothing about it and works as normal.

## Commands

```bash
npm install
npm run dev      # local development
npm test         # the full test suite
npm run build    # production build into dist/
```

## Deploying

Push to `main`. The workflow in `.github/workflows/deploy.yml` runs the tests,
builds with the repository name as the base path, and publishes to GitHub
Pages. Turn on Pages in the repository settings, with **GitHub Actions** as the
source.

## Layout

| Path | Purpose |
|---|---|
| `src/lib/features.ts` | Turn pixels into numbers. |
| `src/lib/feelings.ts` | Turn numbers into a feeling. |
| `src/lib/analyse.ts` | Shrink the photo, then run the two steps above. |
| `src/lib/sticker.ts` | Lay out and draw the sticker. |
| `src/lib/share.ts` | Native sharesheet, with a download fallback. |
| `src/lib/cat-check.ts` | Optional MobileNet cat confirmation. |
| `src/main.ts` | Wire the interface. |
| `test/` | The test suite, including a real cat photograph. |

The test fixture is [Cat August 2010-4.jpg](https://commons.wikimedia.org/wiki/File:Cat_August_2010-4.jpg)
from Wikimedia Commons.
