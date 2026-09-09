# Cat, Talk To Me 🐈

Upload a photo of your cat. The app tells you what it thinks your cat feels,
puts that feeling on the photo as a sticker, and hands it to the native
sharesheet.

The frontend is a static site on GitHub Pages. The reading itself comes from
a small backend on Google Cloud Run — see [server/](server/) for how that
works. Your photo is sent there just long enough to be read, then it's gone;
nothing is stored.

## How the reading works

The backend finds the cat's actual face (an OpenCV Haar cascade, with a dlib
detector as fallback), then measures its real geometry — ear-base angle,
head tilt, muzzle ratio — with an 8-point facial landmark model
([pycatfd](https://github.com/marando/pycatfd)), plus how bright, sharp, and
even the light is on the face itself. That combination maps onto one of ten
feelings, chosen to mirror how cats actually carry those moods.

The app shows its confidence, names its second guess, and lists the evidence.
It says plainly that this is a considered guess, not a diagnosis — no
validated model for reading general cat *emotion* (as opposed to pain) exists
anywhere, and this app doesn't pretend otherwise. Full reasoning and sourcing
in [server/README.md](server/README.md).

## Commands

```bash
npm install
npm run dev      # local development
npm test         # the full test suite
npm run build    # production build into dist/
```

The backend is a separate project — see [server/README.md](server/README.md)
for running or redeploying it.

## Deploying the frontend

Push to `main`. The workflow in `.github/workflows/deploy.yml` runs the tests,
builds with the repository name as the base path, and publishes to GitHub
Pages. Pages is set to build from **GitHub Actions**.

## Layout

| Path | Purpose |
|---|---|
| `src/lib/deeper-read.ts` | Calls the backend, translates its response and its failure modes. |
| `src/lib/sticker.ts` | Lay out and draw the sticker. |
| `src/lib/share.ts` | Native sharesheet, with a download fallback. |
| `src/main.ts` | Wire the interface. |
| `test/` | Unit tests for the modules above, with an injected `fetch`/measurer so nothing touches the network. |
| `server/` | The FastAPI backend: face detection, landmark geometry, the feeling catalogue, and its own test suite. |

The photo used for manual testing is
[Cat August 2010-4.jpg](https://commons.wikimedia.org/wiki/File:Cat_August_2010-4.jpg)
from Wikimedia Commons, at `test/fixtures/cat.jpg`.
