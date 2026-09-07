# Cat, Talk To Me — deeper read (server proof of concept)

A small FastAPI service that reads a cat's actual face geometry, not the
whole photo. Built and verified locally; not yet deployed.

## What this fixes

The browser app measures brightness/contrast/sharpness across the *whole
photo* — sofa, wallpaper, and cat included. Research turned up a real,
if old and narrow, tool for doing better: **pycatfd**, an 8-point cat
facial landmark detector (dlib), paired here with **OpenCV's cat-face
Haar cascade** for detection. This service crops to the cat's actual face
first, then measures real geometry (ear-base angle, head tilt, muzzle
ratio) on top of the same honest photometrics — so every measurement is
about the cat, not the room.

## What it still cannot do

This is a genuinely better proxy, not a validated emotion reader. Be
plain with users about both:

- **No whisker point exists** in this 8-point scheme (or in any released
  cat landmark scheme found in research). Whisker change is one of the
  Feline Grimace Scale's five action units — it is permanently out of
  reach here.
- **Ear "position" here is base-angle in the image plane only** — not
  rotation toward or away from the camera. Treat `left_ear_angle_deg`/
  `right_ear_angle_deg` as a coarse stand-in, documented in
  `app/geometry.py`.
- **The training data is the 2008 Zhang et al. cat dataset**, known to
  have lost most of its original images and some bad landmark points.
  It was never validated against any pain or emotion scale — it just
  finds where facial points are, not what they mean.
- **pycatfd has no declared license** (confirmed via GitHub's license
  API — `null`). Fine for this local proof of concept; do not ship this
  publicly without contacting the author or replacing it.

## Architecture

| Module | Responsibility | Tested how |
|---|---|---|
| `app/geometry.py` | Pure math: 8 landmarks → head tilt, ear angles, muzzle ratio, symmetry | 15 unit tests on synthetic coordinates |
| `app/photometrics.py` | Brightness/contrast/sharpness on the face crop only | 6 unit tests on synthetic images |
| `app/feelings.py` | Maps geometry + photometrics → one of 10 feelings (same catalogue as the browser app) | 12 unit tests, reachability-checked |
| `app/detect.py` | OpenCV Haar cascade (primary) → pycatfd FHOG detector (fallback) → dlib shape predictor | exercised by the integration test |
| `app/pipeline.py` | Wires the above together | 4 integration tests against the real cat fixture |
| `app/guard.py` | Per-IP sliding-window limit + global daily cap | 7 unit tests with an injected fake clock |
| `app/main.py` | FastAPI endpoint, CORS, upload validation, guard wiring | tested live via curl, see below |

44 tests total, all passing. Run them:

```bash
cd server
source .venv/bin/activate
python -m pytest tests/ -q
```

## Proof it works — real cat photo

```bash
curl -X POST http://127.0.0.1:8080/analyze \
  -F "photo=@../test/fixtures/cat.jpg;type=image/jpeg"
```

Verified result on the repo's real fixture: face detected at
`(784, 215, 159×159)` via the Haar cascade, all 8 landmarks landing
correctly on eyes/nose/chin/ears (see `debug_annotate.py` — run it to
regenerate `debug_output.jpg`, a visual overlay). Reading: **Unimpressed**
(confidence 0.38, runner-up "Fully loafed") — a plausible read of a cat
lounging flat on a step, ears relaxed, muzzle open.

## The abuse guard

Two layers, because a per-IP limit alone does nothing against a hundred
different IPs:

1. **Per-IP sliding window** — `PER_IP_LIMIT` requests per `PER_IP_WINDOW_S`
   seconds (default 10/hour). Keyed on the real transport-layer IP
   (`request.client.host`) — never a client-supplied header, which is
   trivially spoofable.
2. **Global daily cap** — `DAILY_CAP` total requests per rolling 24 hours
   (default 100), regardless of who's asking. This is what actually
   protects your bill.

Both are **in-memory and single-process** — correct for one Cloud Run
instance pinned to `min-instances=1 max-instances=1`. Running more than
one instance splits the counters per-instance and the global cap stops
being global; move to Redis or Firestore before scaling past one
instance.

Configure via environment variables:

```bash
PER_IP_LIMIT=10 PER_IP_WINDOW_S=3600 DAILY_CAP=100 uvicorn app.main:app --port 8080
```

## Running locally

```bash
cd server
python3.13 -m venv .venv   # or any Python 3.11-3.13; dlib builds from source
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

First install builds dlib from source (~1-2 minutes, needs cmake + a C++
compiler — `xcode-select --install` on macOS). Model weights already live
in `models/` (downloaded from pycatfd and OpenCV, ~2.7 MB together).

## Running the container (proves Cloud Run readiness)

```bash
cd server
docker build -t cat-talk-to-me-api:slim .
docker run -p 8080:8080 -e DAILY_CAP=100 cat-talk-to-me-api:slim
curl http://localhost:8080/health
```

Multi-stage build: 569 MB final image (compiling dlib needs
build-essential + cmake, ~600 MB; the runtime stage only needs the
compiled output + `libgl1`/`libglib2.0-0`). Verified: correct output,
~70ms per request including the first request after a cold container
start.

## What's NOT done yet — needs your explicit go-ahead

- **Actual cloud deployment.** This is built and container-verified
  locally only. Deploying to Cloud Run needs your GCP project ID and
  `gcloud auth login` — real infrastructure, so I stopped short of
  doing it without your say-so. Sketch:
  ```bash
  gcloud run deploy cat-talk-to-me-api \
    --source server/ \
    --region us-central1 \
    --min-instances=1 --max-instances=1 \
    --set-env-vars DAILY_CAP=100,PER_IP_LIMIT=10
  ```
  (`--min-instances=1` avoids Cloud Run's cold-start and keeps the
  in-memory guard meaningful; it also means the instance runs — and
  bills — continuously rather than scaling to zero. Confirm that
  tradeoff before deploying.)
- **CORS narrowing.** `app/main.py` currently allows every origin
  (`allow_origins=["*"]`) for local testing. Before any public deploy,
  narrow this to the actual `github.io` origin.
- **Frontend wiring.** The browser app doesn't call this endpoint yet —
  it still only does the instant, on-device read. Wiring "deeper read"
  as an opt-in button is the next step, once the API has a real URL.
- **The privacy promise.** The browser app currently tells users their
  photo never leaves the device. Once this API exists, that claim needs
  to become conditional — true for the instant read, false for the
  deeper one — and the UI needs to say so before the photo uploads.
