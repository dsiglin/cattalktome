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

## Two calibration bugs found after shipping, and what fixed them

Both had the same shape: a feature that looked like it should vary with
real content, normalized against a reference point no real photo ever
produces, so it barely varied at all.

**Sharpness** was normalized against a synthetic checkerboard's mean
|Laplacian| (~4.0). Measured across several real cat photos, raw values
landed between 0.02 and 0.12 — every real photo scored sharpness under
0.1 regardless of content. That made Locked-on and Curious unreachable
in practice (they need real sharpness variation to win) and let Sleepy /
Fully loafed win by default through their `(1 - sharpness)` terms.
Recalibrated against real face-crop photographs (a 0.16 reference); also
cut every formula's sharpness weight roughly in half, since it's an
honest but weak proxy for stillness — mostly reflecting camera/focus
quality, not the cat.

**Ear-base angle** was measured from zero degrees, but a relaxed,
upright ear in this 8-point scheme measures ~42 degrees by construction
(its two landmark points sit at different heights on the ear's outline —
a property of the landmark convention, not the cat's mood). Every calm
photo already read as 68-87% of the way to "ears pinned back." Recentered
on the empirically measured ~42-degree baseline instead of 0 — verified
against a real photo of a cat crouching from a dog (55° average,
correctly read as Wary).

Both fixes are grounded in real photos, not just synthetic test fixtures
— see `tests/test_pipeline_integration.py`'s
`test_sharpness_does_not_cluster_near_zero_on_a_real_photo`, which
guards the sharpness fix specifically using the repo's real fixture plus
a genuinely blurred copy of it. A synthetic checkerboard can't stand in
for that test: it has far more edge energy than any real photo even at
a coarse tile size, so it saturates the sharpness scale regardless of
which calibration is in use.

**Update:** this predicted problem became a real user report - "last
two images it didn't find cat face." The Haar cascade's thresholds
(`scaleFactor`/`minNeighbors` in `detect.py`) were loosened, recovering
6/9 → 8/9 real test photos (profile angles, mid-hiss open mouths, and
mid-turn heads now detect; ~70ms slower per request, zero new false
positives on blank/noise checks). One case remains genuinely
undetected: a cat with ears pinned flat against the skull, not just
swept back - see `test_detects_a_defensive_off_angle_cat_...` in
`tests/test_pipeline_integration.py` for the regression fixture and
the full measurement.

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

45 tests total, all passing. Run them:

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
regenerate `debug_output.jpg`, a visual overlay). Reading (after the
calibration fixes below): **Demanding** (confidence 0.58, runner-up
"Unimpressed") — a plausible read of a cat holding a level, symmetric,
direct gaze at the camera.

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

## Where this actually runs

Deployed to Cloud Run, project `cat-talk-to-me-03477` (a personal Google
account, isolated from any work project), region `us-central1`:

```bash
gcloud run deploy cat-talk-to-me-api \
  --source server/ \
  --region us-central1 \
  --min-instances=0 --max-instances=2 \
  --set-env-vars DAILY_CAP=100,PER_IP_LIMIT=10,PER_IP_WINDOW_S=3600
```

`--min-instances=0` was a deliberate correction from an earlier `=1` deploy:
an always-on instance draws real dollars, since it burns through Cloud Run's
free vCPU-second allowance in days rather than months. Scaling to zero costs
$0 at this app's traffic and adds well under a second of cold-start latency
in practice - a fair trade for staying free.

The frontend now calls this endpoint as its **only** way of getting a
reading - there is no client-side fallback. `app/main.py`'s CORS is still
wide open (`allow_origins=["*"]`); fine for a low-stakes personal project,
worth narrowing to the `github.io` origin if this ever needs to be locked
down.

## The privacy story, as it actually is now

The frontend's earlier design ran an on-device heuristic first and offered
this backend as an opt-in "deeper read." That's gone - the client-side
heuristic and its MobileNet cat-check were both deleted once this backend
proved it does the job better (a real face detector beats an ImageNet
classifier every time). Every reading now uploads the photo here. The app
says so plainly, and nothing is stored server-side - the photo is decoded,
measured, and discarded in memory per request.
