# Cat, Talk To Me — deeper read (server proof of concept)

A small FastAPI service that reads a cat's actual face geometry, not the
whole photo. Built and verified locally; not yet deployed.

## What this reads, and what it refuses to read

The service finds the cat's face, places 8 landmarks on it, and measures
four things about **the face only**:

| Signal | Measured how | Module |
|---|---|---|
| Pupil dilation | Dark fraction inside a disc around each eye landmark, averaged | `app/eyes.py` |
| Ear spread | Distance between the ear tips over the interocular distance | `app/geometry.py` |
| Muzzle shape | Nose-to-chin distance over the interocular distance | `app/geometry.py` |
| Head tilt | Angle of the line between the eyes | `app/geometry.py` |

Nothing about the room is an input. Brightness, contrast, and sharpness
were removed after a real photo proved they contaminate the reading: an
alert, wide-eyed cat on a windowsill read as **Unimpressed 74%** because
the window light and a soft focus outvoted its face. The same photo now
reads **Curious**, with the evidence "The pupils are wide open."

Ear *angle* was also removed. In this 8-point scheme a relaxed ear
measures ~42 degrees by construction (its two points sit at different
heights on the ear outline). That is a property of the landmark
convention, not the cat, and it produced "ears swept back" on cats whose
ears were plainly up.

### The five feelings the face can support

| Feeling | Pupils | Ears | Muzzle | Head |
|---|---|---|---|---|
| Curious | wide | out | loose | tilted |
| Locked on | wide | out | tight | level |
| Startled | wide | pulled in | tight | tilted |
| Wary | (any) | pulled in | tight | tilted |
| Unimpressed | narrow | out | loose | level |

Pupils carry the most weight (4 of ~8 units). When the pupils cannot be
read - face too small, or a retinal glow blowing out the eye - `eyes.py`
says so, the pupil term drops out of every formula, and the evidence
reads "I could not read the pupils." Wary needs no pupil term, so a
frightened cat with glowing eyes still lands in the fear family.

### Honest limits

- **Pupils respond to ambient light as well as arousal.** The pupil term
  is about the cat's eye, not the room, but a cat in a dark room has wide
  pupils for optical reasons. The app discloses this.
- **Body posture is out of reach.** Tail, crouch, and piloerection are
  the strongest emotion cues cats give, and no permissively licensed
  cat-pose model exists (YOLO-family models are AGPL, DeepLabCut is
  academic-only). Research documented in the session; the closest
  future step is opencv_zoo's NanoDet (Apache-2.0, 3.6 MB) as a
  whole-cat first stage.
- **No whisker point exists** in the 8-point scheme. Whisker change is
  one of the Feline Grimace Scale's five action units.
- **The training data is the 2008 Zhang et al. cat dataset**, never
  validated against any emotion scale. It finds where points are, not
  what they mean. No emotion-tagged corpus was used for training; the
  ones that exist are small, self-labelled, and would teach the model
  the labeller's guesses.
- **pycatfd has no declared license** (GitHub's license API returns
  `null`). Documented caveat for a personal project.

## The staged detector

Detection quality is the real bottleneck. A single loosened Haar pass
recovered profile shots but, on the windowsill cat, found the cat's
**eye**, fitted a whole face inside a 90-pixel box, and reported it
confidently. `app/detect.py` now runs three stages and stops at the
first hit:

| Stage | `detector` value | `reliable` | What it does |
|---|---|---|---|
| 1 strict | `haar` | true | Extended cascade, `scaleFactor=1.05`, `minNeighbors=3`, `minSize=75` |
| 2 rotated | `haar-rotated` | true | Both cascades at ±20° and ±35°; boxes mapped back, clustered by IoU ≥ 0.3, need ≥ 2 votes; winner by (votes, area); median box |
| 3 loose | `haar-loose` | **false** | Extended cascade at `1.02`/`2`. The reading hedges: `reliable=false` |

Results on the three fixtures in `test/fixtures/`:

| Fixture | Stage | Box | Reading |
|---|---|---|---|
| `cat.jpg` (lounging) | `haar` | (784, 215, 158×158) | Unimpressed - narrow pupils, level head |
| `alert-tabby-windowsill.png` (user's cat) | `haar-rotated` | (365, 152, 326×328) | Curious - pupils 0.63, ears out |
| `angry-sphynx.jpg` (defensive, profile) | `haar-loose` | (183, 118, 109×109) | Locked on, flagged unreliable |

Across 11 real photos measured during development: 7 resolve at stage 1
with IoU ≥ 0.98 against the strict box, 1 at stage 2, 2 extreme poses at
stage 3 flagged unreliable, 1 honest "no cat face found."

## Architecture

| Module | Responsibility | Tested how |
|---|---|---|
| `app/detect.py` | Staged Haar detection (strict → rotated+voted → loose) → dlib shape predictor | 8 integration tests on 3 real fixtures |
| `app/geometry.py` | Pure math: 8 landmarks → head tilt, ear spread, muzzle ratio, symmetry | 15 unit tests on synthetic coordinates |
| `app/eyes.py` | Pupil dilation from the eye landmarks; declares itself unusable rather than guess | 6 unit tests on synthetic eyes |
| `app/feelings.py` | Maps geometry + eyes → one of 5 feelings, with face-only evidence | 15 unit tests, reachability-checked, banned-word check on evidence |
| `app/pipeline.py` | Wires the above together | integration tests above |
| `app/guard.py` | Per-IP sliding-window limit + global daily cap | 7 unit tests with an injected fake clock |
| `app/main.py` | FastAPI endpoint, CORS, upload validation, guard wiring | tested live via curl, see below |

51 tests total, all passing. Run them:

```bash
cd server
source .venv/bin/activate
python -m pytest tests/ -q
```

## Proof it works - real cat photo

```bash
curl -X POST http://127.0.0.1:8080/analyze \
  -F "photo=@../test/fixtures/alert-tabby-windowsill.png;type=image/png"
```

The response carries the feeling, the evidence, `detector`, `reliable`,
the `face_box`, and the raw `geometry` and `eyes` measurements so a
wrong reading can be traced to the number that caused it.

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
