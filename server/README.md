# Cat, Talk To Me — deeper read (server proof of concept)

A small FastAPI service that reads a cat's actual face geometry, not the
whole photo. Built and verified locally; not yet deployed.

## What this reads, and what it refuses to read

The service finds the cat's face, places 8 landmarks on it, and measures
five things about **the face only**:

| Signal | Measured how | Module |
|---|---|---|
| Pupil dilation | Dark fraction inside a disc around each eye landmark, averaged | `app/eyes.py` |
| Ear spread | Distance between the ear tips over the interocular distance | `app/geometry.py` |
| Muzzle shape | Nose-to-chin distance over the interocular distance | `app/geometry.py` |
| Head tilt | Angle of the line between the eyes | `app/geometry.py` |
| Face direction | Nose displacement along the eye line (`nose_offset`); a turned head drifts the nose toward one eye | `app/geometry.py` |

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

### The seven feelings the face can support

The catalogue follows the standard cat facial-expression chart (Happy,
Angry, Frightened, Playful, Content) and the common body-language poster
(Interested, Attentive, Cautious, Trusting, Irritated, Focus, ...), kept
to what the five cues can tell apart:

| Feeling | Pupils | Ears | Muzzle | Head | Face |
|---|---|---|---|---|---|
| Curious | wide | out | loose | tilted | - |
| Focused | wide | out | tight | level | - |
| Frightened | wide | pinned | tight | tilted | - |
| Cautious | (any) | pinned | tight | tilted or low | - |
| Irritated | narrow | pinned | tight | - | toward you |
| Trusting | narrow | out | loose | level | toward you |
| Unimpressed | narrow | out | - | - | turned away |

Two rules came from looking at real photos, not from the charts:

- **Narrow pupils mean trust only on a relaxed face.** A scared tabby
  measured narrow pupils with pinned ears and a tight muzzle. Trusting is
  therefore gated: pinned ears or a tight muzzle scale its score down
  hard, and that cat reads Irritated instead.
- **Unimpressed is about looking away**, not about pupils. Cats facing
  the lens scored nose symmetry 0.90-0.99; heads turned part-way scored
  0.87-0.89. Below ~0.91 the face counts as turned. The separation is
  thin, because a frontal face detector only finds faces that mostly
  face the lens, so this is a weak cue and Unimpressed needs narrow
  pupils and easy ears as well.
- **A "0.00" pupil is not a narrow pupil.** Three frightened cats on
  flash-lit photos measured no dark pixel at all inside the eye. Every
  real pupil, even a slit, has a near-black core. An eye with none is
  unreadable (reflection, squint, or a missed landmark) and drops out;
  before this rule, those frightened cats read as calm.

When the pupils cannot be read - face too small, or a retinal glow
blowing out the eye - `eyes.py` says so, the pupil term drops out of
every formula, and the evidence reads "I could not read the pupils."
Cautious needs no pupil term, so a frightened cat with glowing eyes
still lands in the fear family.

### Honest limits

- **Haar cannot tell chest fur from a face, and cannot always tell a
  face from a tail.** Measured on real photos: a fur "face" scored a Haar
  level weight of 1.96 with 6 neighbours and strong eye-region contrast;
  a real crouching cat's face scored -0.06 with 3. A tail collected the
  same 2 rotation votes, from the same two cascades at the same angle, as
  a real face. No cheap validator separates them. Such boxes only ever
  come from the loose stage, which is flagged unreliable and capped at
  50% confidence, and the app says the box may be fur. A cat in full
  profile therefore returns either "no cat face found" or a hedged,
  capped reading - never a confident one. The real fix is a better
  detector (see NanoDet below).

- **"Content" (half-closed eyes) is not offered.** It needs eye
  aperture, and the 8-point scheme has no eyelid points.
- **Pupils respond to ambient light as well as arousal.** The pupil term
  is about the cat's eye, not the room, but a cat in a dark room has wide
  pupils for optical reasons. The app discloses this. Flash photos of
  frightened cats also measured "narrow" because the pupil reflected
  bright - the tapetum confound.
- **Body posture is out of reach.** Tail, crouch, and piloerection are
  the strongest emotion cues cats give, and no permissively licensed
  cat-pose model exists (YOLO-family models are AGPL, DeepLabCut is
  academic-only). The closest future step is opencv_zoo's NanoDet
  (Apache-2.0, 3.6 MB) as a whole-cat first stage.
- **No whisker point exists** in the 8-point scheme. Whisker change is
  one of the Feline Grimace Scale's five action units.
- **The training data is the 2008 Zhang et al. cat dataset**, never
  validated against any emotion scale. It finds where points are, not
  what they mean. No emotion-tagged corpus was used for training; the
  ones that exist are small, self-labelled, and would teach the model
  the labeller's guesses.
- **The Haar cascades are the weak link, and a replacement is in training.**
  `training/catface_detector_colab.ipynb` fine-tunes YOLOX-Nano
  (Apache-2.0) on the CC BY 4.0 "cat-face-data" set (8,153 images) with
  rotation augmentation. When its ONNX file lands in `models/`, it becomes
  stage 0 automatically; `tools/calibrate_face_box.py` then measures how
  its boxes are framed relative to the Haar boxes the landmark predictor
  expects.
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
| 0 learned | `yolox-face` | true | YOLOX-Nano fine-tuned on cat faces (`app/facedet.py`). Active only when `models/catface_yolox_nano.onnx` exists - see `training/`. |
| 1 strict | `haar` | true | Extended cascade, `scaleFactor=1.05`, `minNeighbors=3`, `minSize=75` |
| 2 rotated | `haar-rotated` | true | Both cascades at ±20° and ±35°; boxes mapped back, clustered by IoU ≥ 0.3, need ≥ 2 votes; winner by (votes, area); median box |
| 3 loose | `haar-loose` | **false** | Extended cascade at `1.02`/`2`. The reading hedges: `reliable=false`, confidence capped at 0.50 |
| fix-up | `haar-contained` | true | After stage 1 or 2: if a box ≥ 2.5× larger, found by the extended cascade at loose settings and confirmed by the standard cascade, *contains* the chosen box, the chosen box was a face part (a Savannah's muzzle, a tabby's eye). Use the container. |

Before any stage runs, **NanoDet** (OpenCV model zoo, Apache-2.0, 3.6 MB,
`app/catdet.py`) finds every whole cat in the photo. Two uses: a face
candidate whose centre lies outside every cat box is vetoed (this removed
a leaf patch and a person's face that the strict stage had accepted), and
if no stage finds a face while a cat is present the API returns a
different message - "I can see a cat, but not its face" - instead of "no
cat". Measured on the corpus: no correct reading changed; one
loose-stage box that was probably fur became "cat, no face".

Detection runs on a copy no larger than **1400 px** on its long side - the
same size the browser app uploads. On a 1450×2576 phone photo every stage
missed a 550 px face at full resolution and the loose stage settled on a
patch of fur; at 500-1600 px the rotated stage found the same face every
time. The landmarks and the pupil measurement still use the full-resolution
pixels.

Results on the three fixtures in `test/fixtures/`:

| Fixture | Stage | Box | Reading |
|---|---|---|---|
| `cat.jpg` (lounging) | `haar` | (784, 215, 158×158) | Trusting - narrow pupils, ears out, facing the lens |
| `alert-tabby-windowsill.png` (user's cat) | `haar-rotated` | (365, 152, 326×328) | Curious - pupils 0.63, ears out |
| `angry-sphynx.jpg` (defensive, profile) | `haar-loose` | (183, 118, 109×109) | Focused, flagged unreliable |

Across 11 real photos measured during development: 7 resolve at stage 1
with IoU ≥ 0.98 against the strict box, 1 at stage 2, 2 extreme poses at
stage 3 flagged unreliable, 1 honest "no cat face found."

## Architecture

| Module | Responsibility | Tested how |
|---|---|---|
| `app/catdet.py` | NanoDet whole-cat boxes: veto off-cat face candidates; "cat but no face" message | 6 tests on real fixtures |
| `app/facedet.py` | Learned YOLOX-Nano cat-face detector (optional model file; see `training/`) | activates when the model is present |
| `app/detect.py` | Downscale to 1400px → learned face → staged Haar (strict → rotated+voted → loose), each gated by the cat boxes → face-part fix-up → dlib shape predictor on full-res | 12 integration tests on 5 real fixtures |
| `app/geometry.py` | Pure math: 8 landmarks → head tilt, ear spread, muzzle ratio, nose offset/symmetry | 15 unit tests on synthetic coordinates |
| `app/eyes.py` | Pupil dilation from the eye landmarks; declares itself unusable rather than guess | 8 unit tests on synthetic eyes |
| `app/feelings.py` | Maps geometry + eyes → one of 7 feelings, with face-only evidence | 20 unit tests, reachability-checked, banned-word check on evidence |
| `app/pipeline.py` | Wires the above together | integration tests above |
| `app/guard.py` | Per-IP sliding-window limit + global daily cap | 7 unit tests with an injected fake clock |
| `app/main.py` | FastAPI endpoint, CORS, upload validation, guard wiring | tested live via curl, see below |

68 tests total, all passing. Run them:

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
