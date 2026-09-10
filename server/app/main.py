"""
FastAPI wrapper around the analysis pipeline. Local proof-of-concept today;
containerizes as-is for a Cloud Run deployment later (see server/README.md).
"""
import os

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .pipeline import analyse_bytes
from .detect import NoCatFaceFound, CatFoundButNoFace
from .guard import RateGuard, GuardConfig

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB - generous for a phone photo, small enough to reject abuse

# Defaults for a proof of concept: a generous per-visitor allowance, and a
# global ceiling that caps the worst-case daily bill regardless of how many
# different IPs show up (see server/README.md for the reasoning and the
# cost math per model choice).
guard = RateGuard(GuardConfig(
    per_ip_limit=int(os.environ.get("PER_IP_LIMIT", 10)),
    per_ip_window_s=int(os.environ.get("PER_IP_WINDOW_S", 3600)),
    daily_cap=int(os.environ.get("DAILY_CAP", 100)),
))

app = FastAPI(title="Cat, Talk To Me - deeper read", version="0.1.0")

# Wide open for the local proof-of-concept. Before any public deploy, this
# must be narrowed to the actual github.io origin (see server/README.md).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(request: Request, photo: UploadFile = File(...)):
    client_ip = request.client.host if request.client else "unknown"
    result_guard = guard.check(client_ip)
    if not result_guard.allowed:
        status = 429
        detail = ("This service has a daily visitor cap and it has been reached today - "
                  "please try again tomorrow." if result_guard.reason == "daily_cap"
                  else "You have hit the per-visitor limit for this hour - please slow down.")
        raise HTTPException(status_code=status, detail=detail)

    if photo.content_type is None or not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="upload must be an image")

    data = await photo.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image too large")

    try:
        result = analyse_bytes(data)
    except CatFoundButNoFace:
        # These two `detail` strings are shown to the user as-is by the app.
        raise HTTPException(status_code=422, detail=(
            "I can see a cat, but not its face. I need the face turned mostly toward the camera - "
            "a side profile or a hidden face gives me nothing to read. Try one where the cat is looking at you?"))
    except NoCatFaceFound:
        raise HTTPException(status_code=422, detail=(
            "I could not find a cat in that photo. If there is one, try a photo where its face is "
            "bigger and turned toward the camera?"))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    g, e, r = result.geometry, result.eyes, result.reading
    return JSONResponse({
        "feeling": {"id": r.feeling.id, "label": r.feeling.label, "emoji": r.feeling.emoji,
                    "blurb": r.feeling.blurb, "cue": r.feeling.cue},
        "runner_up": {"id": r.runner_up.id, "label": r.runner_up.label},
        "confidence": r.confidence,
        "reliable": r.reliable,
        "evidence": r.evidence,
        "face_box": list(result.face.box),
        "detector": result.face.detector,
        # Diagnostics: the measurements the reading was built from. All of
        # these are about the cat's face; none are about the room's light.
        "geometry": {
            "head_tilt_deg": round(g.head_tilt_deg, 2),
            "ear_splay_ratio": round(g.ear_splay_ratio, 2),
            "muzzle_ratio": round(g.muzzle_ratio, 2),
            "nose_symmetry": round(g.nose_symmetry, 2),
            "nose_offset": round(g.nose_offset, 2),
        },
        "eyes": {
            "pupil_dilation": round(e.pupil_dilation, 2),
            "usable": e.usable,
        },
    })
