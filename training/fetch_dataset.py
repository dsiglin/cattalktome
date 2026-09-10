"""Download or repair the CAT_DATASET zips: parallel range chunks, each chunk
size-verified and retried, then whole-file MD5 against archive.org's own value."""
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(HERE, exist_ok=True)
BASE = "https://archive.org/download/CAT_DATASET/"
N = 8

meta = json.loads(subprocess.run(["curl", "-sL", "https://archive.org/metadata/CAT_DATASET"], capture_output=True, text=True).stdout)
files = {f["name"]: (int(f["size"]), f["md5"]) for f in meta["files"] if f["name"].endswith(".zip")}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def fetch_part(name, i, start, end):
    part = os.path.join(HERE, f"{name}.part{i}")
    want = end - start + 1
    for attempt in range(6):
        if os.path.exists(part) and os.path.getsize(part) == want:
            return True
        subprocess.run(["curl", "-sL", "--retry", "3", "-r", f"{start}-{end}", "-o", part, BASE + name])
    ok = os.path.exists(part) and os.path.getsize(part) == want
    print(f"  part {i}: {'ok' if ok else 'FAILED'} ({os.path.getsize(part) if os.path.exists(part) else 0}/{want})", flush=True)
    return ok


def ensure(name):
    size, want_md5 = files[name]
    path = os.path.join(HERE, name)
    if os.path.exists(path) and os.path.getsize(path) == size and md5(path) == want_md5:
        print(f"{name}: already good"); return True
    print(f"{name}: downloading {size/1e6:.0f} MB in {N} chunks", flush=True)
    chunk = (size + N - 1) // N
    ranges = [(i, i * chunk, min(size, (i + 1) * chunk) - 1) for i in range(N)]
    with ThreadPoolExecutor(N) as ex:
        oks = list(ex.map(lambda r: fetch_part(name, *r), ranges))
    if not all(oks):
        print(f"{name}: some chunks failed"); return False
    with open(path, "wb") as out:
        for i in range(N):
            p = os.path.join(HERE, f"{name}.part{i}")
            with open(p, "rb") as f:
                out.write(f.read())
            os.remove(p)
    got = md5(path)
    print(f"{name}: md5 {got} {'OK' if got == want_md5 else 'MISMATCH (want ' + want_md5 + ')'}", flush=True)
    return got == want_md5


if __name__ == "__main__":
    ok = all(ensure(n) for n in ("CAT_DATASET_01.zip", "CAT_DATASET_02.zip"))
    print("ALL_GOOD" if ok else "REPAIR_FAILED")
    sys.exit(0 if ok else 1)
