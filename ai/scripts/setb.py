"""
Set B: real webcam frames. Record, then evaluate.

Spec §6.1 makes Set B the adoption criterion, and it is still the one thing no
amount of work on the AI-Hub data can substitute for. A' (Step 4b) reproduces
the FE->BE imaging path but the pixels underneath are still AI-Hub smartphone
selfies. Set B is the only thing that answers whether the model holds up on a
laptop sensor, in office lighting, on a face that is not in the training
distribution at all.

The full set is 7 people x 4 emotions (§6.1, ~400+ frames). This script also
supports the n=1 version -- one person, 4 emotions, 30 seconds each, ~360
frames. That is statistically weak and must be labelled as such wherever it
appears in a performance table. It is not a substitute for the 7-person set. It
answers exactly one question: does the model fall apart on a real webcam.

Raw frames are stored at full sensor resolution, NOT pre-downscaled. Step 4b
found that the FE transform choice (stretch to square vs. preserve aspect) moves
accuracy by more than 0.35, so the transform has to stay a variable that can be
re-applied, not something baked into the recording.

  record:  uv run python scripts/setb.py record --person me --seconds 30
  eval:    uv run python scripts/setb.py eval

Frames land in ai/data/setb/<person>/<emotion>/*.png plus a manifest.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import SERVICE_PRIOR, score
from eval_aprime import FE_SIZE, _crop48, _detect, _init_worker, predict
from train import ART_DIR, CLASSES, ORIG_COLS, build_model, load_original7

HERE = os.path.dirname(os.path.abspath(__file__))
SETB_DIR = os.path.join(HERE, "..", "data", "setb")
FPS = 3          # §3.1: the service frame rate

PROMPT = {
    "happy": "Smile. Think of something genuinely funny -- not a held grin.",
    "embarrassed": "Flustered: caught off guard by a question you cannot answer.",
    "anxious": "Tense. Interview nerves -- braced, eyes a little fixed.",
    "neutral": "Resting face. Listening to a question, not reacting yet.",
}


def record(args):
    """Capture `seconds` of each emotion at 3 fps from the default camera."""
    person_dir = os.path.join(SETB_DIR, args.person)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"could not open camera {args.camera}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {args.camera}: {w}x{h}, capturing {args.seconds}s per emotion "
          f"at {FPS} fps\n")

    manifest = []
    try:
        for emotion in CLASSES:
            out = os.path.join(person_dir, emotion)
            os.makedirs(out, exist_ok=True)
            print(f"=== {emotion.upper()} ===\n  {PROMPT[emotion]}")
            for c in (3, 2, 1):
                print(f"  starting in {c}...", flush=True)
                t = time.time()
                while time.time() - t < 1.0:
                    cap.read()                    # keep the sensor warm and current
            n = int(args.seconds * FPS)
            for i in range(n):
                due = time.time() + 1.0 / FPS
                ok, frame = cap.read()
                if not ok:
                    print("  ! dropped frame")
                    continue
                path = os.path.join(out, f"{i:04d}.png")
                cv2.imwrite(path, frame)          # BGR on disk, lossless
                manifest.append({"person": args.person, "emotion": emotion,
                                 "frame": i, "path": os.path.relpath(path, SETB_DIR)})
                if i % FPS == 0:
                    print(f"  {i // FPS + 1}/{args.seconds}s", end="\r", flush=True)
                while time.time() < due:
                    pass
            print(f"  captured {n} frames -> {out}        ")
    finally:
        cap.release()

    mpath = os.path.join(SETB_DIR, "manifest.jsonl")
    with open(mpath, "a", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"\nappended {len(manifest)} frames to {mpath}")
    print("\nThe label is the emotion you were ASKED to perform, not what your face")
    print("actually did. A posed 'anxious' that reads as neutral is a mislabelled")
    print("frame, and there is no annotator majority here to catch it. Delete any")
    print("stretch you know you performed badly before evaluating.")


def fe_path(bgr, stretch):
    """The FE transform, both candidates from Step 4b."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if stretch:
        return cv2.resize(rgb, (FE_SIZE, FE_SIZE), interpolation=cv2.INTER_AREA)
    H, W = rgb.shape[:2]
    s = FE_SIZE / min(W, H)
    return cv2.resize(rgb, (int(round(W * s)), int(round(H * s))),
                      interpolation=cv2.INTER_AREA)


def evaluate(args):
    mpath = os.path.join(SETB_DIR, "manifest.jsonl")
    if not os.path.exists(mpath):
        sys.exit(f"no Set B recordings found. Run:  "
                 f"uv run python scripts/setb.py record --person <name>")
    with open(mpath, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    people = sorted({r["person"] for r in rows})

    print("=" * 78)
    print("SET B -- team webcam recordings (spec §6.1, the adoption criterion)")
    print("=" * 78)
    print(f"  {len(rows)} frames, {len(people)} person(s): {', '.join(people)}")
    print(f"  per emotion: " + "  ".join(
        f"{c}={sum(1 for r in rows if r['emotion'] == c)}" for c in CLASSES))
    if len(people) < 7:
        print(f"\n  ** n={len(people)} PERSON(S). Spec §6.1 asks for 7. Report every "
              f"number below\n  ** as provisional -- it can show a collapse, it "
              f"cannot show adequacy.")

    _init_worker()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(ART_DIR, "deliverable", "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    BIAS = np.array(meta["bias"]["value"], dtype=np.float32)

    results = {}
    for stretch in (True, False):
        name = f"{FE_SIZE}x{FE_SIZE} stretched" if stretch else f"{FE_SIZE} short side, aspect kept"
        X, y, fails = [], [], Counter()
        for r in rows:
            bgr = cv2.imread(os.path.join(SETB_DIR, r["path"]))
            if bgr is None:
                fails["unreadable"] += 1
                continue
            frame = fe_path(bgr, stretch)
            box, _ = _detect(frame)
            if box is None:
                fails["no_face"] += 1
                continue
            crop = _crop48(frame, box)
            if crop is None:
                fails["degenerate_box"] += 1
                continue
            X.append(crop)
            y.append(CLASSES.index(r["emotion"]))
        if not X:
            print(f"\n  {name}: no frames survived detection")
            continue
        X, y = np.stack(X), np.array(y)

        ck = torch.load(os.path.join(ART_DIR, "checkpoints", "best.pth"),
                        map_location="cpu", weights_only=False)
        ft = build_model(verbose=False)
        ft.load_state_dict(ck["model"])
        ft.to(device).eval()
        lp = predict(ft, X, device)
        o7 = predict(load_original7(device), X, device)
        pred = {"orig": o7[:, ORIG_COLS].argmax(1), "ft": lp.argmax(1),
                "ft_bias": (lp + BIAS).argmax(1)}

        cols = [("original 7c", "orig"), ("fine-tuned", "ft"), ("ft + bias", "ft_bias")]
        res = {k: score(y, pred[k]) for _, k in cols}
        results[name] = {"n": len(y), "detection_failures": dict(fails), "metrics": res}

        print(f"\n  --- FE transform: {name} ---")
        print(f"  detection failures: {sum(fails.values())}/{len(rows)} "
              f"{dict(fails) if fails else ''}")
        print(f"  {'metric':<26}" + "".join(f"{n:>16}" for n, _ in cols))
        for m in ("anxious_recall", "anxious_precision", "neutral_recall",
                  "neutral_precision", "happy_precision", "embarrassed_precision",
                  "accuracy", "macro_f1", "neutral_to_anxious"):
            print(f"  {m:<26}" + "".join(f"{res[k][m]:>16.4f}" for _, k in cols))

    out = os.path.join(ART_DIR, "setb_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"people": people, "n_people": len(people), "frames": len(rows),
                   "provisional": len(people) < 7, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  wrote {out}")
    if len(people) < 7:
        print("\n  Spec §6.1: if the fine-tuned model does not beat the original on "
              "Set B, do\n  not swap it in. With n<7 this run cannot clear that bar "
              "-- it can only fail it.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--person", required=True, help="identifier, kept out of the split")
    r.add_argument("--seconds", type=int, default=30)
    r.add_argument("--camera", type=int, default=0)
    sub.add_parser("eval")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    (record if args.cmd == "record" else evaluate)(args)


if __name__ == "__main__":
    main()
