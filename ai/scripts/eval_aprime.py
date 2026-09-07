"""
Step 4b: evaluation set A-prime -- the FE -> BE path, end to end.

Set A crops from the full-resolution image and *then* degrades. Serving does the
opposite: the frontend downscales the whole frame before transmission, so
MediaPipe runs on an already-degraded frame. The detector itself is degraded,
the box is less accurate, and the crop jitters. None of that reaches the Set A
numbers, so Set A is optimistic by an unknown amount. A-prime measures it.

  original image
    -> EXIF orientation (§2.1)
    -> resize the WHOLE FRAME to 224x224      <- this is what FE sends
    -> MediaPipe on the 224x224 frame          <- detection on degraded pixels
    -> crop, 0% margin (§2.3)
    -> 48x48 grayscale, /255 (§2.4)

Same test split (split.npy == 2), same people, same rows -- so every number is
comparable to Step 3's Set A line for line.

A control is measured alongside: the same pipeline with the short side scaled
to 224 and the aspect ratio preserved. Stretching a 4:3 frame into a square
distorts the face, and without the control there is no way to tell whether a
drop came from lost resolution or from that distortion.

Outputs (ai/artifacts/):
  aprime/X_aprime.npy        M x 48 x 48 uint8, the frames that detected
  aprime/rows.npy            index into the test split for each kept row
  aprime/_aprime_report.json detection failures, IoU distribution, metrics
  report/aprime_confusion.png

Run with:
  uv run python scripts/eval_aprime.py
  uv run python scripts/eval_aprime.py --reuse    # skip rebuilding the crops
"""
import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import SERVICE_PRIOR, score
from preprocess import FOLDERS, IMAGE_EXTS, MP_MIN_CONFIDENCE, MP_MODEL_SELECTION, iou
from train import ART_DIR, CLASSES, DATA_DIR, INPUT_SIZE, ORIG_COLS, build_model, load_original7

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "..", "data")
CACHE_DIR = os.path.join(RAW_DIR, "_cache_step2")
OUT_DIR = os.path.join(ART_DIR, "aprime")

FE_SIZE = 224          # spec §3.1: the frame size FE transmits
_FD = None


def _init_worker():
    global _FD
    import mediapipe as mp
    _FD = mp.solutions.face_detection.FaceDetection(
        model_selection=MP_MODEL_SELECTION, min_detection_confidence=MP_MIN_CONFIDENCE)


def _detect(rgb):
    """Highest-scoring detection (§2.3 multi-face rule) as pixel box + score."""
    det = _FD.process(rgb)
    if not det.detections:
        return None, None
    best = max(det.detections, key=lambda d: d.score[0])
    bb = best.location_data.relative_bounding_box
    h, w = rgb.shape[:2]
    return ([bb.xmin * w, bb.ymin * h, (bb.xmin + bb.width) * w,
             (bb.ymin + bb.height) * h], float(best.score[0]))


def _crop48(rgb, box):
    x0, y0 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    x1 = min(rgb.shape[1], int(round(box[2])))
    y1 = min(rgb.shape[0], int(round(box[3])))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    gray = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)


def _one(task):
    """(row, file, path, full_res_box) -> record + 48x48 crop bytes or None."""
    row, file, path, ref_box = task
    rec = {"row": row, "file": file, "ok": False}
    try:
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception as exc:
        rec["reason"] = "unreadable"
        rec["error"] = repr(exc)[:120]
        return rec, None, None

    rgb = np.asarray(pil)
    H, W = rgb.shape[:2]
    rec["src_size"] = [W, H]

    # --- the FE path: stretch the whole frame to 224x224 ---
    frame = cv2.resize(rgb, (FE_SIZE, FE_SIZE), interpolation=cv2.INTER_AREA)
    box, sc = _detect(frame)
    if box is None:
        rec["reason"] = "no_face_224"
    else:
        rec["score"] = sc
        rec["box_224"] = [round(v, 2) for v in box]
        rec["face_px_224"] = round(box[2] - box[0], 1)
        # Back-project to full-resolution pixels so the IoU against the Set A
        # box measures crop drift, not the coordinate change.
        proj = [box[0] * W / FE_SIZE, box[1] * H / FE_SIZE,
                box[2] * W / FE_SIZE, box[3] * H / FE_SIZE]
        rec["iou_vs_fullres"] = round(iou(proj, ref_box), 4)
        crop = _crop48(frame, box)
        if crop is None:
            rec["reason"] = "degenerate_box"
        else:
            rec["ok"] = True

    # --- control: same downscale, aspect ratio preserved ---
    s = FE_SIZE / min(W, H)
    ctrl_frame = cv2.resize(rgb, (max(1, int(round(W * s))), max(1, int(round(H * s)))),
                            interpolation=cv2.INTER_AREA)
    cbox, csc = _detect(ctrl_frame)
    ctrl = None
    if cbox is None:
        rec["ctrl_reason"] = "no_face_ctrl"
    else:
        rec["ctrl_score"] = csc
        rec["ctrl_face_px"] = round(cbox[2] - cbox[0], 1)
        cproj = [v / s for v in cbox]
        rec["ctrl_iou_vs_fullres"] = round(iou(cproj, ref_box), 4)
        ctrl = _crop48(ctrl_frame, cbox)
        rec["ctrl_ok"] = ctrl is not None

    return rec, (crop.tobytes() if rec["ok"] else None), (ctrl.tobytes() if ctrl is not None else None)


def full_res_boxes():
    """The Set A MediaPipe boxes, replayed from the Step 2 cache."""
    boxes = {}
    for cls in CLASSES:
        p = os.path.join(CACHE_DIR, f"{cls}.jsonl")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("ok"):
                    boxes[r["file"]] = r["box_px"]
    return boxes


def file_paths():
    out = {}
    for _, (_, source_dir, _) in FOLDERS.items():
        d = os.path.join(RAW_DIR, source_dir)
        for fn in os.listdir(d):
            if fn.lower().endswith(IMAGE_EXTS):
                out[fn] = os.path.join(d, fn)
    return out


@torch.no_grad()
def predict(model, X, device, batch=256):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch].astype(np.float32) / 255.0)
        out.append(model(x.unsqueeze(1).to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT_DIR, exist_ok=True)

    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    files = np.load(os.path.join(DATA_DIR, "files.npy"))
    split = np.load(os.path.join(DATA_DIR, "split.npy"))
    groups = np.load(os.path.join(DATA_DIR, "groups.npy"))
    test_rows = np.flatnonzero(split == 2)

    print("=" * 78)
    print("STEP 4b -- SET A' : evaluation through the real FE -> BE path")
    print("=" * 78)
    print(f"  test split: {len(test_rows)} frames, "
          f"{len(set(groups[test_rows]))} people (identical to Set A)")
    print(f"  FE path: whole frame -> {FE_SIZE}x{FE_SIZE} -> MediaPipe -> crop -> 48x48")

    rec_path = os.path.join(OUT_DIR, "_records.json")
    if args.reuse and os.path.exists(rec_path):
        with open(rec_path, encoding="utf-8") as f:
            records = json.load(f)
        Xa = np.load(os.path.join(OUT_DIR, "X_aprime.npy"))
        Xc = np.load(os.path.join(OUT_DIR, "X_aprime_ctrl.npy"))
        print("  reused cached A' crops")
    else:
        boxes, paths = full_res_boxes(), file_paths()
        tasks = []
        for r in test_rows:
            fn = str(files[r])
            assert fn in paths, f"source image for {fn} not found on disk"
            assert fn in boxes, f"{fn} has no Set A box in the Step 2 cache"
            tasks.append((int(r), fn, paths[fn], boxes[fn]))

        print(f"\n  detecting on {len(tasks)} frames with {args.workers} workers...")
        records, buf_a, buf_c = [], [], []
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker) as pool:
            for i, (rec, blob, cblob) in enumerate(
                    pool.map(_one, tasks, chunksize=8), 1):
                records.append(rec)
                if blob is not None:
                    buf_a.append(np.frombuffer(blob, np.uint8).reshape(INPUT_SIZE, INPUT_SIZE))
                if cblob is not None:
                    buf_c.append(np.frombuffer(cblob, np.uint8).reshape(INPUT_SIZE, INPUT_SIZE))
                if i % 500 == 0:
                    print(f"    {i}/{len(tasks)}", flush=True)
        Xa = np.stack(buf_a) if buf_a else np.empty((0, INPUT_SIZE, INPUT_SIZE), np.uint8)
        Xc = np.stack(buf_c) if buf_c else np.empty((0, INPUT_SIZE, INPUT_SIZE), np.uint8)
        np.save(os.path.join(OUT_DIR, "X_aprime.npy"), Xa)
        np.save(os.path.join(OUT_DIR, "X_aprime_ctrl.npy"), Xc)
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False)

    ok_rows = np.array([r["row"] for r in records if r["ok"]], dtype=np.int64)
    ctrl_rows = np.array([r["row"] for r in records if r.get("ctrl_ok")], dtype=np.int64)
    np.save(os.path.join(OUT_DIR, "rows.npy"), ok_rows)
    ya, yc = y[ok_rows], y[ctrl_rows]

    # ---- detection ----
    reasons = Counter(r.get("reason") for r in records if not r["ok"])
    fail = len(records) - len(ok_rows)
    cfail = len(records) - len(ctrl_rows)
    print("\n" + "=" * 78)
    print("DETECTION -- the service risk that exists regardless of which model ships")
    print("=" * 78)
    print(f"  full resolution (Set A, Step 2):  0 failures in 27,633 attempts "
          f"across the whole run.\n"
          f"    For these {len(records)} test rows the rate is 0/{len(records)} by "
          f"construction --\n    X.npy only ever contained frames that detected, so "
          f"that side cannot fail here.")
    print(f"  {FE_SIZE}x{FE_SIZE} stretched frame:      {fail}/{len(records)} failed "
          f"({fail / len(records) * 100:.2f}%)  {dict(reasons) if reasons else ''}")
    print(f"  {FE_SIZE} short side, aspect kept:   {cfail}/{len(records)} failed "
          f"({cfail / len(records) * 100:.2f}%)")
    print("  A frame with no face produces no prediction at all -- it is not a "
          "wrong answer,\n  it is a gap in the 5-second window the alert rule "
          "counts over.")

    ious = np.array([r["iou_vs_fullres"] for r in records if r["ok"]])
    cious = np.array([r["ctrl_iou_vs_fullres"] for r in records if r.get("ctrl_ok")])
    fpx = np.array([r["face_px_224"] for r in records if r["ok"]])
    scs = np.array([r["score"] for r in records if r["ok"]])
    fscs = np.array([r["score"] for r in records if "score" in r])
    print(f"\n  CROP JITTER -- IoU(A' box back-projected, full-resolution box)")
    print(f"    stretched 224:  p5 {np.percentile(ious, 5):.3f}  "
          f"p25 {np.percentile(ious, 25):.3f}  median {np.median(ious):.3f}  "
          f"p75 {np.percentile(ious, 75):.3f}  mean {ious.mean():.3f}")
    print(f"    aspect kept:    p5 {np.percentile(cious, 5):.3f}  "
          f"p25 {np.percentile(cious, 25):.3f}  median {np.median(cious):.3f}  "
          f"p75 {np.percentile(cious, 75):.3f}  mean {cious.mean():.3f}")
    print(f"    below 0.5 IoU:  {(ious < 0.5).mean() * 100:.1f}% stretched, "
          f"{(cious < 0.5).mean() * 100:.1f}% aspect kept")
    print(f"  face width in the 224 frame: p5 {np.percentile(fpx, 5):.0f}px  "
          f"median {np.median(fpx):.0f}px  p95 {np.percentile(fpx, 95):.0f}px  "
          f"(spec §3.1 predicted 80-120)")
    print(f"  detector confidence: median {np.median(scs):.3f} "
          f"(Set A full-resolution median was 0.94 on the §2.1 samples)")

    # ---- models ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta_path = os.path.join(ART_DIR, "deliverable", "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    BIAS = np.array(meta["bias"]["value"], dtype=np.float32)

    ck = torch.load(os.path.join(ART_DIR, "checkpoints", "best.pth"),
                    map_location="cpu", weights_only=False)
    ft = build_model(verbose=False)
    ft.load_state_dict(ck["model"])
    ft.to(device).eval()
    orig = load_original7(device)

    lp_a = predict(ft, Xa, device)
    o7_a = predict(orig, Xa, device)
    pred_a = {"orig": o7_a[:, ORIG_COLS].argmax(1), "ft": lp_a.argmax(1),
              "ft_bias": (lp_a + BIAS).argmax(1)}

    lp_c = predict(ft, Xc, device)
    o7_c = predict(orig, Xc, device)
    pred_c = {"orig": o7_c[:, ORIG_COLS].argmax(1), "ft": lp_c.argmax(1),
              "ft_bias": (lp_c + BIAS).argmax(1)}

    cols = [("original 7c", "orig"), ("fine-tuned", "ft"), ("ft + bias", "ft_bias")]
    keys = ["anxious_recall", "anxious_precision", "neutral_recall", "neutral_precision",
            "happy_precision", "embarrassed_precision", "accuracy", "macro_f1",
            "neutral_to_anxious", "anxious_call_rate", "anxious_calls_from_neutral"]

    out = {}
    for tag, yy, pp, label in (("aprime", ya, pred_a, f"A' -- stretched {FE_SIZE}"),
                               ("ctrl", yc, pred_c, f"A' control -- aspect preserved")):
        for prior_name, prior in (("uniform", None), ("service", SERVICE_PRIOR)):
            out[f"{tag}_{prior_name}"] = {k: score(yy, pp[k], prior) for _, k in cols}
        for prior_name in ("uniform", "service"):
            print("\n" + "=" * 78)
            print(f"{label}, {prior_name.upper()} PRIOR  (n={len(yy)})")
            print("=" * 78)
            print(f"  {'metric':<30}" + "".join(f"{n:>16}" for n, _ in cols))
            for m in keys:
                print(f"  {m:<30}" + "".join(
                    f"{out[f'{tag}_{prior_name}'][k][m]:>16.4f}" for _, k in cols))

    # ---- the number that matters: how much Set A was overstating ----
    setA = meta["results"]["step4a_bias"]
    print("\n" + "=" * 78)
    print("SET A vs SET A'  -- what the current evaluation was hiding")
    print("=" * 78)
    aprime_hdr = "Set A'"
    print(f"  {'metric':<24}{'model':>12}{'Set A':>10}{aprime_hdr:>10}{'delta':>9}")
    for m in ("anxious_recall", "anxious_precision", "accuracy", "macro_f1"):
        for name, k in cols:
            a = setA["service"][k][m]
            b = out["aprime_service"][k][m]
            print(f"  {m:<24}{name:>12}{a:>10.4f}{b:>10.4f}{b - a:>+9.4f}")
    print("\n  (service prior on both sides, so only the imaging path differs)")

    with open(os.path.join(OUT_DIR, "_aprime_report.json"), "w", encoding="utf-8") as f:
        json.dump({"fe_size": FE_SIZE, "n_test_rows": len(records),
                   "detection": {"aprime_failures": fail, "ctrl_failures": cfail,
                                 "reasons": dict(reasons)},
                   "iou_vs_fullres": {"stretched": {"mean": float(ious.mean()),
                                                    "median": float(np.median(ious)),
                                                    "p5": float(np.percentile(ious, 5)),
                                                    "frac_below_0.5": float((ious < 0.5).mean())},
                                      "aspect_kept": {"mean": float(cious.mean()),
                                                      "median": float(np.median(cious)),
                                                      "p5": float(np.percentile(cious, 5)),
                                                      "frac_below_0.5": float((cious < 0.5).mean())}},
                   "face_px_224": {"p5": float(np.percentile(fpx, 5)),
                                   "median": float(np.median(fpx)),
                                   "p95": float(np.percentile(fpx, 95))},
                   "detector_score_median": float(np.median(fscs)),
                   "metrics": out}, f, ensure_ascii=False, indent=2)

    meta["results"]["step4b_aprime"] = out
    meta["set_a_prime"] = {
        "what": "test split re-evaluated through the FE path: whole frame to "
                f"{FE_SIZE}x{FE_SIZE}, MediaPipe on the downscaled frame, crop, 48x48",
        "why": "Set A crops at full resolution and degrades afterwards, so the "
               "detector never sees degraded pixels. Serving does.",
        "detection_failure_rate": fail / len(records),
        "median_iou_vs_fullres_box": float(np.median(ious)),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_plot(ya, pred_a, out, os.path.join(ART_DIR, "report", "aprime_confusion.png"))
    print(f"\n  wrote aprime/X_aprime.npy, aprime/_aprime_report.json, "
          f"report/aprime_confusion.png")
    print("  updated deliverable/meta.json")


def save_plot(y, pred, out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (title, k) in zip(axes, (("original 7c", "orig"), ("fine-tuned", "ft"),
                                     ("ft + bias", "ft_bias"))):
        cm = np.array(out["aprime_uniform"][k]["confusion"]).round().astype(int)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
        ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"A' -- {title}", fontsize=10)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    main()
