"""
Step 1b: spot-check the axis-swap (orientation 8) and mirror (orientation 2) families.

The spec's §2.2 coordinate-frame conclusion ("label boxes are in the EXIF-applied
frame, use them as-is") was verified only on orientations 3 and 4. Orientation 8
swaps the W/H axes and 2 is a horizontal mirror, so both need confirming before
the full preprocessing run.

Procedure is spec Appendix B: load with exif_transpose, run MediaPipe, then compare
IoU(MediaPipe box, annot_A as-is) against IoU(MediaPipe box, annot_A inverted).
As-is winning decisively => the §2.2 conclusion holds for that family.

Orientation 0 (5,488 images, non-standard tag) is included as a supplementary group:
exif_transpose no-ops on it, so as-is and inverted are identical and only the
baseline check is meaningful there.

Run with: uv run python scripts/spotcheck_orientation.py
"""
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageOps

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FOLDERS = {
    "happy": ("[라벨]EMOIMG_기쁨_TRAIN", "[원천]EMOIMG_기쁨_TRAIN_01",
              "img_emotion_training_data(기쁨).json"),
    "embarrassed": ("[라벨]EMOIMG_당황_TRAIN", "[원천]EMOIMG_당황_TRAIN_01",
                    "img_emotion_training_data(당황).json"),
    "anxious": ("[라벨]EMOIMG_불안_TRAIN", "[원천]EMOIMG_불안_TRAIN_01",
                "img_emotion_training_data(불안).json"),
    "neutral": ("[라벨]EMOIMG_중립_TRAIN", "[원천]EMOIMG_중립_TRAIN_01",
                "img_emotion_training_data(중립).json"),
}

TARGET_ORIENTATIONS = [8, 2, 0]   # 0 is supplementary
N_PER_ORIENTATION = 20
BASELINE_LO, BASELINE_HI = 0.67, 0.80   # §2.2 baseline for normal images

random.seed(0)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area = lambda z: (z[2] - z[0]) * (z[3] - z[1])
    return inter / (area(a) + area(b) - inter)


def invert_box(box, ori, W, H):
    """Box in post-EXIF coords (W, H) -> pre-EXIF frame. Spec §2.2 table."""
    x0, y0, x1, y1 = box
    if   ori == 2: r = (W - x1, y0, W - x0, y1)
    elif ori == 3: r = (W - x1, H - y1, W - x0, H - y0)
    elif ori == 4: r = (x0, H - y1, x1, H - y0)
    elif ori == 5: r = (y0, x0, y1, x1)
    elif ori == 6: r = (y0, W - x1, y1, W - x0)
    elif ori == 7: r = (H - y1, W - x1, H - y0, W - x0)
    elif ori == 8: r = (H - y1, x0, H - y0, x1)
    else:          r = (x0, y0, x1, y1)
    return [min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3])]


def main():
    # ---- index label boxes by filename ----
    print("Indexing label JSON...")
    boxes = {}
    for cls, (label_dir, _, json_name) in FOLDERS.items():
        with open(os.path.join(DATA_DIR, label_dir, json_name), encoding="utf-8") as f:
            for rec in json.load(f):
                b = rec["annot_A"]["boxes"]
                boxes[rec["filename"]] = [b["minX"], b["minY"], b["maxX"], b["maxY"]]

    # ---- find images per target orientation ----
    print("Scanning for target orientations...")
    pool = defaultdict(list)
    for cls, (_, source_dir, _) in FOLDERS.items():
        source_path = os.path.join(DATA_DIR, source_dir)
        for fn in os.listdir(source_path):
            if not fn.lower().endswith(".jpg"):
                continue
            path = os.path.join(source_path, fn)
            try:
                ori = Image.open(path).getexif().get(274, 1)
            except Exception:
                continue
            if ori in TARGET_ORIENTATIONS and fn in boxes:
                pool[ori].append((cls, fn, path))

    for ori in TARGET_ORIENTATIONS:
        print(f"  orientation {ori}: {len(pool[ori])} images found (with label match)")

    import mediapipe as mp
    fd = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5)

    results = {}
    for ori in TARGET_ORIENTATIONS:
        sample = random.sample(pool[ori], min(N_PER_ORIENTATION, len(pool[ori])))
        rows = []
        no_detect = 0
        for cls, fn, path in sample:
            pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            det = fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not det.detections:
                no_detect += 1
                continue
            bb = det.detections[0].location_data.relative_bounding_box
            M = [bb.xmin * w, bb.ymin * h, (bb.xmin + bb.width) * w, (bb.ymin + bb.height) * h]
            A = boxes[fn]
            iou_asis = iou(M, A)
            iou_inv = iou(M, invert_box(A, ori, w, h))

            # how far the face sits from the mirror/rotation axis => discriminating power
            face_cx, face_cy = (A[0] + A[2]) / 2, (A[1] + A[3]) / 2
            off_x = abs(face_cx - w / 2) / w      # matters for horizontal mirror (ori 2)
            off_y = abs(face_cy - h / 2) / h
            rows.append({
                "class": cls, "file": fn, "size": (w, h),
                "iou_asis": iou_asis, "iou_inv": iou_inv,
                "off_x": off_x, "off_y": off_y,
                "score": det.detections[0].score[0],
            })
        results[ori] = {"rows": rows, "no_detect": no_detect, "sampled": len(sample)}

    # ---- report ----
    for ori in TARGET_ORIENTATIONS:
        r = results[ori]
        rows = r["rows"]
        tag = {8: "axis swap (90 CCW)", 2: "horizontal mirror", 0: "non-standard, treated as normal"}[ori]
        print("\n" + "=" * 76)
        print(f"ORIENTATION {ori} — {tag}")
        print("=" * 76)
        print(f"sampled {r['sampled']}, detected {len(rows)}, no-detection {r['no_detect']}")
        if not rows:
            continue

        asis = [x["iou_asis"] for x in rows]
        inv = [x["iou_inv"] for x in rows]
        print(f"\nIoU as-is    : mean {np.mean(asis):.3f}  median {np.median(asis):.3f}  "
              f"min {min(asis):.3f}  max {max(asis):.3f}")
        if ori != 0:
            print(f"IoU inverted : mean {np.mean(inv):.3f}  median {np.median(inv):.3f}  "
                  f"min {min(inv):.3f}  max {max(inv):.3f}")

        above_06 = sum(1 for v in asis if v > 0.6)
        in_baseline = sum(1 for v in asis if BASELINE_LO <= v <= BASELINE_HI)
        print(f"\nas-is > 0.6            : {above_06}/{len(rows)}")
        print(f"as-is in 0.67-0.80     : {in_baseline}/{len(rows)}  (§2.2 baseline band)")

        if ori != 0:
            decisive = sum(1 for x in rows if x["iou_asis"] > 0.6 and x["iou_inv"] < 0.3)
            inv_wins = sum(1 for x in rows if x["iou_inv"] > 0.6 and x["iou_asis"] < 0.3)
            print(f"as-is wins decisively  : {decisive}/{len(rows)}  (as-is>0.6 AND inverted<0.3)")
            print(f"inverted wins          : {inv_wins}/{len(rows)}")
            # discriminating power: for a horizontal mirror a centered face barely moves
            key = "off_x" if ori == 2 else "off_y"
            weak = sum(1 for x in rows if x[key] < 0.10)
            print(f"weak-discrimination    : {weak}/{len(rows)}  "
                  f"(face within 10% of the {'vertical' if ori == 2 else 'horizontal'} centre line)")

        print("\nper-image (as-is / inverted / detection score):")
        for x in sorted(rows, key=lambda z: z["iou_asis"]):
            flag = "  <-- LOW" if x["iou_asis"] < 0.6 else ""
            print(f"  {x['iou_asis']:.3f} / {x['iou_inv']:.3f} / {x['score']:.3f}  "
                  f"[{x['class']}] {x['file'][:44]}...{flag}")

        # verdict
        print()
        if ori == 0:
            if above_06 >= len(rows) * 0.9:
                print("VERDICT: consistent with upright pixels — labels align without transform.")
            else:
                print("VERDICT: INCONSISTENT — orientation-0 images may not be upright. Investigate.")
        else:
            if above_06 >= len(rows) * 0.9 and np.mean(asis) > np.mean(inv):
                print("VERDICT: as-is wins — §2.2 conclusion HOLDS for this family.")
            else:
                print("VERDICT: FAILED — do not start the preprocessing run. Escalate.")

    out = os.path.join(os.path.dirname(__file__), "..", "data", "_step1b_spotcheck.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
