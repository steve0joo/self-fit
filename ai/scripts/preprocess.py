"""
Step 2: preprocessing. Turns the raw AI-Hub folders into the frozen array
artifacts that Step 3 (training) consumes. After this runs, the label JSON is
never needed again -- training reads only the .npy files.

Pipeline per sample, following the spec:
  load with EXIF applied (§2.1)  ->  majority-vote label (§1.2)
  ->  MediaPipe crop, 0% margin, highest-scoring detection (§2.3)
  ->  IoU(MediaPipe box, annot_A box) >= 0.4 quality filter (§2.3)
  ->  grayscale, cached at 160x160 (§3.2 -- NOT 48x48; degradation needs headroom)

Class balance is enforced by capping every class at 6,800 (§1.3, §4), and the
train/val/test split is by person and written once to split.npy -- training must
never recompute it.

Outputs (ai/data/processed/, all index-aligned, same N and same order):
  X.npy             N x 160 x 160 uint8 grayscale crops (source crops, not model input)
  y.npy             0=happy 1=embarrassed 2=anxious 3=neutral
  groups.npy        person ID (leading filename hash)
  w.npy             §3.4 sample weight (indoor 1.0 / outdoor 0.5)
  files.npy         source filename per row, for tracing misclassifications
  split.npy         0=train 1=val 2=test
  split_people.json person ID -> split, for eyeballing
  _step2_report.json  counts, drop reasons, parameters

Resumable: every attempt is appended to ai/data/_cache_step2/<class>.jsonl and
every kept crop to <class>.bin, so an interrupted run picks up where it stopped.

Run with:
  uv run python scripts/preprocess.py                 # full run
  uv run python scripts/preprocess.py --plan-only     # selection + split, no images
  uv run python scripts/preprocess.py --limit 200     # smoke test, 200/class
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

# Silence the mediapipe / TF-Lite banner in every worker process.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
OUT_DIR = os.path.join(DATA_DIR, "processed")
CACHE_DIR = os.path.join(DATA_DIR, "_cache_step2")

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

# Spec §5 output order. y.npy uses these indices.
CLASSES = ["happy", "embarrassed", "anxious", "neutral"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

TARGET_KO2EN = {"기쁨": "happy", "당황": "embarrassed", "불안": "anxious", "중립": "neutral"}
EXCLUDE_KO2EN = {"분노": "angry", "상처": "hurt", "슬픔": "sad"}
UNKNOWN = "알수없음"

# §3.4: outdoor-type backgrounds are further from the service environment
# (indoors at a desk) -- keep them, but at half sampling weight.
OUTDOOR_BG = {"실외 자연환경", "문화재 및 유적지", "스포츠 관람 및 레저시설", "도심 환경"}
WEIGHT_OUTDOOR = 0.5
WEIGHT_INDOOR = 1.0

SEED = 0
CAP_PER_CLASS = 6800          # §1.3 -- anxious is the binding class
CACHE_SIZE = 160              # §3.2 -- never 48; degradation downscales to 72-132 first
IOU_MIN = 0.4                 # §2.3 quality filter
MP_MODEL_SELECTION = 0        # §7.1 Q1: pending BE confirmation
MP_MIN_CONFIDENCE = 0.5       # §7.1 Q1: pending BE confirmation
SPLIT_RATIOS = [("train", 0.8), ("val", 0.1), ("test", 0.1)]
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}

IMAGE_EXTS = (".jpg", ".jpeg")
BYTES_PER_CROP = CACHE_SIZE * CACHE_SIZE

PREVIEW_PER_CLASS = 25
PREVIEW_COLS = 10
# BGR border colours so the preview grid is readable at a glance.
PREVIEW_COLORS = {0: (80, 200, 80), 1: (0, 170, 255), 2: (60, 60, 235), 3: (190, 190, 190)}


# ----------------------------------------------------------------------------
# label rules (§1.2)
# ----------------------------------------------------------------------------

def majority_vote(record):
    """Spec §1.2. Returns (outcome, majority_label_ko).

    outcome in {used, excluded_unknown, excluded_no_majority,
                excluded_out_of_scope, excluded_other}.
    The uploader label (= folder name = filename) is never used as ground truth.
    """
    votes = [record["annot_A"]["faceExp"],
             record["annot_B"]["faceExp"],
             record["annot_C"]["faceExp"]]
    if any(v == UNKNOWN for v in votes):
        return "excluded_unknown", None
    label, n = Counter(votes).most_common(1)[0]
    if n < 2:
        return "excluded_no_majority", None
    if label in TARGET_KO2EN:
        return "used", label
    if label in EXCLUDE_KO2EN:
        return "excluded_out_of_scope", label
    return "excluded_other", label


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area = lambda z: (z[2] - z[0]) * (z[3] - z[1])
    return inter / (area(a) + area(b) - inter)


# ----------------------------------------------------------------------------
# candidate pool
# ----------------------------------------------------------------------------

def build_pool():
    """Scan all 4 folders, apply §1.2, and bucket every usable image by its
    majority-vote class. A folder contributes to classes other than its own name
    -- e.g. 1,861 images in the anxious folder have a majority label of
    embarrassed -- so the pools are built across all folders, not per folder."""
    pool = {c: [] for c in CLASSES}
    folder_stats = {}
    seen_files = {}
    duplicates = []

    for folder_en, (label_dir, source_dir, json_name) in FOLDERS.items():
        source_path = os.path.join(DATA_DIR, source_dir)
        disk = {fn for fn in os.listdir(source_path) if fn.lower().endswith(IMAGE_EXTS)}
        with open(os.path.join(DATA_DIR, label_dir, json_name), encoding="utf-8") as f:
            records = json.load(f)

        outcomes = Counter()
        for r in records:
            fn = r["filename"]
            if fn not in disk:
                outcomes["no_image_on_disk"] += 1
                continue
            outcome, label = majority_vote(r)
            outcomes[outcome] += 1
            if outcome != "used":
                continue
            if fn in seen_files:
                duplicates.append(fn)
                continue
            seen_files[fn] = folder_en
            b = r["annot_A"]["boxes"]
            pool[TARGET_KO2EN[label]].append({
                "file": fn,
                "path": os.path.join(source_path, fn),
                "person": fn.split("_")[0],
                "box": [b["minX"], b["minY"], b["maxX"], b["maxY"]],
                "weight": WEIGHT_OUTDOOR if r["bg_uploader"] in OUTDOOR_BG else WEIGHT_INDOOR,
                "folder": folder_en,
            })

        folder_stats[folder_en] = {
            "json_records": len(records),
            "images_on_disk": len(disk),
            "outcomes": dict(outcomes),
        }

    return pool, folder_stats, duplicates


def person_round_robin(items, rng):
    """Order a class pool so that taking the first N spreads them across as many
    people as possible. Without this the 6,800 cap would be filled by whoever
    happens to sort first -- one person contributes up to 253 images to a class.
    """
    by_person = defaultdict(list)
    for it in items:
        by_person[it["person"]].append(it)
    persons = sorted(by_person)          # deterministic base order before shuffling
    rng.shuffle(persons)
    for p in persons:
        rng.shuffle(by_person[p])

    ordered = []
    depth = 0
    while True:
        added = False
        for p in persons:
            if depth < len(by_person[p]):
                ordered.append(by_person[p][depth])
                added = True
        if not added:
            return ordered
        depth += 1


# ----------------------------------------------------------------------------
# cropping (§2.1 load, §2.3 crop, §2.4 grayscale)
# ----------------------------------------------------------------------------

_FD = None


def _init_worker():
    global _FD
    import mediapipe as mp
    _FD = mp.solutions.face_detection.FaceDetection(
        model_selection=MP_MODEL_SELECTION,
        min_detection_confidence=MP_MIN_CONFIDENCE)


def _crop_one(task):
    """(file, path, annot_box) -> (log record, 160x160 uint8 bytes or None).

    Everything here stays in RGB. §2.4: PIL hands back RGB and OpenCV assumes
    BGR, and running COLOR_BGR2GRAY on an RGB array swaps the R and B luma
    weights (mean error 3.8, max 48 of 255) -- which hits faces hardest. So the
    array is RGB from PIL, MediaPipe is fed RGB, and grayscale uses RGB2GRAY.
    """
    file, path, annot_box = task
    rec = {"file": file, "ok": False}

    try:
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception as exc:
        rec["reason"] = "unreadable"
        rec["error"] = repr(exc)[:200]
        return rec, None

    rgb = np.asarray(pil)
    h, w = rgb.shape[:2]
    rec["src_size"] = [w, h]

    det = _FD.process(rgb)
    if not det.detections:
        rec["reason"] = "no_face"
        return rec, None

    best = max(det.detections, key=lambda d: d.score[0])      # §2.3 multi-face rule
    bb = best.location_data.relative_bounding_box
    mp_box = [bb.xmin * w, bb.ymin * h, (bb.xmin + bb.width) * w, (bb.ymin + bb.height) * h]
    rec["score"] = float(best.score[0])

    # 0% margin (§2.3), clipped to the frame -- MediaPipe boxes can run off-image.
    x0 = max(0, int(round(mp_box[0])))
    y0 = max(0, int(round(mp_box[1])))
    x1 = min(w, int(round(mp_box[2])))
    y1 = min(h, int(round(mp_box[3])))
    rec["clipped"] = bool(x0 != round(mp_box[0]) or y0 != round(mp_box[1])
                          or x1 != round(mp_box[2]) or y1 != round(mp_box[3]))
    rec["box_px"] = [x0, y0, x1, y1]
    if x1 - x0 < 2 or y1 - y0 < 2:
        rec["reason"] = "degenerate_box"
        return rec, None

    # §2.3 quality filter: a MediaPipe box that disagrees with the annotator box
    # means a different person's face was detected, or detection drifted.
    rec["iou"] = round(iou([x0, y0, x1, y1], annot_box), 4)
    if rec["iou"] < IOU_MIN:
        rec["reason"] = "low_iou"
        return rec, None

    crop = rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    upscaled = min(gray.shape[:2]) < CACHE_SIZE
    interp = cv2.INTER_CUBIC if upscaled else cv2.INTER_AREA
    resized = cv2.resize(gray, (CACHE_SIZE, CACHE_SIZE), interpolation=interp)

    rec["ok"] = True
    rec["upscaled"] = bool(upscaled)
    rec["box_aspect"] = round((x1 - x0) / (y1 - y0), 3)
    rec["box_w"] = x1 - x0
    return rec, resized.tobytes()


def _load_cache(cls):
    """Replay the append-only cache for one class. Writes go crop-then-log, so a
    crash can only leave extra crop bytes -- never a logged row without one."""
    bin_path = os.path.join(CACHE_DIR, f"{cls}.bin")
    log_path = os.path.join(CACHE_DIR, f"{cls}.jsonl")
    attempted, kept = [], []
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                attempted.append(rec)
                if rec.get("ok"):
                    kept.append(rec)

    have = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
    want = len(kept) * BYTES_PER_CROP
    if have > want:
        with open(bin_path, "r+b") as f:
            f.truncate(want)
        print(f"  [{cls}] trimmed {(have - want) // BYTES_PER_CROP} orphaned crop(s) from cache")
    elif have < want:
        # Should not happen given the write order, but recover rather than
        # silently pairing crops with the wrong rows.
        usable = have // BYTES_PER_CROP
        cut, seen = len(attempted), 0
        for i, rec in enumerate(attempted):
            if rec.get("ok"):
                seen += 1
                if seen > usable:
                    cut = i
                    break
        print(f"  [{cls}] cache log ran ahead of crops; rewinding "
              f"{len(attempted) - cut} attempt(s)")
        attempted = attempted[:cut]
        kept = [r for r in attempted if r.get("ok")]
        with open(log_path, "w", encoding="utf-8") as f:
            for rec in attempted:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return attempted, kept


def process_class(cls, ordered, cap, workers):
    """Crop candidates in order until `cap` succeed. Returns every attempt."""
    attempted, kept = _load_cache(cls)
    if attempted:
        print(f"  [{cls}] resuming: {len(attempted)} attempted, {len(kept)} kept in cache")

    done = {rec["file"] for rec in attempted}
    todo = [c for c in ordered if c["file"] not in done]
    if len(kept) >= cap or not todo:
        print(f"  [{cls}] nothing to do ({len(kept)}/{cap} kept)")
        return attempted

    bin_path = os.path.join(CACHE_DIR, f"{cls}.bin")
    log_path = os.path.join(CACHE_DIR, f"{cls}.jsonl")
    batch_size = max(64, workers * 32)
    t0 = time.time()
    processed = 0

    binf = open(bin_path, "ab")
    logf = open(log_path, "a", encoding="utf-8")
    pool_ctx = None
    try:
        if workers > 1:
            pool_ctx = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker)
            runner = lambda tasks: pool_ctx.map(_crop_one, tasks, chunksize=4)
        else:
            _init_worker()
            runner = lambda tasks: map(_crop_one, tasks)

        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            tasks = [(c["file"], c["path"], c["box"]) for c in batch]
            for rec, blob in runner(tasks):
                if blob is not None:
                    binf.write(blob)
                    binf.flush()
                logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logf.flush()
                attempted.append(rec)
                if rec.get("ok"):
                    kept.append(rec)
                processed += 1

            rate = processed / max(time.time() - t0, 1e-6)
            eta = (cap - len(kept)) / max(rate, 1e-6)
            print(f"  [{cls}] kept {len(kept)}/{cap}  attempted {len(attempted)}  "
                  f"{rate:.1f} img/s  eta {eta / 60:.1f} min", flush=True)
            if len(kept) >= cap:
                break
    except KeyboardInterrupt:
        print(f"\n  [{cls}] interrupted -- cache is consistent, rerun to resume")
        raise
    finally:
        binf.close()
        logf.close()
        if pool_ctx is not None:
            pool_ctx.shutdown(wait=True)

    return attempted


# ----------------------------------------------------------------------------
# person-level split (§4)
# ----------------------------------------------------------------------------

def assign_splits(persons_classes):
    """persons_classes: person -> Counter(class_idx -> n). Greedy assignment of
    whole people to splits, largest first, minimising squared per-class fill
    ratio. Assigning people (not samples) is what makes leakage impossible: a
    random split puts different frames of the same person in train and test and
    inflates reported performance."""
    totals = Counter()
    for counts in persons_classes.values():
        totals.update(counts)

    targets = {name: {c: max(totals[c] * ratio, 1e-9) for c in range(len(CLASSES))}
               for name, ratio in SPLIT_RATIOS}
    current = {name: Counter() for name, _ in SPLIT_RATIOS}

    order = sorted(persons_classes,
                   key=lambda p: (-sum(persons_classes[p].values()), p))
    assignment = {}
    for person in order:
        counts = persons_classes[person]
        best_name, best_cost = None, None
        for name, _ in SPLIT_RATIOS:                       # fixed order breaks ties
            cost = sum(((current[name][c] + counts[c]) / targets[name][c]) ** 2
                       for c in range(len(CLASSES)))
            if best_cost is None or cost < best_cost:
                best_name, best_cost = name, cost
        assignment[person] = best_name
        current[best_name].update(counts)

    return assignment, current


# ----------------------------------------------------------------------------
# preview grids
# ----------------------------------------------------------------------------

def save_preview(X, y, rng, out_dir):
    """Numbers cannot show an upside-down face, a crop that drifted into
    background, or the wrong person's face being detected. The grid can."""
    picks = []
    for ci in range(len(CLASSES)):
        idx = np.flatnonzero(y == ci)
        if len(idx) == 0:
            continue
        take = min(PREVIEW_PER_CLASS, len(idx))
        picks.extend(sorted(rng.sample(list(idx), take)))

    def grid(tile_px, downscale_to=None):
        tiles = []
        for i in picks:
            img = X[i]
            if downscale_to:
                img = cv2.resize(img, (downscale_to, downscale_to), interpolation=cv2.INTER_AREA)
                img = cv2.resize(img, (tile_px, tile_px), interpolation=cv2.INTER_NEAREST)
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            tiles.append(cv2.copyMakeBorder(bgr, 3, 3, 3, 3, cv2.BORDER_CONSTANT,
                                            value=PREVIEW_COLORS[int(y[i])]))
        rows = []
        for r in range(0, len(tiles), PREVIEW_COLS):
            row = tiles[r:r + PREVIEW_COLS]
            while len(row) < PREVIEW_COLS:
                row.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(row))
        return np.vstack(rows)

    p160 = os.path.join(out_dir, "_preview_160.png")
    p48 = os.path.join(out_dir, "_preview_48_as160.png")
    cv2.imwrite(p160, grid(CACHE_SIZE))
    cv2.imwrite(p48, grid(CACHE_SIZE, downscale_to=48))
    return p160, p48


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--cap", type=int, default=CAP_PER_CLASS)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: cap per class, smoke-test sized")
    ap.add_argument("--plan-only", action="store_true",
                    help="selection + split preview only, no image processing")
    ap.add_argument("--fresh", action="store_true", help="ignore the resume cache")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cap = args.limit or args.cap

    print("=" * 76)
    print("STEP 2 PREPROCESSING")
    print("=" * 76)
    print(f"seed={SEED}  cap/class={cap}  cache={CACHE_SIZE}x{CACHE_SIZE} gray  "
          f"IoU>={IOU_MIN}  mediapipe(ms={MP_MODEL_SELECTION}, conf={MP_MIN_CONFIDENCE})  "
          f"workers={args.workers}")

    # ---- 1. candidate pool ----
    print("\n[1/5] Building candidate pool (§1.2 majority vote)...")
    pool, folder_stats, duplicates = build_pool()
    for folder_en, st in folder_stats.items():
        used = st["outcomes"].get("used", 0)
        on_disk = st["json_records"] - st["outcomes"].get("no_image_on_disk", 0)
        print(f"  {folder_en:<12} on disk {on_disk:>6} / json {st['json_records']:>6}  "
              f"-> used {used:>6}  " +
              "  ".join(f"{k.replace('excluded_', '-')}={v}"
                        for k, v in sorted(st["outcomes"].items())
                        if k.startswith("excluded")))
    if duplicates:
        print(f"  WARNING: {len(duplicates)} filename(s) appear in more than one folder; "
              f"kept the first occurrence")

    print("\n  class pools (majority vote across all 4 folders):")
    for c in CLASSES:
        people = len({it["person"] for it in pool[c]})
        outdoor = sum(1 for it in pool[c] if it["weight"] == WEIGHT_OUTDOOR)
        headroom = len(pool[c]) - cap
        note = "  <-- binding class, little headroom" if 0 <= headroom < 300 else ""
        print(f"    {c:<12} {len(pool[c]):>6} images  {people:>3} people  "
              f"outdoor {outdoor / max(len(pool[c]), 1) * 100:.1f}%  "
              f"headroom over cap {headroom:+}{note}")

    # ---- 2. ordering ----
    print(f"\n[2/5] Ordering candidates person-round-robin (cap {cap}/class)...")
    rng = random.Random(SEED)
    ordered = {}
    for c in CLASSES:
        ordered[c] = person_round_robin(pool[c], random.Random(SEED + CLASS_IDX[c]))
        head = Counter(it["person"] for it in ordered[c][:cap])
        print(f"    {c:<12} first {min(cap, len(ordered[c]))}: {len(head)} people, "
              f"max {max(head.values())} from one person "
              f"(uncapped max would be {max(Counter(it['person'] for it in pool[c]).values())})")

    if args.plan_only:
        print("\n[plan-only] simulating split on pre-detection candidates...")
        pc = defaultdict(Counter)
        for c in CLASSES:
            for it in ordered[c][:cap]:
                pc[it["person"]][CLASS_IDX[c]] += 1
        _, current = assign_splits(pc)
        for name, _ in SPLIT_RATIOS:
            tot = sum(current[name].values())
            print(f"    {name:<6} {tot:>6} samples  " +
                  "  ".join(f"{CLASSES[i]}={current[name][i]}" for i in range(len(CLASSES))))
        print("\nPlan only -- no images processed, no artifacts written.")
        return

    # ---- 3. crop ----
    if args.fresh:
        for c in CLASSES:
            for suffix in (".bin", ".jsonl"):
                path = os.path.join(CACHE_DIR, c + suffix)
                if os.path.exists(path):
                    os.remove(path)
        print("\n  --fresh: cleared the resume cache")

    print(f"\n[3/5] Cropping (§2.1 EXIF -> §2.3 MediaPipe 0% margin -> {CACHE_SIZE}px gray)...")
    attempts = {}
    for c in CLASSES:
        attempts[c] = process_class(c, ordered[c], cap, args.workers)

    # ---- 4. assemble ----
    print("\n[4/5] Assembling arrays...")
    drop_report = {}
    kept_rows = {}
    for c in CLASSES:
        succeeded = [r for r in attempts[c] if r.get("ok")]
        kept = succeeded[:cap]                      # surplus past the cap is cached, not used
        kept_rows[c] = kept
        reasons = Counter(r.get("reason", "?") for r in attempts[c] if not r.get("ok"))
        failed = sum(reasons.values())
        drop_report[c] = {
            "attempted": len(attempts[c]),
            "succeeded": len(succeeded),
            "kept": len(kept),
            "surplus_beyond_cap": len(succeeded) - len(kept),
            "dropped": dict(reasons),
            "failure_rate": round(failed / max(len(attempts[c]), 1), 5),
        }
        short = "" if len(kept) >= cap else f"  <-- SHORT of cap by {cap - len(kept)}"
        print(f"    {c:<12} kept {len(kept):>5} / attempted {len(attempts[c]):>5}  "
              f"failed {failed} ({failed / max(len(attempts[c]), 1) * 100:.2f}%): "
              f"{dict(reasons) or '{}'}{short}")

    meta_by_file = {}
    for c in CLASSES:
        for it in pool[c]:
            meta_by_file[it["file"]] = it

    total = sum(len(kept_rows[c]) for c in CLASSES)
    X = np.empty((total, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
    y = np.empty(total, dtype=np.int64)
    w = np.empty(total, dtype=np.float32)
    groups, files = [], []

    row = 0
    for c in CLASSES:
        n = len(kept_rows[c])
        if n == 0:
            continue
        blob = np.fromfile(os.path.join(CACHE_DIR, f"{c}.bin"),
                           dtype=np.uint8, count=n * BYTES_PER_CROP)
        X[row:row + n] = blob.reshape(n, CACHE_SIZE, CACHE_SIZE)
        y[row:row + n] = CLASS_IDX[c]
        for i, rec in enumerate(kept_rows[c]):
            it = meta_by_file[rec["file"]]
            w[row + i] = it["weight"]
            groups.append(it["person"])
            files.append(rec["file"])
        row += n

    groups = np.array(groups)
    files = np.array(files)

    # ---- 5. split + write ----
    print("\n[5/5] Freezing the person-level split (§4)...")
    persons_classes = defaultdict(Counter)
    for person, cls_idx in zip(groups, y):
        persons_classes[person][int(cls_idx)] += 1
    assignment, current = assign_splits(persons_classes)

    split = np.array([SPLIT_CODE[assignment[p]] for p in groups], dtype=np.int64)

    # The whole point of splitting by person: verify it actually held.
    by_person_split = defaultdict(set)
    for person, s in zip(groups, split):
        by_person_split[person].add(int(s))
    crossing = [p for p, s in by_person_split.items() if len(s) > 1]
    assert not crossing, f"person(s) in more than one split: {crossing[:5]}"

    print(f"    {len(persons_classes)} people -> " +
          ", ".join(f"{name} {sum(1 for v in assignment.values() if v == name)}"
                    for name, _ in SPLIT_RATIOS))
    print(f"    {'split':<8}{'n':>7}{'share':>8}   " +
          "".join(f"{c:>13}" for c in CLASSES))
    for name, ratio in SPLIT_RATIOS:
        n = sum(current[name].values())
        print(f"    {name:<8}{n:>7}{n / total * 100:>7.1f}%   " +
              "".join(f"{current[name][i]:>13}" for i in range(len(CLASSES))))
        for i in range(len(CLASSES)):
            assert current[name][i] > 0, f"{name} split has no {CLASSES[i]} samples"

    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUT_DIR, "w.npy"), w)
    np.save(os.path.join(OUT_DIR, "groups.npy"), groups)
    np.save(os.path.join(OUT_DIR, "files.npy"), files)
    np.save(os.path.join(OUT_DIR, "split.npy"), split)
    with open(os.path.join(OUT_DIR, "split_people.json"), "w", encoding="utf-8") as f:
        json.dump(dict(sorted(assignment.items())), f, ensure_ascii=False, indent=2)

    p160, p48 = save_preview(X, y, random.Random(SEED), OUT_DIR)

    kept_all = [r for c in CLASSES for r in kept_rows[c]]
    aspects = [r["box_aspect"] for r in kept_all if "box_aspect" in r]
    widths = [r["box_w"] for r in kept_all if "box_w" in r]
    scores = [r["score"] for r in kept_all if "score" in r]
    report = {
        "spec": "ai/docs/emotion-finetune-spec.md",
        "params": {
            "seed": SEED, "cap_per_class": cap, "cache_size": CACHE_SIZE,
            "crop_margin": 0.0, "iou_min": IOU_MIN,
            "mediapipe": {"model_selection": MP_MODEL_SELECTION,
                          "min_detection_confidence": MP_MIN_CONFIDENCE,
                          "note": "pending BE confirmation, spec §7.1 Q1"},
            "resize": "stretch to 160x160 (INTER_AREA down / INTER_CUBIC up); "
                      "spec §7.2 cropImages.py padding-vs-stretch still unconfirmed",
            "grayscale": "cv2.COLOR_RGB2GRAY on the PIL RGB array (spec §2.4 channel order)",
            "class_order": CLASSES,
            "split_ratios": dict(SPLIT_RATIOS),
            "outdoor_backgrounds": sorted(OUTDOOR_BG),
        },
        "folders": folder_stats,
        "pool_sizes": {c: len(pool[c]) for c in CLASSES},
        "duplicate_filenames": len(duplicates),
        "per_class": drop_report,
        # spec §7.2: "measure MediaPipe detection failure rate on the full run"
        "failures_overall": {
            "attempted": sum(d["attempted"] for d in drop_report.values()),
            "by_reason": dict(sum((Counter(d["dropped"]) for d in drop_report.values()),
                                  Counter())),
            "rate": round(sum(sum(d["dropped"].values()) for d in drop_report.values())
                          / max(sum(d["attempted"] for d in drop_report.values()), 1), 5),
        },
        "total_samples": total,
        "people": {
            "total": len(persons_classes),
            "per_split": {name: sum(1 for v in assignment.values() if v == name)
                          for name, _ in SPLIT_RATIOS},
        },
        "split_counts": {name: {CLASSES[i]: current[name][i] for i in range(len(CLASSES))}
                         for name, _ in SPLIT_RATIOS},
        "weights": {"indoor_1.0": int((w == WEIGHT_INDOOR).sum()),
                    "outdoor_0.5": int((w == WEIGHT_OUTDOOR).sum())},
        "mediapipe_box": {
            "aspect_median": round(float(np.median(aspects)), 3) if aspects else None,
            "width_px_p5_p50_p95": [int(np.percentile(widths, q)) for q in (5, 50, 95)] if widths else None,
            "score_mean": round(float(np.mean(scores)), 4) if scores else None,
            "clipped_to_frame": sum(1 for r in kept_all if r.get("clipped")),
            "upscaled_to_cache": sum(1 for r in kept_all if r.get("upscaled")),
        },
    }
    with open(os.path.join(OUT_DIR, "_step2_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  X.npy {X.shape} {X.nbytes / 1e6:.0f} MB  ->  {OUT_DIR}")
    print(f"  weights: indoor {report['weights']['indoor_1.0']} / "
          f"outdoor {report['weights']['outdoor_0.5']}")
    b = report["mediapipe_box"]
    print(f"  MediaPipe box: aspect median {b['aspect_median']}, "
          f"width p5/p50/p95 {b['width_px_p5_p50_p95']}, mean score {b['score_mean']}, "
          f"clipped {b['clipped_to_frame']}, upscaled to cache {b['upscaled_to_cache']}")
    print(f"\n  OPEN THESE BEFORE TRAINING:\n    {p160}\n    {p48}")
    print("  Border colour = class: green happy / orange embarrassed / "
          "red anxious / grey neutral")
    print("  Look for upside-down faces, crops drifting into background, and the "
          "wrong person's face.")


if __name__ == "__main__":
    main()
