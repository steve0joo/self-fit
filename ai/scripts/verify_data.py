"""
Step 1 verification per CLAUDE.md handoff spec.
Checks the actual data at ai/data/ against the spec's assumptions BEFORE any
preprocessing code is written. Does not write any pipeline artifacts.

Run with: uv run python scripts/verify_data.py
"""
import json
import os
import random
from collections import Counter

from PIL import Image, ImageOps
import numpy as np
import cv2

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

FOLDERS = {
    "happy": ("기쁨", "[라벨]EMOIMG_기쁨_TRAIN", "[원천]EMOIMG_기쁨_TRAIN_01",
              "img_emotion_training_data(기쁨).json"),
    "embarrassed": ("당황", "[라벨]EMOIMG_당황_TRAIN", "[원천]EMOIMG_당황_TRAIN_01",
                    "img_emotion_training_data(당황).json"),
    "anxious": ("불안", "[라벨]EMOIMG_불안_TRAIN", "[원천]EMOIMG_불안_TRAIN_01",
                "img_emotion_training_data(불안).json"),
    "neutral": ("중립", "[라벨]EMOIMG_중립_TRAIN", "[원천]EMOIMG_중립_TRAIN_01",
                "img_emotion_training_data(중립).json"),
}

TARGET_KO2EN = {"기쁨": "happy", "당황": "embarrassed", "불안": "anxious", "중립": "neutral"}
EXCLUDE_KO2EN = {"분노": "angry", "상처": "hurt", "슬픔": "sad"}
UNKNOWN = "알수없음"

random.seed(0)


def majority_vote(record):
    """Apply spec §1.2 rules. Returns (outcome, majority_label_ko) where
    outcome in {"used", "excluded_unknown", "excluded_no_majority", "excluded_out_of_scope"}."""
    votes = [record["annot_A"]["faceExp"], record["annot_B"]["faceExp"], record["annot_C"]["faceExp"]]
    if any(v == UNKNOWN for v in votes):
        return "excluded_unknown", None
    counts = Counter(votes)
    label, n = counts.most_common(1)[0]
    if n < 2:
        return "excluded_no_majority", None
    if label in TARGET_KO2EN:
        return "used", label
    if label in EXCLUDE_KO2EN:
        return "excluded_out_of_scope", label
    return "excluded_other", label  # safety net for unexpected label strings


def main():
    report = {}

    # ---- 1 & 3: per-folder counts, JSON match rate, label-rule outcomes ----
    print("=" * 70)
    print("1) IMAGE COUNT vs JSON MATCH RATE, per folder")
    print("=" * 70)
    all_disk_files = {}   # class_en -> dict(filename -> full path)
    all_json = {}         # class_en -> list of records

    for class_en, (ko, label_dir, source_dir, json_name) in FOLDERS.items():
        json_path = os.path.join(DATA_DIR, label_dir, json_name)
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
        all_json[class_en] = records

        source_path = os.path.join(DATA_DIR, source_dir)
        # .jpeg is 1.5% of the source files (935 images) -- an earlier .jpg-only
        # filter dropped them silently, which mattered most for anxious.
        disk_files = {fn: os.path.join(source_path, fn) for fn in os.listdir(source_path)
                      if fn.lower().endswith((".jpg", ".jpeg"))}
        all_disk_files[class_en] = disk_files

        json_filenames = {r["filename"] for r in records}
        matched = json_filenames & disk_files.keys()
        unmatched_disk = disk_files.keys() - json_filenames

        print(f"\n[{class_en} / {ko}]")
        print(f"  JSON records:        {len(records)}")
        print(f"  Images on disk:      {len(disk_files)}")
        print(f"  Matched (disk∩json): {len(matched)}  "
              f"({len(matched)/len(records)*100:.1f}% of JSON records have an image on disk)")
        print(f"  Disk files with NO matching JSON entry: {len(unmatched_disk)}")

        report[class_en] = {
            "json_records": len(records),
            "disk_images": len(disk_files),
            "matched": len(matched),
            "match_rate_vs_json": len(matched) / len(records),
            "disk_files_without_json_entry": len(unmatched_disk),
        }

    # ---- 2: EXIF orientation distribution across full dataset (all files on disk) ----
    print("\n" + "=" * 70)
    print("2) EXIF ORIENTATION DISTRIBUTION (all images on disk)")
    print("=" * 70)
    exif_counter = Counter()
    exif_by_class = {c: Counter() for c in FOLDERS}
    total_files = 0
    unreadable = 0
    for class_en, disk_files in all_disk_files.items():
        for fn, path in disk_files.items():
            total_files += 1
            try:
                ori = Image.open(path).getexif().get(274, 1)
            except Exception:
                unreadable += 1
                continue
            exif_counter[ori] += 1
            exif_by_class[class_en][ori] += 1

    print(f"\nTotal images scanned: {total_files}  (unreadable: {unreadable})")
    print(f"{'Orientation':<12}{'Count':<10}{'Share':<10}")
    for ori, cnt in sorted(exif_counter.items(), key=lambda x: -x[1]):
        print(f"{ori:<12}{cnt:<10}{cnt/total_files*100:.2f}%")
    rot_mirror = sum(v for k, v in exif_counter.items() if k in {2, 3, 4, 5, 6, 7, 8})
    print(f"\nNon-normal (rotation/mirror family, ori != 1): {rot_mirror} "
          f"({rot_mirror/total_files*100:.2f}%)  [spec estimated 31% from 13 samples]")

    print("\nPer-class breakdown:")
    for class_en, ctr in exif_by_class.items():
        tot = sum(ctr.values())
        nonnorm = sum(v for k, v in ctr.items() if k != 1)
        print(f"  {class_en:<14} total={tot:<8} non-normal={nonnorm} ({nonnorm/tot*100:.2f}%)  dist={dict(ctr)}")

    report["exif"] = {
        "total_scanned": total_files,
        "unreadable": unreadable,
        "distribution": dict(exif_counter),
        "non_normal_rate": rot_mirror / total_files,
    }

    # ---- 3: final per-class counts after §1.2 label rules ----
    print("\n" + "=" * 70)
    print("3) LABEL RULE OUTCOMES (§1.2), applied to ALL JSON records per folder")
    print("=" * 70)
    label_report = {}
    for class_en, records in all_json.items():
        outcome_counts = Counter()
        used_label_counts = Counter()
        for r in records:
            outcome, label = majority_vote(r)
            outcome_counts[outcome] += 1
            if outcome == "used":
                used_label_counts[TARGET_KO2EN[label]] += 1
        print(f"\n[{class_en} folder, {len(records)} records]")
        for k, v in outcome_counts.items():
            print(f"  {k:<28} {v:>7}  ({v/len(records)*100:.1f}%)")
        print(f"  -> majority-vote class breakdown among 'used': {dict(used_label_counts)}")
        label_report[class_en] = {
            "total": len(records),
            "outcomes": dict(outcome_counts),
            "used_class_breakdown": dict(used_label_counts),
        }

    # Also: same label-rule outcomes restricted to records whose image is actually on disk
    print("\n" + "-" * 70)
    print("3b) Same, but restricted to records with an image ACTUALLY ON DISK")
    print("-" * 70)
    label_report_disk = {}
    for class_en, records in all_json.items():
        disk_files = all_disk_files[class_en]
        subset = [r for r in records if r["filename"] in disk_files]
        outcome_counts = Counter()
        used_label_counts = Counter()
        for r in subset:
            outcome, label = majority_vote(r)
            outcome_counts[outcome] += 1
            if outcome == "used":
                used_label_counts[TARGET_KO2EN[label]] += 1
        print(f"\n[{class_en} folder, {len(subset)} records with image on disk]")
        for k, v in outcome_counts.items():
            pct = v / len(subset) * 100 if subset else 0
            print(f"  {k:<28} {v:>7}  ({pct:.1f}%)")
        print(f"  -> majority-vote class breakdown among 'used': {dict(used_label_counts)}")
        label_report_disk[class_en] = {
            "total_with_image": len(subset),
            "outcomes": dict(outcome_counts),
            "used_class_breakdown": dict(used_label_counts),
        }

    report["label_rules_full_json"] = label_report
    report["label_rules_disk_only"] = label_report_disk

    # ---- 4: MediaPipe detection rate on 30 random images ----
    print("\n" + "=" * 70)
    print("4) MEDIAPIPE DETECTION RATE on 30 random images (model_selection=0, min_detection_confidence=0.5)")
    print("=" * 70)
    import mediapipe as mp
    fd = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    pool = []
    for class_en, disk_files in all_disk_files.items():
        for fn, path in disk_files.items():
            pool.append((class_en, path))
    sample = random.sample(pool, 30)

    n_detected = 0
    scores = []
    failures = []
    for class_en, path in sample:
        pil = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        result = fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if result.detections:
            n_detected += 1
            scores.append(result.detections[0].score[0])
        else:
            failures.append((class_en, os.path.basename(path)))

    print(f"\nDetected: {n_detected} / 30  ({n_detected/30*100:.1f}%)")
    if scores:
        print(f"Mean score: {sum(scores)/len(scores):.3f}  (min={min(scores):.3f}, max={max(scores):.3f})")
    if failures:
        print("Failures:")
        for c, fn in failures:
            print(f"  [{c}] {fn}")

    report["mediapipe_sample30"] = {
        "detected": n_detected,
        "total": 30,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "failures": [{"class": c, "file": fn} for c, fn in failures],
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "_step1_verification_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
