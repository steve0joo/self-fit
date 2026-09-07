"""
Preprocessing quality analysis: the figures and metrics that evaluate the
Step 2 pipeline, written for ai/docs/02b-preprocessing-analysis.md.

preprocess.py measures IoU, detection score and box geometry for every image
but only summarises them in _step2_report.json -- the IoU distribution, which
is the number that says whether the >=0.4 quality filter is placed sensibly,
never surfaced anywhere. This script reads those per-image records back out of
the resume cache and turns them into the evidence.

Re-runs nothing: no detection, no cropping, no training. Reads
  data/_cache_step2/{class}.jsonl   every attempt, with iou/score/box_px
  data/processed/*.npy              the frozen arrays
  data/[label]/*.json               annot_A boxes, for the box comparison
and opens source JPEGs only for the ~33 images drawn in the two galleries.

Run with:
  uv run python scripts/analyze_preprocess.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
PROC_DIR = os.path.join(DATA_DIR, "processed")
CACHE_DIR = os.path.join(DATA_DIR, "_cache_step2")
OUT_DIR = os.path.join(HERE, "..", "artifacts", "report_preprocess")

CLASSES = ["happy", "embarrassed", "anxious", "neutral"]
CAP = 6800
IOU_MIN = 0.4
CACHE_SIZE = 160

# Same folder map as preprocess.py; only the label JSON and source dir are used.
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
KO2EN = {"기쁨": "happy", "당황": "embarrassed", "불안": "anxious", "중립": "neutral"}

CLASS_COLOR = {"happy": "#4fa64f", "embarrassed": "#e0a020",
               "anxious": "#c0392b", "neutral": "#7f8c8d"}
ACCENT = "#c0392b"
MUTED = "#5b8fa8"

PCT_KEYS = ["p0.1", "p1", "p5", "p25", "p50", "p75", "p95", "p99"]
PCT_QS = [0.1, 1, 5, 25, 50, 75, 95, 99]


def pctd(a, keys, qs):
    return dict(zip(keys, [round(float(v), 4) for v in np.percentile(a, qs)]))


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------

def load_attempts():
    """class -> (all attempts, the first CAP successes = the rows in X.npy).

    Slicing to CAP mirrors preprocess.py exactly: successes past the cap stay in
    the cache but were never assembled into the arrays, so counting them here
    would describe a dataset that does not exist.
    """
    out = {}
    for c in CLASSES:
        with open(os.path.join(CACHE_DIR, f"{c}.jsonl"), encoding="utf-8") as f:
            attempts = [json.loads(line) for line in f if line.strip()]
        out[c] = (attempts, [r for r in attempts if r.get("ok")][:CAP])
    return out


def load_annot_boxes():
    """filename -> annot_A box. First occurrence wins, as in preprocess.py."""
    boxes = {}
    for label_dir, _, json_name in FOLDERS.values():
        with open(os.path.join(DATA_DIR, label_dir, json_name), encoding="utf-8") as f:
            for r in json.load(f):
                if r["filename"] not in boxes:
                    b = r["annot_A"]["boxes"]
                    boxes[r["filename"]] = [b["minX"], b["minY"], b["maxX"], b["maxY"]]
    return boxes


def source_path(file):
    """Source JPEG for a filename. The folder is the uploader label embedded in
    the name, not the majority-vote class, so it is read off the filename."""
    return os.path.join(DATA_DIR, FOLDERS[KO2EN[file.split("_")[3]]][1], file)


def rejected_records(att):
    return sorted((r for c in CLASSES for r in att[c][0]
                   if not r.get("ok") and r.get("reason") == "low_iou"),
                  key=lambda r: r["iou"])


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------

def iou_metrics(att):
    """The distribution preprocess.py computes per image and then discards."""
    per_class = {}
    for c in CLASSES:
        v = np.array([r["iou"] for r in att[c][1]])
        per_class[c] = {
            "n": int(v.size),
            "min": round(float(v.min()), 4),
            "percentiles": pctd(v, PCT_KEYS, PCT_QS),
            "max": round(float(v.max()), 4),
            "mean": round(float(v.mean()), 4),
            "below_0.5": int((v < 0.5).sum()),
            "below_0.6": int((v < 0.6).sum()),
        }

    kept = np.concatenate([[r["iou"] for r in att[c][1]] for c in CLASSES])
    rej = np.array([r["iou"] for r in rejected_records(att)])
    return {
        "threshold": IOU_MIN,
        "kept_overall": {
            "n": int(kept.size),
            "min": round(float(kept.min()), 4),
            "percentiles": pctd(kept, PCT_KEYS, PCT_QS),
            "max": round(float(kept.max()), 4),
            "mean": round(float(kept.mean()), 4),
            "below_0.5": int((kept < 0.5).sum()),
            "below_0.6": int((kept < 0.6).sum()),
        },
        "per_class": per_class,
        "rejected": {
            "n": int(rej.size),
            "exactly_zero_overlap": int((rej == 0).sum()),
            "max": round(float(rej.max()), 4),
            "values": [round(float(v), 4) for v in rej],
        },
        # The gap between the worst kept and the best rejected image is what
        # makes 0.4 a safe choice rather than an arbitrary one.
        "separation_gap": {
            "worst_kept": round(float(kept.min()), 4),
            "best_rejected": round(float(rej.max()), 4),
            "width": round(float(kept.min() - rej.max()), 4),
        },
    }


def threshold_sensitivity(att, pool_sizes):
    """What a different threshold would have cost. The binding constraint is not
    crop quality but the 6,800 cap: anxious has only 97 spare images, so a
    threshold that rejects more than ~1.4% of its pool breaks class balance."""
    rows = []
    for t in [0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        row = {"threshold": t, "per_class": {}}
        for c in CLASSES:
            scored = np.array([r["iou"] for r in att[c][0] if "iou" in r])
            retain = float((scored >= t).mean())
            # Attempts are a person-round-robin prefix of the pool, so their
            # retention rate carries over to the untouched remainder of it.
            reachable = int(pool_sizes[c] * retain)
            row["per_class"][c] = {
                "retention_rate": round(retain, 5),
                "max_reachable": reachable,
                "fills_cap": reachable >= CAP,
            }
        row["all_classes_fill_cap"] = all(v["fills_cap"] for v in row["per_class"].values())
        rows.append(row)
    return rows


def detection_metrics(att, boxes):
    kept = [r for c in CLASSES for r in att[c][1]]
    score = np.array([r["score"] for r in kept])
    iou = np.array([r["iou"] for r in kept])
    min_side = np.array([min(r["box_px"][2] - r["box_px"][0],
                             r["box_px"][3] - r["box_px"][1]) for r in kept])
    mp_aspect = np.array([r["box_aspect"] for r in kept])
    mp_area = np.array([(r["box_px"][2] - r["box_px"][0]) *
                        (r["box_px"][3] - r["box_px"][1]) for r in kept], dtype=float)

    ann = np.array([boxes[r["file"]] for r in kept], dtype=float)
    aw, ah = ann[:, 2] - ann[:, 0], ann[:, 3] - ann[:, 1]
    ok = (aw > 0) & (ah > 0)

    return {
        "detection_score": {
            "mean": round(float(score.mean()), 4),
            "min": round(float(score.min()), 4),
            "percentiles": pctd(score, ["p1", "p5", "p50", "p95"], [1, 5, 50, 95]),
            "below_0.8": int((score < 0.8).sum()),
            "correlation_with_iou": round(float(np.corrcoef(score, iou)[0, 1]), 4),
        },
        # §3.2 headroom: a crop must already be bigger than the 160px cache, or
        # the degradation augmentation has nothing left to degrade.
        "crop_resolution": {
            "cache_size": CACHE_SIZE,
            "min_side_px": {
                "min": int(min_side.min()),
                "percentiles": pctd(min_side, ["p1", "p5", "p50", "p95"], [1, 5, 50, 95]),
                "max": int(min_side.max()),
            },
            "below_cache_size": int((min_side < CACHE_SIZE).sum()),
            "below_2x_cache_size": int((min_side < 2 * CACHE_SIZE).sum()),
        },
        # §2.3's stated reason for not cropping with the annotator boxes.
        "box_shape": {
            "mediapipe_aspect": {
                "mean": round(float(mp_aspect.mean()), 4),
                "percentiles": pctd(mp_aspect, ["p5", "p50", "p95"], [5, 50, 95]),
            },
            "annotator_aspect": {
                "mean": round(float((aw[ok] / ah[ok]).mean()), 4),
                "percentiles": pctd(aw[ok] / ah[ok], ["p5", "p50", "p95"], [5, 50, 95]),
            },
            "area_ratio_mp_over_annot": pctd(mp_area[ok] / (aw[ok] * ah[ok]),
                                             ["p5", "p50", "p95"], [5, 50, 95]),
        },
    }


def annotation_health(att, boxes):
    """Why the IoU filter actually fires. Most rejections are not detection
    errors: the annotator box is zero-area or a few pixels wide, so even a
    perfect detection scores IoU 0 against it."""
    w = np.array([b[2] - b[0] for b in boxes.values()], dtype=float)
    h = np.array([b[3] - b[1] for b in boxes.values()], dtype=float)
    degenerate_all = int(((w <= 0) | (h <= 0)).sum())

    buckets = Counter()
    detail = []
    for r in rejected_records(att):
        b = boxes[r["file"]]
        bw, bh = b[2] - b[0], b[3] - b[1]
        m = r["box_px"]
        mw, mh = m[2] - m[0], m[3] - m[1]
        if bw <= 0 or bh <= 0:
            kind = "annot_box_degenerate"
        elif bw < 0.25 * mw:
            kind = "annot_box_tiny"
        else:
            kind = "geometric_disagreement"
        buckets[kind] += 1
        detail.append({"file": r["file"], "kind": kind, "iou": r["iou"],
                       "score": round(r["score"], 4), "annot_wh": [int(bw), int(bh)],
                       "mp_wh": [int(mw), int(mh)]})

    attempted = [r["file"] for c in CLASSES for r in att[c][0]]
    aw = np.array([boxes[f][2] - boxes[f][0] for f in attempted], dtype=float)
    ah = np.array([boxes[f][3] - boxes[f][1] for f in attempted], dtype=float)
    kept = [r["file"] for c in CLASSES for r in att[c][1]]
    kw = np.array([boxes[f][2] - boxes[f][0] for f in kept], dtype=float)
    kh = np.array([boxes[f][3] - boxes[f][1] for f in kept], dtype=float)

    return {
        "label_set": {"records": len(boxes),
                      "degenerate_annot_a_boxes": degenerate_all,
                      "rate": round(degenerate_all / len(boxes), 6)},
        "attempted": {"n": len(attempted),
                      "degenerate_annot_a_boxes": int(((aw <= 0) | (ah <= 0)).sum())},
        "kept": {"n": len(kept),
                 "degenerate_annot_a_boxes": int(((kw <= 0) | (kh <= 0)).sum())},
        "rejection_causes": dict(buckets),
        "rejected_detail": detail,
    }


def composition_metrics():
    y = np.load(os.path.join(PROC_DIR, "y.npy"))
    w = np.load(os.path.join(PROC_DIR, "w.npy"))
    g = np.load(os.path.join(PROC_DIR, "groups.npy"))
    s = np.load(os.path.join(PROC_DIR, "split.npy"))
    names = ["train", "val", "test"]

    per_person = Counter(g.tolist())
    splits_of = {}
    for person, si in zip(g.tolist(), s.tolist()):
        splits_of.setdefault(person, set()).add(si)

    return {
        "n": int(y.size),
        "class_counts": {CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))},
        "split_counts": {n: int((s == i).sum()) for i, n in enumerate(names)},
        "class_by_split": {n: {CLASSES[ci]: int(((y == ci) & (s == i)).sum())
                               for ci in range(len(CLASSES))}
                           for i, n in enumerate(names)},
        "people_total": len(per_person),
        "people_per_split": {n: sum(1 for v in splits_of.values() if v == {i})
                             for i, n in enumerate(names)},
        "images_per_person": {"min": int(min(per_person.values())),
                              "median": int(np.median(list(per_person.values()))),
                              "max": int(max(per_person.values())),
                              "max_share": round(max(per_person.values()) / y.size, 5)},
        "weights": {"indoor_1.0": int((w == 1.0).sum()), "outdoor_0.5": int((w == 0.5).sum())},
        "outdoor_share_by_split": {n: round(float((w[s == i] == 0.5).mean()), 4)
                                   for i, n in enumerate(names)},
        "person_split_leakage": sum(1 for v in splits_of.values() if len(v) > 1),
    }


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------

def fig_iou(att, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    kept = np.concatenate([[r["iou"] for r in att[c][1]] for c in CLASSES])
    rej = np.array([r["iou"] for r in rejected_records(att)])

    ax = axes[0]
    ax.hist(kept, bins=np.arange(0, 1.005, 0.01), color=MUTED,
            label=f"kept  (n={kept.size:,})")
    ax.axvspan(rej.max(), kept.min(), color="#000000", alpha=0.07)
    ax.axvline(IOU_MIN, color=ACCENT, lw=1.6, ls="--", label=f"threshold {IOU_MIN}")
    ax.plot(rej, np.full(rej.size, 0), "x", color=ACCENT, ms=7, mew=1.6,
            label=f"rejected  (n={rej.size})", clip_on=False)
    ax.annotate(f"empty band {rej.max():.2f} – {kept.min():.2f}\n"
                f"nothing to reject near the threshold",
                xy=((rej.max() + kept.min()) / 2, ax.get_ylim()[1] * 0.30),
                xytext=(0.05, ax.get_ylim()[1] * 0.62), fontsize=8.5, color="#444",
                arrowprops={"arrowstyle": "->", "color": "#888", "lw": 1})
    ax.set_xlabel("IoU(MediaPipe box, annot_A box)")
    ax.set_ylabel("images")
    ax.set_xlim(-0.02, 1.0)
    ax.set_title("The 0.4 threshold falls in an empty band, not in the mass", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    data = [[r["iou"] for r in att[c][1]] for c in CLASSES]
    bp = ax.boxplot(data, orientation="horizontal", widths=0.6, showfliers=False,
                    patch_artist=True,
                    medianprops={"color": "#222"})
    for patch, c in zip(bp["boxes"], CLASSES):
        patch.set_facecolor(CLASS_COLOR[c])
        patch.set_alpha(0.55)
    for i, values in enumerate(data):
        ax.plot(min(values), i + 1, "|", color=ACCENT, ms=13, mew=1.8)
    ax.axvline(IOU_MIN, color=ACCENT, lw=1.4, ls="--")
    ax.set_yticks(range(1, len(CLASSES) + 1))
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("IoU   (box = quartiles, whisker = 1.5 IQR, | = class minimum)")
    ax.set_xlim(0.3, 1.0)
    ax.set_title("Identical across classes — no class-specific crop bias", fontsize=10)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "iou_distribution.png"), dpi=130)
    plt.close(fig)


def fig_threshold(rows, pool_sizes, plt):
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ts = [r["threshold"] for r in rows]
    for c in CLASSES:
        ax.plot(ts, [r["per_class"][c]["max_reachable"] for r in rows], "-o", ms=3.5,
                color=CLASS_COLOR[c], label=f"{c}  (pool {pool_sizes[c]:,})")
    safe = [r["threshold"] for r in rows if r["all_classes_fill_cap"]]
    ax.axvspan(min(safe), max(safe), color=MUTED, alpha=0.12)
    ax.axhline(CAP, color="#222", lw=1.2, ls=":", label=f"cap {CAP:,} / class")
    ax.axvline(IOU_MIN, color=ACCENT, lw=1.5, ls="--", label=f"chosen {IOU_MIN}")
    ax.annotate(f"all 4 classes still fill the cap:  {min(safe):.2f} – {max(safe):.2f}",
                xy=(max(safe) - 0.02, CAP * 1.35), ha="right", fontsize=8.5, color="#333")
    ax.set_xlabel("IoU threshold")
    ax.set_ylabel("images reachable in the class pool")
    ax.set_title("Threshold sensitivity — anxious, the smallest pool, is what constrains "
                 "the choice", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "iou_threshold_sensitivity.png"), dpi=130)
    plt.close(fig)


def fig_detection(att, boxes, plt):
    kept = [r for c in CLASSES for r in att[c][1]]
    score = np.array([r["score"] for r in kept])
    iou = np.array([r["iou"] for r in kept])
    min_side = np.array([min(r["box_px"][2] - r["box_px"][0],
                             r["box_px"][3] - r["box_px"][1]) for r in kept])
    mp_aspect = np.array([r["box_aspect"] for r in kept])
    ann = np.array([boxes[r["file"]] for r in kept], dtype=float)
    aw, ah = ann[:, 2] - ann[:, 0], ann[:, 3] - ann[:, 1]
    ok = (aw > 0) & (ah > 0)
    ann_aspect = aw[ok] / ah[ok]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))

    ax = axes[0][0]
    ax.hist(score, bins=60, color=MUTED)
    ax.axvline(0.5, color=ACCENT, ls="--", lw=1.5, label="min_detection_confidence 0.5")
    ax.axvline(float(score.mean()), color="#222", ls=":", lw=1.2,
               label=f"mean {score.mean():.4f}")
    ax.set_xlabel("MediaPipe detection score")
    ax.set_ylabel("images")
    ax.set_title(f"Detection confidence — min {score.min():.3f}, "
                 f"{int((score < 0.8).sum())} images below 0.8", fontsize=9.5)
    ax.legend(fontsize=8)

    ax = axes[0][1]
    hb = ax.hexbin(score, iou, gridsize=45, cmap="Blues", bins="log", mincnt=1)
    ax.axhline(IOU_MIN, color=ACCENT, ls="--", lw=1.4)
    ax.set_xlabel("detection score")
    ax.set_ylabel("IoU")
    ax.set_title(f"Score barely predicts IoU (r = {np.corrcoef(score, iou)[0, 1]:+.3f}) — "
                 "the two filters catch different things", fontsize=9.5)
    fig.colorbar(hb, ax=ax, label="images (log)")

    ax = axes[1][0]
    ax.hist(min_side, bins=80, color=MUTED)
    ax.axvline(CACHE_SIZE, color=ACCENT, ls="--", lw=1.6, label=f"cache {CACHE_SIZE}px")
    ax.axvline(48, color="#222", ls=":", lw=1.2, label="model input 48px")
    ax.set_xlabel("crop min side, source pixels")
    ax.set_ylabel("images")
    ax.set_title(f"Every crop clears the cache size — min {int(min_side.min())}px, "
                 f"{int((min_side < CACHE_SIZE).sum())} upscaled", fontsize=9.5)
    ax.legend(fontsize=8)

    ax = axes[1][1]
    bins = np.arange(0.4, 1.35, 0.01)
    ax.hist(ann_aspect, bins=bins, color="#e0a020", alpha=0.75,
            label=f"annot_A box  (median {np.median(ann_aspect):.2f})")
    ax.hist(mp_aspect, bins=bins, color=MUTED, alpha=0.85,
            label=f"MediaPipe box  (median {np.median(mp_aspect):.2f})")
    ax.set_xlabel("box aspect ratio (w / h)")
    ax.set_ylabel("images")
    ax.set_title("Why IoU tops out near 0.72: the two boxes are different shapes",
                 fontsize=9.5)
    ax.legend(fontsize=8)

    for row in axes:
        for ax in row:
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "detection_quality.png"), dpi=130)
    plt.close(fig)


def fig_composition(plt):
    y = np.load(os.path.join(PROC_DIR, "y.npy"))
    g = np.load(os.path.join(PROC_DIR, "groups.npy"))
    s = np.load(os.path.join(PROC_DIR, "split.npy"))
    w = np.load(os.path.join(PROC_DIR, "w.npy"))
    names = ["train", "val", "test"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # Shares, not counts: train is 8x val/test, so raw stacked bars would squash
    # the two splits whose balance actually needs checking.
    ax = axes[0]
    totals = np.array([int((s == si).sum()) for si in range(3)], dtype=float)
    bottom = np.zeros(3)
    for ci, c in enumerate(CLASSES):
        vals = np.array([int(((y == ci) & (s == si)).sum()) for si in range(3)], dtype=float)
        share = vals / totals * 100
        ax.bar(names, share, bottom=bottom, color=CLASS_COLOR[c], label=c)
        for xi in range(3):
            ax.text(xi, bottom[xi] + share[xi] / 2, f"{int(vals[xi]):,}\n{share[xi]:.1f}%",
                    ha="center", va="center", fontsize=8, color="white")
        bottom += share
    ax.axhline(25, color="#fff", lw=0.8, ls=":")
    for xi, tot in enumerate(totals):
        ax.text(xi, 101.5, f"n = {int(tot):,}  ({tot / y.size * 100:.1f}%)",
                ha="center", fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("share of the split (%)")
    ax.set_title("Class balance holds inside every split", fontsize=10)
    ax.legend(fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.28),
              frameon=False)

    ax = axes[1]
    counts = np.array(sorted(Counter(g.tolist()).values(), reverse=True))
    ax.bar(range(len(counts)), counts, color=MUTED, width=1.0)
    ax.axhline(float(np.median(counts)), color=ACCENT, ls="--", lw=1.3,
               label=f"median {int(np.median(counts))}")
    ax.set_xlabel(f"person, largest first  (n={len(counts)})")
    ax.set_ylabel("images")
    ax.set_title(f"Per-person contribution after round-robin capping\n"
                 f"largest = {counts.max()} images = {counts.max() / y.size * 100:.2f}% "
                 f"of the set", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    ind = np.array([int(((w == 1.0) & (s == si)).sum()) for si in range(3)])
    out = np.array([int(((w == 0.5) & (s == si)).sum()) for si in range(3)])
    ax.bar(names, ind, color=MUTED, label="indoor, weight 1.0")
    ax.bar(names, out, bottom=ind, color="#e0a020", label="outdoor, weight 0.5")
    for xi in range(3):
        ax.text(xi, ind[xi] + out[xi] / 2, f"{out[xi] / (ind[xi] + out[xi]) * 100:.1f}%",
                ha="center", va="center", fontsize=9, color="#333")
    ax.set_ylabel("images")
    ax.set_title("§3.4 sample weights — outdoor share even across splits", fontsize=10)
    ax.legend(fontsize=8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dataset_composition.png"), dpi=130)
    plt.close(fig)


def _draw_case(ax, rec, boxes, cv2, Image, ImageOps):
    """One panel: the source image with the MediaPipe box (red) and the annot_A
    box (orange) drawn on it. EXIF is applied here exactly as preprocess.py does
    it, so the boxes land where the pipeline actually put them."""
    try:
        pil = ImageOps.exif_transpose(Image.open(source_path(rec["file"]))).convert("RGB")
    except Exception:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "unreadable", ha="center", va="center", fontsize=8)
        return
    img = np.asarray(pil)
    h, w = img.shape[:2]
    scale = 420 / max(h, w)
    img = cv2.resize(img, (max(int(w * scale), 1), max(int(h * scale), 1)))
    a = [int(v * scale) for v in boxes[rec["file"]]]
    m = [int(v * scale) for v in rec["box_px"]]
    if a[2] - a[0] > 1 and a[3] - a[1] > 1:
        cv2.rectangle(img, (a[0], a[1]), (a[2], a[3]), (224, 160, 32), 2)
    else:
        # A degenerate annotator box has nothing to outline -- mark the point.
        cv2.drawMarker(img, (a[0], a[1]), (224, 160, 32), cv2.MARKER_TILTED_CROSS, 20, 2)
    cv2.rectangle(img, (m[0], m[1]), (m[2], m[3]), (192, 57, 43), 2)
    ax.imshow(img)
    ax.set_axis_off()


def fig_galleries(att, boxes, health, plt):
    import cv2
    from PIL import Image, ImageOps

    # ---- every image the IoU filter threw away ----
    rejected = rejected_records(att)
    kind_of = {d["file"]: d["kind"] for d in health["rejected_detail"]}
    short = {"annot_box_degenerate": "annot box empty",
             "annot_box_tiny": "annot box tiny",
             "geometric_disagreement": "boxes disagree"}
    cols = 6
    rows = (len(rejected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.5))
    flat = axes.ravel()
    for ax, rec in zip(flat, rejected):
        _draw_case(ax, rec, boxes, cv2, Image, ImageOps)
        ax.set_title(f"IoU {rec['iou']:.3f} · score {rec['score']:.2f}\n"
                     f"{short[kind_of[rec['file']]]}", fontsize=7.5)
    for ax in flat[len(rejected):]:
        ax.set_axis_off()
    fig.suptitle("Every image rejected by the IoU filter — red = MediaPipe box "
                 "(which found the face), orange = annot_A box\n"
                 "Most rejections are broken annotations, not detection failures",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, "rejected_gallery.png"), dpi=110)
    plt.close(fig)

    # ---- the lowest-IoU images that survived the filter ----
    n, cols = 12, 6
    worst = sorted((r for c in CLASSES for r in att[c][1]), key=lambda r: r["iou"])[:n]
    X = np.load(os.path.join(PROC_DIR, "X.npy"), mmap_mode="r")
    files = np.load(os.path.join(PROC_DIR, "files.npy"))
    row_of = {f: i for i, f in enumerate(files.tolist())}

    # Source and its resulting crop are stacked in adjacent rows, so a 12-wide
    # strip becomes two readable blocks instead of one unreadably wide one.
    blocks = n // cols
    fig, axes = plt.subplots(2 * blocks, cols, figsize=(cols * 2.6, blocks * 4.6),
                             gridspec_kw={"height_ratios": [3, 1.7] * blocks})
    for i, rec in enumerate(worst):
        block, col = divmod(i, cols)
        src_ax, crop_ax = axes[2 * block][col], axes[2 * block + 1][col]
        _draw_case(src_ax, rec, boxes, cv2, Image, ImageOps)
        src_ax.set_title(f"IoU {rec['iou']:.3f}", fontsize=8.5)
        crop_ax.imshow(X[row_of[rec["file"]]], cmap="gray", vmin=0, vmax=255)
        crop_ax.set_axis_off()
    fig.suptitle(f"The {n} lowest-IoU images that were KEPT — for each: both boxes on the "
                 "source, and underneath it the 160×160 crop that went into X.npy\n"
                 "Even at the bottom of the distribution the crop frames the right face",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "borderline_kept_gallery.png"), dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Reading the Step 2 attempt log, the label boxes and the frozen arrays...")
    att = load_attempts()
    boxes = load_annot_boxes()
    with open(os.path.join(PROC_DIR, "_step2_report.json"), encoding="utf-8") as f:
        pool_sizes = json.load(f)["pool_sizes"]

    print("Computing metrics...")
    m_iou = iou_metrics(att)
    m_thr = threshold_sensitivity(att, pool_sizes)
    m_det = detection_metrics(att, boxes)
    m_health = annotation_health(att, boxes)
    m_comp = composition_metrics()

    # The two properties the whole pipeline rests on -- fail loudly, not in a table.
    assert m_health["kept"]["degenerate_annot_a_boxes"] == 0, \
        "a kept row has a degenerate annotator box"
    assert m_comp["person_split_leakage"] == 0, "a person crosses splits"

    report = {
        "source": {
            "script": "ai/scripts/preprocess.py",
            "attempt_log": "ai/data/_cache_step2/{class}.jsonl",
            "step2_report": "ai/data/processed/_step2_report.json",
        },
        "iou": m_iou,
        "iou_threshold_sensitivity": m_thr,
        "detection": m_det,
        "annotation_health": m_health,
        "composition": m_comp,
    }
    with open(os.path.join(OUT_DIR, "_preprocess_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Rendering figures...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_iou(att, plt)
    fig_threshold(m_thr, pool_sizes, plt)
    fig_detection(att, boxes, plt)
    fig_composition(plt)
    fig_galleries(att, boxes, m_health, plt)

    k, gap = m_iou["kept_overall"], m_iou["separation_gap"]
    safe = [r["threshold"] for r in m_thr if r["all_classes_fill_cap"]]
    print(f"\n  IoU kept:    median {k['percentiles']['p50']}  p1 {k['percentiles']['p1']}  "
          f"min {k['min']}  ({k['below_0.5']} below 0.5)")
    print(f"  separation:  worst kept {gap['worst_kept']} vs best rejected "
          f"{gap['best_rejected']}  ->  empty band {gap['width']} wide")
    print(f"  rejections:  {m_iou['rejected']['n']} images, causes "
          f"{m_health['rejection_causes']}")
    print(f"  degenerate annotator boxes: {m_health['label_set']['degenerate_annot_a_boxes']} "
          f"in the label set, {m_health['attempted']['degenerate_annot_a_boxes']} reached "
          f"this run, {m_health['kept']['degenerate_annot_a_boxes']} survived")
    print(f"  threshold could be anywhere in {min(safe)}-{max(safe)} and still fill the cap")
    print(f"\n  {os.path.normpath(OUT_DIR)}")
    for name in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, name))
        print(f"    {name:<34} {size / 1024:>7.0f} KB")


if __name__ == "__main__":
    main()
