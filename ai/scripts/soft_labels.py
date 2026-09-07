"""
Step 4c, part 1: build soft targets from the three annotator votes.

Step 2 collapsed annot_A/B/C into one majority label and threw the disagreement
away. That disagreement is not noise to be discarded -- it is concentrated
exactly on the embarrassed<->anxious and neutral<->anxious boundaries, which is
where the confusion matrix actually leaks (spec §1.3: the anxious folder is 37.8%
pure with 26.7% three-way disagreement, versus 0.9-9.9% elsewhere).

    hard:  anxious                    -> [0, 0, 1, 0]
    soft:  2 anxious + 1 embarrassed  -> [0, 0.33, 0.67, 0]

The §1.2 exclusion rules do not move: the same rows are kept, in the same order,
with the same split. Only the target changes.

One case the spec does not cover: a kept sample can still carry a vote for an
out-of-scope class (2 anxious + 1 angry). Those votes are dropped and the
remainder renormalised, so the target stays a distribution over the 4 classes.
4,397 of the 27,200 kept rows (16.2%) are in that situation; note that this
makes some of them one-hot again, since a 2-1 split with the 1 dropped is
unanimous among what remains.

Output: ai/data/processed/y_soft.npy, N x 4 float32, row-aligned with y.npy.

Run with:
  uv run python scripts/soft_labels.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import (CLASSES, CLASS_IDX, DATA_DIR, EXCLUDE_KO2EN, FOLDERS,
                        OUT_DIR, TARGET_KO2EN, UNKNOWN)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    files = np.load(os.path.join(OUT_DIR, "files.npy"))
    y = np.load(os.path.join(OUT_DIR, "y.npy"))
    want = {str(f): i for i, f in enumerate(files)}

    print("=" * 76)
    print("STEP 4c -- SOFT TARGETS FROM THE 3 ANNOTATORS")
    print("=" * 76)
    print(f"  {len(files)} kept rows; reading the label JSON for their votes")

    soft = np.zeros((len(files), len(CLASSES)), dtype=np.float32)
    found = np.zeros(len(files), dtype=bool)
    vote_shape = Counter()

    for _, (label_dir, _, json_name) in FOLDERS.items():
        with open(os.path.join(DATA_DIR, label_dir, json_name), encoding="utf-8") as f:
            records = json.load(f)
        for r in records:
            i = want.get(r["filename"])
            if i is None or found[i]:
                continue
            votes = [r["annot_A"]["faceExp"], r["annot_B"]["faceExp"],
                     r["annot_C"]["faceExp"]]
            assert UNKNOWN not in votes, f"{r['filename']} kept despite an unknown vote"
            in_scope = [v for v in votes if v in TARGET_KO2EN]
            dropped = [v for v in votes if v in EXCLUDE_KO2EN]
            for v in in_scope:
                soft[i, CLASS_IDX[TARGET_KO2EN[v]]] += 1.0
            soft[i] /= soft[i].sum()
            found[i] = True
            vote_shape[(len(in_scope), len(dropped))] += 1

    assert found.all(), f"{(~found).sum()} kept rows had no matching JSON record"

    # The hard label must still be the soft target's argmax -- if it is not,
    # the majority vote and the vote histogram disagree and something is wrong.
    assert (soft.argmax(1) == y).all(), "soft target argmax does not match y.npy"

    top = soft.max(1)
    print(f"\n  vote composition of the kept rows:")
    print(f"    unanimous in-scope (target is one-hot)    "
          f"{int((top == 1.0).sum()):>7}  {(top == 1.0).mean() * 100:5.1f}%")
    print(f"    2 of 3, third in scope   (0.67 / 0.33)    "
          f"{int((np.abs(top - 2 / 3) < 1e-4).sum()):>7}  "
          f"{(np.abs(top - 2 / 3) < 1e-4).mean() * 100:5.1f}%")
    print(f"    out-of-scope votes dropped, renormalised  "
          f"{sum(n for (i, d), n in vote_shape.items() if d):>7}  "
          f"{sum(n for (i, d), n in vote_shape.items() if d) / len(files) * 100:5.1f}%")

    print(f"\n  softness by class (mean weight on the majority class):")
    for c, name in enumerate(CLASSES):
        m = y == c
        print(f"    {name:<12} {top[m].mean():.4f}   "
              f"unanimous {(top[m] == 1.0).mean() * 100:5.1f}%   n={int(m.sum())}")
    print("  anxious carries the least agreement, which is exactly the spec §1.3 "
          "finding\n  reappearing as a training signal instead of being averaged away.")

    print(f"\n  where the dissenting vote goes, per majority class:")
    print(f"    {'majority':<13}" + "".join(f"{c[:5]:>9}" for c in CLASSES))
    for c, name in enumerate(CLASSES):
        m = (y == c) & (top < 1.0)
        share = soft[m].mean(0)
        print(f"    {name:<13}" + "".join(f"{v:>9.3f}" for v in share))

    out = os.path.join(OUT_DIR, "y_soft.npy")
    np.save(out, soft)
    print(f"\n  wrote {out}  {soft.shape} float32")


if __name__ == "__main__":
    main()
