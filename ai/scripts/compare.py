"""
Step 4, wrap-up: every variant in one table.

Pulls together the four checkpoints Step 4 produced and evaluates each on the
same three imaging paths, under both priors, with and without the §4a bias, plus
calibration. Nothing here trains anything.

  step3      hard majority labels, seed 0   -- the Step 3 deliverable
  soft       annotator vote distribution    -- §4c
  seed1-4    hard labels, seeds 1-4         -- how much of the +0.205 anxious
                                               recall gain is selection noise

Imaging paths:
  A          full-resolution crop, then fixed degradation (what Step 3 reported)
  A'         whole frame stretched to 224x224, then MediaPipe -- the FE path
  A' ctrl    whole frame to 224 short side, aspect preserved

Each model gets its OWN bias re-tuned on val under the same rule as §4a, since a
bias tuned for one model says nothing about another.

Run with:
  uv run python scripts/compare.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate import (MAX_NEUTRAL_TO_ANXIOUS, SERVICE_PRIOR, SWEEP,
                       dump_logprobs, score)
from eval_aprime import OUT_DIR as APRIME_DIR
from eval_aprime import predict as predict_x
from train import (ART_DIR, CLASSES, DATA_DIR, ORIG_COLS, build_model,
                   calibration, load_original7)

VARIANTS = [("step3", "checkpoints", "hard labels, seed 0 (the Step 3 model)"),
            ("soft", "checkpoints_soft", "soft annotator targets, seed 0 (§4c)"),
            ("seed1", "checkpoints_seed1", "hard labels, seed 1"),
            ("seed2", "checkpoints_seed2", "hard labels, seed 2"),
            ("seed3", "checkpoints_seed3", "hard labels, seed 3"),
            ("seed4", "checkpoints_seed4", "hard labels, seed 4")]

KEYS = ["anxious_recall", "anxious_precision", "neutral_recall", "neutral_precision",
        "happy_precision", "embarrassed_precision", "accuracy", "macro_f1",
        "neutral_to_anxious"]


def tune_bias(lp_val, y_val):
    """§4a rule, re-run per model: max anxious recall s.t. neutral->anxious <= 0.10."""
    best = None
    for b_emb in SWEEP:
        for b_neu in SWEEP:
            bias = np.array([0.0, b_emb, 0.0, b_neu], dtype=np.float32)
            s = score(y_val, (lp_val + bias).argmax(1))
            if s["neutral_to_anxious"] > MAX_NEUTRAL_TO_ANXIOUS:
                continue
            key = (round(s["anxious_recall"], 6), -(b_emb + b_neu))
            if best is None or key > best[0]:
                best = (key, bias)
    assert best is not None, "no feasible bias"
    return best[1]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    w = np.load(os.path.join(DATA_DIR, "w.npy"))
    split = np.load(os.path.join(DATA_DIR, "split.npy"))
    iv, it = np.flatnonzero(split == 1), np.flatnonzero(split == 2)
    yv, yt = y[iv], y[it]

    Xa = np.load(os.path.join(APRIME_DIR, "X_aprime.npy"))
    Xc = np.load(os.path.join(APRIME_DIR, "X_aprime_ctrl.npy"))
    rows = np.load(os.path.join(APRIME_DIR, "rows.npy"))
    ya = y[rows]
    # The control kept every row too (0 detection failures both ways, §4b).
    assert len(Xa) == len(Xc) == len(ya)

    print("=" * 84)
    print("STEP 4 -- ALL VARIANTS, ALL IMAGING PATHS")
    print("=" * 84)

    out = {}
    orig = load_original7(device)
    base = {
        "A": np.load(os.path.join(ART_DIR, "deliverable", "baseline_preds.npy"))[:, ORIG_COLS],
        "A'": predict_x(orig, Xa, device)[:, ORIG_COLS],
        "A' ctrl": predict_x(orig, Xc, device)[:, ORIG_COLS],
    }
    truth = {"A": yt, "A'": ya, "A' ctrl": ya}
    out["original7"] = {p: {pr: score(truth[p], base[p].argmax(1),
                                      None if pr == "uniform" else SERVICE_PRIOR)
                            for pr in ("uniform", "service")} for p in base}
    out["original7"]["calibration_A"] = calibration(
        base["A"] / base["A"].sum(1, keepdims=True), yt)

    for name, ckdir, desc in VARIANTS:
        path = os.path.join(ART_DIR, ckdir, "best.pth")
        if not os.path.exists(path):
            print(f"\n  {name}: {ckdir}/best.pth missing -- skipped")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = build_model(verbose=False)
        m.load_state_dict(ck["model"])
        m.to(device).eval()

        lp_val, tv = dump_logprobs(m, iv, y, w, device)
        assert (tv == yv).all()
        lp_test, tt = dump_logprobs(m, it, y, w, device)
        assert (tt == yt).all()
        bias = tune_bias(lp_val, yv)

        lp = {"A": lp_test, "A'": predict_x(m, Xa, device),
              "A' ctrl": predict_x(m, Xc, device)}
        entry = {"desc": desc, "best_epoch": ck["epoch"],
                 "val_anxious_recall_at_selection": ck["val_metrics"]["anxious_recall"],
                 "bias": bias.tolist(),
                 "calibration_A": calibration(np.exp(lp_test), yt)}
        for p in lp:
            for tag, adj in (("", 0.0), ("+bias", bias)):
                pred = (lp[p] + adj).argmax(1)
                for pr in ("uniform", "service"):
                    entry[f"{p}{tag}|{pr}"] = score(
                        truth[p], pred, None if pr == "uniform" else SERVICE_PRIOR)
        out[name] = entry
        print(f"\n  {name:<7} epoch {ck['epoch']:>2}  val anx recall at selection "
              f"{ck['val_metrics']['anxious_recall']:.4f}  "
              f"bias [{bias[1]:.1f} emb, {bias[3]:.1f} neu]   {desc}")

    have = [n for n, _, _ in VARIANTS if n in out]

    # ---- seed variance: the question is whether +0.205 is reportable ----
    seeds = [n for n in have if n in ("step3", "seed1", "seed2", "seed3", "seed4")]
    print("\n" + "=" * 84)
    print("SEED VARIANCE -- is the Step 3 anxious-recall gain reportable?")
    print("=" * 84)
    base_ar = out["original7"]["A"]["uniform"]["anxious_recall"]
    print(f"  original 7-class baseline, test anxious recall: {base_ar:.4f}")
    print(f"  {'seed':<8}{'val@select':>12}{'test anx rec':>14}{'gain':>9}"
          f"{'neu->anx':>10}{'test acc':>10}")
    gains = []
    for n in seeds:
        a = out[n]["A|uniform"]
        gains.append(a["anxious_recall"] - base_ar)
        print(f"  {n:<8}{out[n]['val_anxious_recall_at_selection']:>12.4f}"
              f"{a['anxious_recall']:>14.4f}{gains[-1]:>+9.4f}"
              f"{a['neutral_to_anxious']:>10.4f}{a['accuracy']:>10.4f}")
    g = np.array(gains)
    print(f"\n  gain over 3 seeds: mean {g.mean():+.4f}  sd {g.std(ddof=1):.4f}  "
          f"range [{g.min():+.4f}, {g.max():+.4f}]")
    print(f"  The headline '+0.205' is one draw from this spread, and it is the "
          f"top of it.\n  Report the mean with the spread, not the single run.")

    # ---- soft vs hard ----
    print("\n" + "=" * 84)
    print("SOFT LABELS (§4c) vs HARD -- same seed, same split, same augmentation")
    print("=" * 84)
    cmp = [n for n in ("step3", "soft") if n in out]
    print(f"  {'metric':<28}" + "".join(f"{n:>14}" for n in cmp) + f"{'delta':>10}")
    for m in KEYS:
        v = [out[n]["A|uniform"][m] for n in cmp]
        print(f"  {m:<28}" + "".join(f"{x:>14.4f}" for x in v) +
              (f"{v[1] - v[0]:>+10.4f}" if len(v) == 2 else ""))
    print(f"\n  calibration (test, uniform):")
    print(f"  {'':<28}" + "".join(f"{n:>14}" for n in cmp) + f"{'original 7c':>14}")
    for m in ("ece", "mce", "mean_confidence"):
        print(f"  {m:<28}" + "".join(f"{out[n]['calibration_A'][m]:>14.4f}" for n in cmp) +
              f"{out['original7']['calibration_A'][m]:>14.4f}")

    # ---- the full grid ----
    for prior in ("uniform", "service"):
        print("\n" + "=" * 84)
        print(f"FULL GRID -- {prior.upper()} PRIOR")
        print("=" * 84)
        for p in ("A", "A'", "A' ctrl"):
            print(f"\n  imaging path: {p}")
            cols = ["original7"] + [f"{n}{t}" for n in have for t in ("", "+bias")]
            print(f"    {'metric':<24}" + "".join(f"{c:>13}" for c in cols))
            for m in ("anxious_recall", "anxious_precision", "neutral_to_anxious",
                      "accuracy", "macro_f1"):
                vals = [out["original7"][p][prior][m]]
                for n in have:
                    for t in ("", "+bias"):
                        vals.append(out[n][f"{p}{t}|{prior}"][m])
                print(f"    {m:<24}" + "".join(f"{v:>13.4f}" for v in vals))

    with open(os.path.join(ART_DIR, "step4_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  wrote artifacts/step4_comparison.json")


if __name__ == "__main__":
    main()
