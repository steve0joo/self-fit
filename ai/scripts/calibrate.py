"""
Step 4a: per-class bias on the log-probabilities. No retraining.

The Step 3 model was trained on a class-balanced set (6,800 per class, spec
§1.3), so it learned a uniform prior. An interview stream is neutral-dominant.
The decision boundary is therefore calibrated for the wrong prior, and the
balanced test set hides the cost: 28% of neutral frames come back anxious.

A bias added to the log-probabilities before the argmax moves that boundary:

    adjusted = logprob + bias        bias = [b_hap, b_emb, 0.0, b_neu]

bias[anxious] is pinned to 0 -- adding a constant to every class changes
nothing after softmax, so one degree of freedom is redundant. b_hap is also
pinned to 0: the sweep is over the two classes that actually leak into anxious.

Tuned on VAL only, reported on TEST, exactly like tau.

Two priors are reported side by side:
  uniform  -- the test set as built (~25% each). What Step 3 reported.
  service  -- neutral .60 / anxious .15 / embarrassed .15 / happy .10, applied
              by reweighting each test sample by prior[c]/empirical[c]. Recall
              is a within-class rate and does not move; precision does, and
              precision is what a false tension alert costs.

Outputs (ai/artifacts/):
  deliverable/ft_logprobs_val.npy   N_val x 4 float32, so this is re-tunable
  deliverable/ft_logprobs_test.npy  without re-running the model
  deliverable/bias_sweep.json       the full trade-off surface
  deliverable/meta.json             updated in place with the bias contract
  report/bias_tradeoff.png

Run with:
  uv run python scripts/calibrate.py
  uv run python scripts/calibrate.py --reuse     # skip the forward passes
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (ANXIOUS, ART_DIR, CLASSES, Crops, DATA_DIR, ORIG_COLS,
                   build_model, choose_tau)

NEUTRAL = CLASSES.index("neutral")
EMBARRASSED = CLASSES.index("embarrassed")
HAPPY = CLASSES.index("happy")

# The deployment mix an interview stream actually produces. Not measured on our
# own footage -- it is a stated assumption, and every "service prior" number
# below inherits it. Set B is what would replace it with a measurement.
SERVICE_PRIOR = {"happy": 0.10, "embarrassed": 0.15, "anxious": 0.15, "neutral": 0.60}

SWEEP = np.round(np.arange(0.0, 3.01, 0.1), 2)
MAX_NEUTRAL_TO_ANXIOUS = 0.08      # the constraint the operating point is chosen under


# ----------------------------------------------------------------------------
# prior-aware metrics
# ----------------------------------------------------------------------------

def weighted_confusion(y_true, y_pred, sw):
    cm = np.zeros((len(CLASSES), len(CLASSES)))
    np.add.at(cm, (y_true, y_pred), sw)
    return cm


def prior_weights(y_true, prior):
    """Per-sample weight that turns the empirical class mix into `prior`.

    The test set is ~25% per class by construction; multiplying each sample of
    class c by prior[c]/empirical[c] gives the confusion matrix the same model
    would produce on a stream with that mix, without needing such a stream.
    """
    counts = np.bincount(y_true, minlength=len(CLASSES)).astype(float)
    per_class = np.array([prior[c] / max(n, 1) for c, n in zip(CLASSES, counts)])
    return per_class[y_true]


def score(y_true, y_pred, prior=None):
    sw = np.ones(len(y_true)) if prior is None else prior_weights(y_true, prior)
    cm = weighted_confusion(y_true, y_pred, sw)
    out = {}
    for i, name in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i].sum() - tp
        out[f"{name}_recall"] = float(tp / (tp + fn)) if tp + fn else 0.0
        out[f"{name}_precision"] = float(tp / (tp + fp)) if tp + fp else 0.0
    out["accuracy"] = float(np.trace(cm) / cm.sum())
    out["macro_f1"] = float(np.mean([
        2 * out[f"{c}_precision"] * out[f"{c}_recall"]
        / max(out[f"{c}_precision"] + out[f"{c}_recall"], 1e-12) for c in CLASSES]))
    # Share of all frames the model calls anxious, and how much of that call is
    # actually neutral. These are the two numbers a false tension alert is made of.
    out["anxious_call_rate"] = float(cm[:, ANXIOUS].sum() / cm.sum())
    out["anxious_calls_from_neutral"] = float(
        cm[NEUTRAL, ANXIOUS] / max(cm[:, ANXIOUS].sum(), 1e-12))
    out["neutral_to_anxious"] = float(cm[NEUTRAL, ANXIOUS] / max(cm[NEUTRAL].sum(), 1e-12))
    out["confusion"] = cm.tolist()
    return out


# ----------------------------------------------------------------------------
# log-probability dumps
# ----------------------------------------------------------------------------

@torch.no_grad()
def dump_logprobs(model, indices, y, w, device, batch=256):
    """Same fixed val/test degradation as Step 3 (Crops(train=False))."""
    loader = DataLoader(Crops(indices, y, w, train=False), batch_size=batch,
                        shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    lp, trues = [], []
    model.eval()
    for x, yy, *_ in loader:
        lp.append(model(x.to(device)).cpu().numpy())
        trues.append(yy.numpy())
    return np.concatenate(lp).astype(np.float32), np.concatenate(trues)


def kofn_table(p_fp, p_tp, n=15, ks=range(6, 15)):
    """4d: 'anxious >= 0.5 sustained 5s' reframed as K of the last N frames.

    Binomial, i.e. independent frame errors -- which they are not. A person
    holding one expression the model misreads produces correlated errors, so
    these false-alert numbers are an optimistic bound. Reported anyway because
    the shape of the trade-off is right even when the level is not.
    """
    from math import comb
    tail = lambda p, k: sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
    return [{"k": k, "n": n, "false_alert_from_neutral": tail(p_fp, k),
             "true_anxious_detected": tail(p_tp, k)} for k in ks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="use existing ft_logprobs_*.npy instead of re-running the model")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dv_dir = os.path.join(ART_DIR, "deliverable")
    rp_dir = os.path.join(ART_DIR, "report")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    w = np.load(os.path.join(DATA_DIR, "w.npy"))
    split = np.load(os.path.join(DATA_DIR, "split.npy"))
    idx = {"val": np.flatnonzero(split == 1), "test": np.flatnonzero(split == 2)}

    print("=" * 78)
    print("STEP 4a -- PER-CLASS BIAS ON THE LOG-PROBABILITIES (no retraining)")
    print("=" * 78)

    paths = {s: os.path.join(dv_dir, f"ft_logprobs_{s}.npy") for s in idx}
    if args.reuse and all(os.path.exists(p) for p in paths.values()):
        lp = {s: np.load(p) for s, p in paths.items()}
        print("  reused existing log-probability dumps")
    else:
        ck = torch.load(os.path.join(ART_DIR, "checkpoints", "best.pth"),
                        map_location="cpu", weights_only=False)
        model = build_model(verbose=False)
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        print(f"  loaded checkpoints/best.pth (epoch {ck['epoch']}, "
              f"val anxious recall {ck['val_metrics']['anxious_recall']:.4f})")
        lp = {}
        for s in idx:
            lp[s], trues = dump_logprobs(model, idx[s], y, w, device)
            assert (trues == y[idx[s]]).all(), f"{s} ordering drifted from split.npy"
            np.save(paths[s], lp[s])
            print(f"  dumped {s:<5} {lp[s].shape} -> deliverable/ft_logprobs_{s}.npy")

    yt = {s: y[idx[s]] for s in idx}

    # ---- baselines ----
    base7 = np.load(os.path.join(dv_dir, "baseline_preds.npy"))       # test, 7-class probs
    assert len(base7) == len(yt["test"]), "baseline_preds.npy does not match the test split"
    pred = {
        "orig": base7[:, ORIG_COLS].argmax(1),                        # no renormalisation, §5
        "ft": lp["test"].argmax(1),
    }

    # ---- sweep on VAL ----
    print(f"\n  sweeping b_emb x b_neu over [0, 3] step 0.1 on VAL "
          f"({len(yt['val'])} frames); b_hap and b_anx pinned to 0")
    sweep = []
    for b_emb in SWEEP:
        for b_neu in SWEEP:
            bias = np.array([0.0, b_emb, 0.0, b_neu], dtype=np.float32)
            p = (lp["val"] + bias).argmax(1)
            u = score(yt["val"], p)
            sv = score(yt["val"], p, SERVICE_PRIOR)
            sweep.append({
                "b_emb": float(b_emb), "b_neu": float(b_neu),
                "anxious_recall": u["anxious_recall"],
                "neutral_to_anxious": u["neutral_to_anxious"],
                "anxious_precision_uniform": u["anxious_precision"],
                "anxious_precision_service": sv["anxious_precision"],
                "accuracy_uniform": u["accuracy"],
                "macro_f1_uniform": u["macro_f1"],
                "anxious_call_rate_service": sv["anxious_call_rate"],
            })

    feasible = [s for s in sweep if s["neutral_to_anxious"] <= MAX_NEUTRAL_TO_ANXIOUS]
    assert feasible, "no point in the sweep meets the neutral->anxious constraint"
    # Maximise anxious recall under the constraint; ties break toward the
    # smallest bias, i.e. the least distortion of the trained model.
    chosen = max(feasible, key=lambda s: (round(s["anxious_recall"], 6),
                                          -(s["b_emb"] + s["b_neu"])))
    BIAS = np.array([0.0, chosen["b_emb"], 0.0, chosen["b_neu"]], dtype=np.float32)
    pred["ft_bias"] = (lp["test"] + BIAS).argmax(1)

    # ---- the curve, not just the point ----
    print("\n  TRADE-OFF CURVE on VAL -- b_emb held at the chosen value, b_neu swept")
    print(f"  {'b_neu':>6} {'anx recall':>11} {'neu->anx':>9} "
          f"{'anx prec (unif)':>16} {'anx prec (svc)':>15} {'acc':>7}")
    for s in sweep:
        if abs(s["b_emb"] - chosen["b_emb"]) < 1e-9 and abs(s["b_neu"] * 10 % 2) < 1e-6:
            mark = "  <- chosen" if abs(s["b_neu"] - chosen["b_neu"]) < 1e-9 else ""
            print(f"  {s['b_neu']:>6.1f} {s['anxious_recall']:>11.4f} "
                  f"{s['neutral_to_anxious']:>9.4f} {s['anxious_precision_uniform']:>16.4f} "
                  f"{s['anxious_precision_service']:>15.4f} {s['accuracy_uniform']:>7.4f}{mark}")

    print("\n  the frontier -- best anxious recall at each neutral->anxious ceiling (VAL)")
    print(f"  {'ceiling':>8} {'b_emb':>6} {'b_neu':>6} {'anx recall':>11} "
          f"{'neu->anx':>9} {'anx prec (svc)':>15}")
    frontier = []
    for ceiling in (0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.28):
        cand = [s for s in sweep if s["neutral_to_anxious"] <= ceiling]
        if not cand:
            continue
        b = max(cand, key=lambda s: (round(s["anxious_recall"], 6),
                                     -(s["b_emb"] + s["b_neu"])))
        frontier.append({"ceiling": ceiling, **b})
        mark = "  <- chosen" if abs(ceiling - MAX_NEUTRAL_TO_ANXIOUS) < 1e-9 else ""
        print(f"  {ceiling:>8.2f} {b['b_emb']:>6.1f} {b['b_neu']:>6.1f} "
              f"{b['anxious_recall']:>11.4f} {b['neutral_to_anxious']:>9.4f} "
              f"{b['anxious_precision_service']:>15.4f}{mark}")

    print(f"\n  CHOSEN on VAL: bias = [{BIAS[0]:.1f}, {BIAS[1]:.1f}, {BIAS[2]:.1f}, "
          f"{BIAS[3]:.1f}]  (happy, embarrassed, anxious, neutral)")
    print(f"    rule: maximise anxious recall subject to neutral->anxious <= "
          f"{MAX_NEUTRAL_TO_ANXIOUS:.2f}")

    # For reference only: the analytic prior correction, log(service/train prior).
    # Not adopted -- it optimises likelihood under the assumed prior, not the
    # anxious-recall-under-a-false-positive-ceiling objective §6 actually cares
    # about. Printed so the chosen vector can be read against a principled one.
    analytic = np.log(np.array([SERVICE_PRIOR[c] for c in CLASSES]) / 0.25)
    analytic = analytic - analytic[ANXIOUS]
    a_pred = (lp["val"] + analytic.astype(np.float32)).argmax(1)
    a_u = score(yt["val"], a_pred)
    print(f"    reference, analytic prior correction log(prior/0.25) = "
          f"[{', '.join(f'{v:+.2f}' for v in analytic)}]: "
          f"anx recall {a_u['anxious_recall']:.4f}, "
          f"neu->anx {a_u['neutral_to_anxious']:.4f}  (not adopted)")

    # ---- the table: three models x two priors, on TEST ----
    cols = [("original 7c", "orig"), ("fine-tuned", "ft"), ("ft + bias", "ft_bias")]
    results = {}
    for prior_name, prior in (("uniform", None), ("service", SERVICE_PRIOR)):
        results[prior_name] = {k: score(yt["test"], pred[k], prior) for _, k in cols}

    for prior_name, note in (("uniform", "the balanced test set as built"),
                             ("service", "reweighted to neutral .60 / anxious .15 "
                                         "/ embarrassed .15 / happy .10")):
        print("\n" + "=" * 78)
        print(f"TEST (Set A), {prior_name.upper()} PRIOR -- {note}")
        print("=" * 78)
        print(f"  {'metric':<30}" + "".join(f"{n:>16}" for n, _ in cols))
        keys = ["anxious_recall", "anxious_precision", "neutral_recall",
                "neutral_precision", "happy_precision", "embarrassed_precision",
                "accuracy", "macro_f1", "neutral_to_anxious",
                "anxious_call_rate", "anxious_calls_from_neutral"]
        for m in keys:
            print(f"  {m:<30}" + "".join(
                f"{results[prior_name][k][m]:>16.4f}" for _, k in cols))

    print("\n  confusion, ft + bias (rows = true, cols = predicted, uniform prior):")
    cmb = np.array(results["uniform"]["ft_bias"]["confusion"]).round().astype(int)
    print("       " + "".join(f"{c[:5]:>8}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c[:5]:<5}" + "".join(f"{v:>8}" for v in cmb[i]))

    # ---- tau must be re-picked: the bias moves both argmax and confidence ----
    adj_val = lp["val"] + BIAS
    adj_val = adj_val - adj_val.max(1, keepdims=True)
    pv = np.exp(adj_val) / np.exp(adj_val).sum(1, keepdims=True)
    tau_b, tau_curve_b = choose_tau(pv, yt["val"], pv.argmax(1))
    at = next(c for c in tau_curve_b if abs(c["tau"] - tau_b) < 1e-9)
    adj_test = lp["test"] + BIAS
    adj_test = adj_test - adj_test.max(1, keepdims=True)
    pt = np.exp(adj_test) / np.exp(adj_test).sum(1, keepdims=True)
    keep = pt.max(1) >= tau_b
    print(f"\n  tau re-selected on the BIASED probabilities (the old tau=0.94 was "
          f"picked on\n  unbiased ones and no longer means the same thing): tau = {tau_b:.2f}, "
          f"val coverage {at['coverage']:.3f}, test coverage {keep.mean():.3f}, "
          f"test accepted accuracy {(pt.argmax(1) == yt['test'])[keep].mean():.4f}")

    # ---- 4d: K-of-N ----
    p_fp = results["uniform"]["ft"]["neutral_to_anxious"]
    p_tp = results["uniform"]["ft"]["anxious_recall"]
    p_fp_b = results["uniform"]["ft_bias"]["neutral_to_anxious"]
    p_tp_b = results["uniform"]["ft_bias"]["anxious_recall"]
    kofn = {"unbiased": kofn_table(p_fp, p_tp), "biased": kofn_table(p_fp_b, p_tp_b)}
    print("\n" + "=" * 78)
    print("4d -- 'anxious for 5s' as K of the last N=15 frames (3 fps)")
    print("=" * 78)
    print("  binomial, so frame errors are assumed independent. They are not: a held")
    print("  expression the model misreads produces correlated errors. The false-alert")
    print("  column is an optimistic bound.")
    print(f"\n  {'K':>3}" + f"{'false alert (ft)':>19}{'detect (ft)':>14}"
          f"{'false alert (+bias)':>21}{'detect (+bias)':>16}")
    for a, b in zip(kofn["unbiased"], kofn["biased"]):
        print(f"  {a['k']:>3}{a['false_alert_from_neutral']:>19.4f}"
              f"{a['true_anxious_detected']:>14.4f}"
              f"{b['false_alert_from_neutral']:>21.4f}"
              f"{b['true_anxious_detected']:>16.4f}")

    # ---- persist ----
    with open(os.path.join(dv_dir, "bias_sweep.json"), "w", encoding="utf-8") as f:
        json.dump({"service_prior": SERVICE_PRIOR, "constraint": MAX_NEUTRAL_TO_ANXIOUS,
                   "chosen": chosen, "bias": BIAS.tolist(), "frontier": frontier,
                   "sweep": sweep, "tau_curve_biased": tau_curve_b,
                   "kofn": kofn, "test": results}, f, ensure_ascii=False, indent=2)

    meta_path = os.path.join(dv_dir, "meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["bias"] = {
        # Rounded: these came off a 0.1 grid, and float32 noise in a value the
        # backend types into its own config is only a source of confusion.
        "value": [round(float(v), 1) for v in BIAS],
        "class_order": CLASSES,
        "applies_to": "log-probabilities, i.e. the raw model output",
        "serving_contract":
            "adjusted = model(x) + bias;  prob = softmax(adjusted);  "
            "pred = argmax(prob);  accept only if prob.max() >= tau. "
            "The bias MUST be added before both the argmax and the tau "
            "comparison. Adding it after softmax is wrong.",
        "why": "training was class-balanced at 6,800/class (spec §1.3) so the "
               "model carries a uniform prior; the interview stream is "
               "neutral-dominant. Without this the model calls 28% of neutral "
               "frames anxious.",
        "chosen_on": "val",
        "rule": f"max anxious recall s.t. neutral->anxious <= {MAX_NEUTRAL_TO_ANXIOUS}",
        "assumed_service_prior": SERVICE_PRIOR,
        "assumed_service_prior_note":
            "an assumption, not a measurement. Set B would replace it.",
    }
    meta["tau_biased"] = {"value": tau_b, "chosen_on": "val",
                          "note": "use with meta['bias']; the unbiased tau=0.94 "
                                  "was selected on a different probability scale",
                          "val_coverage": at["coverage"],
                          "test_coverage": float(keep.mean())}
    meta["results"]["step4a_bias"] = results
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_plot(sweep, chosen, os.path.join(rp_dir, "bias_tradeoff.png"))
    print(f"\n  wrote deliverable/bias_sweep.json, updated deliverable/meta.json,")
    print(f"        report/bias_tradeoff.png")


def save_plot(sweep, chosen, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    row = [s for s in sweep if abs(s["b_emb"] - chosen["b_emb"]) < 1e-9]
    bn = [s["b_neu"] for s in row]
    ax[0].plot(bn, [s["anxious_recall"] for s in row], label="anxious recall")
    ax[0].plot(bn, [s["neutral_to_anxious"] for s in row], label="neutral -> anxious")
    ax[0].plot(bn, [s["anxious_precision_service"] for s in row],
               label="anxious precision (service prior)", linestyle="--")
    ax[0].axhline(MAX_NEUTRAL_TO_ANXIOUS, color="grey", linewidth=0.8)
    ax[0].axvline(chosen["b_neu"], color="crimson", linewidth=1,
                  label=f"chosen b_neu = {chosen['b_neu']:.1f}")
    ax[0].set_xlabel(f"b_neu   (b_emb = {chosen['b_emb']:.1f})")
    ax[0].legend(fontsize=7); ax[0].set_title("bias sweep on validation")

    ax[1].scatter([s["neutral_to_anxious"] for s in sweep],
                  [s["anxious_recall"] for s in sweep], s=4, alpha=0.35,
                  label="all (b_emb, b_neu)")
    ax[1].scatter([chosen["neutral_to_anxious"]], [chosen["anxious_recall"]],
                  color="crimson", s=40, zorder=3, label="chosen")
    ax[1].axvline(MAX_NEUTRAL_TO_ANXIOUS, color="grey", linewidth=0.8,
                  label=f"constraint {MAX_NEUTRAL_TO_ANXIOUS}")
    ax[1].set_xlabel("neutral -> anxious rate"); ax[1].set_ylabel("anxious recall")
    ax[1].legend(fontsize=7); ax[1].set_title("the trade-off the choice sits on")
    fig.tight_layout(); fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
