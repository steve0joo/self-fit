"""
Final report figures: candidate comparison, seed variance, calibration,
before/after confusion matrices.

Reads only artifacts already produced by train.py / calibrate.py / compare.py
(step4_comparison.json, deliverable/meta.json) -- trains nothing, re-evaluates
nothing. Written for ai/docs/05-model-comparison.md.

Run with:
  uv run python scripts/report_comparison.py
"""
import json
import os

import numpy as np

ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
OUT_DIR = os.path.join(ART_DIR, "report_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATE_COLOR = "#c0392b"   # seed1+bias -- the current deliverable
ORIG_COLOR = "#7f8c8d"        # original 7-class baseline
OTHER_COLOR = "#5b8fa8"       # other candidates considered


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(ART_DIR, "step4_comparison.json"), encoding="utf-8") as f:
        cmp = json.load(f)
    with open(os.path.join(ART_DIR, "deliverable", "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    # ---- Figure 1: candidate comparison, service prior, Set A + bias ----
    order = ["original7", "step3", "soft", "seed1", "seed2", "seed3", "seed4"]
    labels = ["original\n7-class", "step3\n(hard, seed0)", "soft\nlabels",
              "seed1\n(adopted)", "seed2", "seed3", "seed4"]
    metrics = [("accuracy", "accuracy"), ("macro_f1", "macro F1"),
               ("anxious_recall", "anxious recall"), ("anxious_precision", "anxious precision")]

    def get(name, metric):
        if name == "original7":
            return cmp["original7"]["A"]["service"][metric]
        key = "A+bias|service"
        return cmp[name][key][metric]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (mkey, mlabel) in zip(axes, metrics):
        vals = [get(n, mkey) for n in order]
        colors = [ORIG_COLOR if n == "original7" else
                  (CANDIDATE_COLOR if n == "seed1" else OTHER_COLOR) for n in order]
        bars = ax.bar(range(len(order)), vals, color=colors)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(mlabel, fontsize=10)
        ax.set_ylim(0, max(vals) * 1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.3f}",
                     ha="center", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Candidate comparison -- Set A, service prior, bias re-tuned per model "
                  "(neutral→anxious ≤ 0.10 rule)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT_DIR, "candidate_comparison.png"), dpi=130)
    plt.close(fig)

    # ---- Figure 2: seed variance in the anxious-recall gain ----
    seeds = ["step3", "seed1", "seed2", "seed3", "seed4"]
    seed_labels = ["seed0\n(step3)", "seed1\n(adopted)", "seed2", "seed3", "seed4"]
    base_ar = cmp["original7"]["A"]["uniform"]["anxious_recall"]
    gains = np.array([cmp[s]["A|uniform"]["anxious_recall"] - base_ar for s in seeds])
    mean, sd = gains.mean(), gains.std(ddof=1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhspan(mean - sd, mean + sd, color="#5b8fa8", alpha=0.15,
               label=f"mean ± 1 sd  ({mean:+.3f} ± {sd:.3f})")
    ax.axhline(mean, color="#5b8fa8", linewidth=1, linestyle="--")
    colors = [CANDIDATE_COLOR if s == "seed1" else OTHER_COLOR for s in seeds]
    ax.scatter(range(len(seeds)), gains, color=colors, s=90, zorder=3)
    for i, g in enumerate(gains):
        ax.text(i, g + (0.012 if g >= 0 else -0.02), f"{g:+.3f}", ha="center", fontsize=8.5)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels(seed_labels, fontsize=9)
    ax.set_ylabel("anxious recall gain over original 7-class\n(test, uniform prior, no bias)")
    ax.set_title("The headline \"+0.205\" (seed0) is one draw from a wide spread,\n"
                  "not a stable property of fine-tuning", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "seed_variance.png"), dpi=130)
    plt.close(fig)

    # ---- Figure 3: calibration -- ECE, hard vs soft vs original ----
    ece_orig = cmp["original7"]["calibration_A"]["ece"]
    ece_hard = cmp["step3"]["calibration_A"]["ece"]
    ece_soft = cmp["soft"]["calibration_A"]["ece"]
    names = ["original\n7-class", "fine-tuned\n(hard labels)", "fine-tuned\n(soft labels)"]
    vals = [ece_orig, ece_hard, ece_soft]
    colors = [ORIG_COLOR, OTHER_COLOR, "#2e8b57"]

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_ylabel("Expected Calibration Error (lower is better)")
    ax.set_title("Soft annotator labels cut calibration error 2.9×\nvs. the hard-label model "
                  "(test, uniform prior)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "calibration_ece.png"), dpi=130)
    plt.close(fig)

    # ---- Figure 4: confusion matrices, original vs adopted candidate ----
    classes = ["happy", "embar.", "anxious", "neutral"]
    cm_orig = np.array(meta["results"]["test_original7_restricted"]["confusion"], dtype=float)
    # NOTE: results.test_finetuned is the RAW seed1 model (no bias) -- the deployed
    # contract in meta["bias"] is applied downstream of that. Use the bias-adjusted
    # confusion (results.step4a_bias.uniform.ft_bias) so the figure matches what
    # actually ships, not the intermediate unbiased checkpoint.
    cm_new = np.array(meta["results"]["step4a_bias"]["uniform"]["ft_bias"]["confusion"], dtype=float)
    cm_orig_n = cm_orig / cm_orig.sum(1, keepdims=True)
    cm_new_n = cm_new / cm_new.sum(1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, cm, cmn, title in ((axes[0], cm_orig, cm_orig_n, "original 7-class\n(restricted to 4 cols)"),
                                (axes[1], cm_new, cm_new_n, "adopted candidate\nseed1 + bias (deliverable/)")):
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(classes, fontsize=8)
        ax.set_yticks(range(4)); ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
        ax.set_title(title, fontsize=10)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{int(cm[i, j])}\n{cmn[i, j]:.0%}", ha="center", va="center",
                         fontsize=7.5, color="white" if cmn[i, j] > 0.5 else "black")
    fig.suptitle("anxious recall recovers (row 3, 52%→62%) by reclaiming frames that used to\n"
                 "leak into neutral (24%→14%); neutral recall is preserved (row 4, 88%→89%)",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, "confusion_before_after.png"), dpi=130)
    plt.close(fig)

    print("wrote:")
    for fn in ("candidate_comparison.png", "seed_variance.png", "calibration_ece.png",
               "confusion_before_after.png"):
        print(" ", os.path.join(OUT_DIR, fn))
    print(f"\nseed anxious-recall gain: mean {mean:+.4f} sd {sd:.4f} "
          f"range [{gains.min():+.4f}, {gains.max():+.4f}]")


if __name__ == "__main__":
    main()
