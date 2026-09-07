"""
Step 3: fine-tune EmotionNet from 7 classes to the 4 the service uses.

Reads only the frozen arrays from Step 2 -- the label JSON is not needed here.
The split comes from split.npy and is never recomputed: the fine-tuned model has
to be compared against the original on the *same* test set, and tau is chosen on
val and reported on test. Both break silently if the boundary moves.

What differs from the original AI-Hub training run:
  * final layer 7 -> 4, warm-started from the pretrained rows for
    [기쁨, 당황, 불안, 중립] rather than randomly initialised
  * resolution degradation (spec §3.2) -- the actual point of fine-tuning, since
    the original was trained on this same dataset at full resolution
  * per-sample weights from w.npy (spec §3.4 outdoor 0.5)
  * model selection on ANXIOUS RECALL, not overall accuracy (spec §6)

Outputs land in ai/artifacts/ (gitignored):
  checkpoints/best.pth        best epoch by val anxious recall
  checkpoints/last.pth        every epoch, so a crash does not lose the run
  deliverable/emotionnet_v2.pth   handoff weights, {'model': state_dict}
  deliverable/meta.json       class order, tau, seeds, hyperparameters
  deliverable/history.json    per-epoch losses and metrics
  deliverable/baseline_preds.npy  original 7-class model's test probabilities
  report/*.png                confusion matrices, tau curve, history

Run with:
  uv run python scripts/train.py                  # full run
  uv run python scripts/train.py --epochs 2       # smoke test
  uv run python scripts/train.py --eval-only      # metrics from checkpoints/best.pth

Step 4 added three flags. All of them leave the default run byte-identical:
  --soft            train on the annotator vote distribution (§4c) instead of
                    the hard majority label. Same rows, same split, same seed --
                    only the target changes.
  --seed N          for measuring how much of the Step 3 anxious-recall gain is
                    selection noise (val anxious is 651 frames from 29 people).
  --tag NAME        write to artifacts/<dir>_NAME/ so a comparison run cannot
                    overwrite the Step 3 deliverable.
"""
import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotionnet import EmotionNet

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "processed")
ART_DIR = os.path.join(HERE, "..", "artifacts")
WEIGHTS = os.path.join(HERE, "..", "weights", "model.pth")

# Spec §5 output order.
CLASSES = ["happy", "embarrassed", "anxious", "neutral"]
ANXIOUS = CLASSES.index("anxious")

# The original model's 7 classes, from the AI-Hub source (cropImages.py /
# train.py / emotion.py all agree): ['기쁨','당황','분노','불안','상처','슬픔','중립'].
# These are the columns matching our 4, in our order.
ORIG_7 = ["기쁨", "당황", "분노", "불안", "상처", "슬픔", "중립"]
ORIG_COLS = [0, 1, 3, 6]

SEED = 0
INPUT_SIZE = 48        # spec §2.4
CACHE_SIZE = 160

# §3.2 -- degrade before the 48x48 resize, on 80% of training samples.
DEGRADE_P = 0.8
DEGRADE_PX = (72, 132)
DEGRADE_JPEG_Q = (55, 92)
# Validation and test use ONE fixed degradation; a random one makes scores
# jitter between runs and destroys comparability.
EVAL_PX = 100
EVAL_JPEG_Q = 75

# §3.3
FLIP_P = 0.5
ROT_DEG = 10.0
BRIGHT_CONTRAST = 0.20

# torchvision's Resize((48,48)) in the original train.py is a plain stretch with
# bilinear interpolation -- no aspect-preserving pad. Matched here.
RESIZE_INTERP = cv2.INTER_LINEAR


# ----------------------------------------------------------------------------
# augmentation
# ----------------------------------------------------------------------------

def degrade(face, rng):
    """Spec §3.2. High-resolution crop -> what the webcam path actually delivers.
    Input/output are single-channel uint8; cv2 handles grayscale JPEG fine."""
    t = rng.randint(*DEGRADE_PX)
    small = cv2.resize(face, (t, t), interpolation=cv2.INTER_AREA)

    q = rng.randint(*DEGRADE_JPEG_Q)
    ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        small = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    if rng.random() < 0.5:
        noise = np.random.normal(0, rng.uniform(2, 7), small.shape)
        small = np.clip(small.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.3:
        small = cv2.GaussianBlur(small, (3, 3), rng.uniform(0.3, 1.0))

    return small


def degrade_fixed(face):
    """Deterministic counterpart for val/test."""
    small = cv2.resize(face, (EVAL_PX, EVAL_PX), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, EVAL_JPEG_Q])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE) if ok else small


def augment(face, rng):
    """§3.3 geometric/photometric augmentation, applied on the 160px crop before
    degradation -- lighting and pose happen at capture time, not after transport.
    Vertical flip is forbidden; colour jitter is pointless on grayscale."""
    if rng.random() < FLIP_P:
        face = cv2.flip(face, 1)

    if ROT_DEG > 0:
        angle = rng.uniform(-ROT_DEG, ROT_DEG)
        c = (face.shape[1] / 2, face.shape[0] / 2)
        M = cv2.getRotationMatrix2D(c, angle, 1.0)
        face = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    alpha = 1.0 + rng.uniform(-BRIGHT_CONTRAST, BRIGHT_CONTRAST)   # contrast
    beta = rng.uniform(-BRIGHT_CONTRAST, BRIGHT_CONTRAST) * 255    # brightness
    return cv2.convertScaleAbs(face, alpha=alpha, beta=beta)


class Crops(Dataset):
    """X.npy stays memory-mapped and is opened lazily so DataLoader workers on
    Windows (spawn) do not each pickle a 696 MB array."""

    def __init__(self, indices, y, w, train, seed=SEED, soft=None):
        self.indices = np.asarray(indices)
        self.y = y
        self.w = w
        self.train = train
        self.seed = seed
        # §4c: when present, the per-sample target is the annotator vote
        # distribution. y is still carried so every metric stays comparable.
        self.soft = soft
        self._X = None

    def __len__(self):
        return len(self.indices)

    def _array(self):
        if self._X is None:
            self._X = np.load(os.path.join(DATA_DIR, "X.npy"), mmap_mode="r")
        return self._X

    def __getitem__(self, i):
        idx = int(self.indices[i])
        face = np.asarray(self._array()[idx])

        if self.train:
            rng = random.Random((self.seed, idx, torch.initial_seed()).__hash__())
            face = augment(face, rng)
            if rng.random() < DEGRADE_P:
                face = degrade(face, rng)
        else:
            face = degrade_fixed(face)

        face = cv2.resize(face, (INPUT_SIZE, INPUT_SIZE), interpolation=RESIZE_INTERP)
        return self._pack(face, idx)

    def _pack(self, face, idx):
        x = torch.from_numpy(face.astype(np.float32) / 255.0).unsqueeze(0)
        return x, int(self.y[idx]), float(self.w[idx]), self._target(idx)

    def _target(self, idx):
        if self.soft is not None:
            return torch.from_numpy(self.soft[idx].astype(np.float32))
        return F.one_hot(torch.tensor(int(self.y[idx])), len(CLASSES)).float()


class RawCrops(Crops):
    """No degradation at all -- used to show the size of the domain gap."""

    def __getitem__(self, i):
        idx = int(self.indices[i])
        face = cv2.resize(np.asarray(self._array()[idx]), (INPUT_SIZE, INPUT_SIZE),
                          interpolation=RESIZE_INTERP)
        return self._pack(face, idx)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

def build_model(pretrained=WEIGHTS, verbose=True):
    """EmotionNet with the final layer replaced 7 -> 4 (spec §4).

    The new fc3 is warm-started from the pretrained rows for our four classes
    instead of being randomly initialised -- the original already separates
    them, so there is no reason to throw that away.
    """
    model = EmotionNet(num_classes=len(CLASSES))
    if pretrained and os.path.exists(pretrained):
        sd = torch.load(pretrained, map_location="cpu", weights_only=False)["model"]
        old_w, old_b = sd.pop("fc3.weight"), sd.pop("fc3.bias")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys in checkpoint: {unexpected}"
        assert set(missing) == {"fc3.weight", "fc3.bias"}, f"unexpected missing: {missing}"
        with torch.no_grad():
            model.fc3.weight.copy_(old_w[ORIG_COLS])
            model.fc3.bias.copy_(old_b[ORIG_COLS])
        if verbose:
            print(f"  loaded {pretrained}; fc3 warm-started from original rows "
                  f"{ORIG_COLS} = {[ORIG_7[i] for i in ORIG_COLS]}")
    elif verbose:
        print(f"  WARNING: {pretrained} not found -- training from scratch")
    return model


def load_original7(device):
    """The unmodified 7-class model, for the spec §6 before/after comparison."""
    model = EmotionNet(num_classes=7)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu",
                                     weights_only=False)["model"])
    return model.to(device).eval()


# ----------------------------------------------------------------------------
# metrics (spec §6)
# ----------------------------------------------------------------------------

def confusion(y_true, y_pred, k=len(CLASSES)):
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def metrics(y_true, y_pred):
    cm = confusion(y_true, y_pred)
    out = {"accuracy": float((y_true == y_pred).mean()), "confusion": cm.tolist(),
           "per_class": {}}
    f1s = []
    for i, name in enumerate(CLASSES):
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i].sum() - cm[i, i]
        prec = float(tp / (tp + fp)) if tp + fp else 0.0
        rec = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out["per_class"][name] = {"precision": prec, "recall": rec, "f1": f1,
                                  "support": int(cm[i].sum())}
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    out["anxious_recall"] = out["per_class"]["anxious"]["recall"]
    return out


def calibration(probs, y_true, n_bins=15):
    """§4c: expected calibration error of the top-1 confidence.

    Bins predictions by confidence and compares mean confidence against actual
    accuracy in each bin. This matters here beyond being a nice property: tau is
    a threshold on that confidence, so a miscalibrated model makes tau selection
    mean something different on val than on test.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.25, 1.0, n_bins + 1)
    bins, ece, mce = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > edges[0] else (conf >= lo) & (conf <= hi)
        if not m.any():
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0,
                         "confidence": None, "accuracy": None})
            continue
        c, a = float(conf[m].mean()), float(correct[m].mean())
        gap = abs(c - a)
        ece += m.mean() * gap
        mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "confidence": c, "accuracy": a})
    return {"ece": float(ece), "mce": float(mce),
            "mean_confidence": float(conf.mean()),
            "accuracy": float(correct.mean()), "bins": bins}


def choose_tau(probs, y_true, y_pred):
    """Spec §5/§6: with 4 outputs, D5's 'is top-1 one of the 4 classes' check is
    vacuous, so it becomes 'is top-1 probability >= tau'.

    Picks the tau that best separates correct from incorrect predictions --
    argmax over [P(p>=tau | correct) - P(p>=tau | incorrect)], the Youden J of
    the confidence score. Parameter-free, and the full curve is emitted so a
    human can override with a coverage requirement.
    """
    conf = probs.max(axis=1)
    correct = (y_pred == y_true)
    curve = []
    best = (None, -1.0)
    for tau in np.round(np.arange(0.25, 1.0, 0.01), 3):
        keep = conf >= tau
        tpr = float((keep & correct).sum() / max(correct.sum(), 1))
        fpr = float((keep & ~correct).sum() / max((~correct).sum(), 1))
        acc = float(correct[keep].mean()) if keep.any() else float("nan")
        curve.append({"tau": float(tau), "coverage": float(keep.mean()),
                      "accepted_accuracy": acc, "tpr_correct": tpr,
                      "fpr_incorrect": fpr, "youden": tpr - fpr})
        if tpr - fpr > best[1]:
            best = (float(tau), tpr - fpr)
    return best[0], curve


# ----------------------------------------------------------------------------
# train / eval loops
# ----------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device, n_out=len(CLASSES)):
    model.eval()
    probs, trues = [], []
    for x, y, *_ in loader:
        out = model(x.to(device, non_blocking=True))
        probs.append(out.exp().cpu().numpy())   # model emits log-probabilities
        trues.append(y.numpy())
    return np.concatenate(probs).astype(np.float32), np.concatenate(trues)


def run_epoch(model, loader, device, optimizer, soft=False):
    model.train()
    total, n = 0.0, 0
    for x, y, w, t in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        # The model's last layer is LogSoftmax, so nll_loss is the matching
        # criterion. reduction='none' lets the §3.4 per-sample weights apply.
        # With soft targets it becomes the cross-entropy against the annotator
        # vote distribution, -(t * logprob).sum(1) -- which reduces to exactly
        # nll_loss when t is one-hot, so the two paths agree on unanimous rows.
        if soft:
            per_sample = -(t.to(device, non_blocking=True) * model(x)).sum(1)
        else:
            per_sample = F.nll_loss(model(x), y, reduction="none")
        loss = (per_sample * w).sum() / w.sum()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--soft", action="store_true",
                    help="§4c: train on the annotator vote distribution")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="",
                    help="suffix for the artifact directories, so a comparison "
                         "run does not overwrite the Step 3 deliverable")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    suffix = f"_{args.tag}" if args.tag else ""
    ck_dir = os.path.join(ART_DIR, f"checkpoints{suffix}")
    dv_dir = os.path.join(ART_DIR, f"deliverable{suffix}")
    rp_dir = os.path.join(ART_DIR, f"report{suffix}")
    for d in (ck_dir, dv_dir, rp_dir):
        os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 76)
    print("STEP 3 FINE-TUNING")
    print("=" * 76)
    print(f"device={device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if device.type == 'cuda' else ''}  "
          f"seed={seed}  input={INPUT_SIZE}x{INPUT_SIZE}  classes={CLASSES}")
    print(f"targets={'SOFT (annotator vote distribution, §4c)' if args.soft else 'hard majority label'}"
          f"  artifacts=<...>{suffix or ' (default)'}")

    # ---- data ----
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    w = np.load(os.path.join(DATA_DIR, "w.npy"))
    soft = None
    if args.soft:
        soft_path = os.path.join(DATA_DIR, "y_soft.npy")
        assert os.path.exists(soft_path), "run scripts/soft_labels.py first"
        soft = np.load(soft_path)
        assert (soft.argmax(1) == y).all(), "y_soft.npy is not aligned with y.npy"
    split = np.load(os.path.join(DATA_DIR, "split.npy"))
    groups = np.load(os.path.join(DATA_DIR, "groups.npy"))

    # The frozen split is load-bearing for every comparison in the report.
    # Re-verify it here rather than trusting that nothing touched it.
    seen = {}
    for person, s in zip(groups, split):
        if person in seen:
            assert seen[person] == s, (
                f"person {person} appears in split {seen[person]} and {s}; "
                f"split.npy is corrupt -- do not train on it")
        else:
            seen[person] = s
    print(f"\n  split.npy verified: {len(seen)} people, none crossing splits")

    idx = {name: np.flatnonzero(split == code)
           for name, code in (("train", 0), ("val", 1), ("test", 2))}
    for name in ("train", "val", "test"):
        counts = np.bincount(y[idx[name]], minlength=len(CLASSES))
        print(f"  {name:<6} {len(idx[name]):>6}  " +
              "  ".join(f"{c}={n}" for c, n in zip(CLASSES, counts)))

    def loader(name, train, cls=Crops):
        return DataLoader(cls(idx[name], y, w, train, seed=seed,
                              soft=soft if train else None),
                          batch_size=args.batch_size,
                          shuffle=train, num_workers=args.workers,
                          pin_memory=(device.type == "cuda"), drop_last=False,
                          persistent_workers=bool(args.workers))

    train_loader = loader("train", True)
    val_loader = loader("val", False)
    test_loader = loader("test", False)
    test_raw_loader = loader("test", False, cls=RawCrops)

    # ---- model ----
    print("\n  building model (spec §4: EmotionNet backbone, fc3 7 -> 4)")
    model = build_model().to(device)

    history = []
    best = {"epoch": -1, "anxious_recall": -1.0}

    if not args.eval_only:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        print(f"\n  training: Adam lr={args.lr}, cosine decay, batch={args.batch_size}, "
              f"max {args.epochs} epochs, early stop patience {args.patience}")
        print("  model selection = VAL ANXIOUS RECALL (spec §6), not accuracy\n")
        print(f"  {'ep':>3} {'train_loss':>11} {'val_acc':>8} {'anx_rec':>8} "
              f"{'hap_prec':>9} {'neu_prec':>9} {'emb_prec':>9} {'macroF1':>8} {'s':>6}")

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss = run_epoch(model, train_loader, device, optimizer, soft=args.soft)
            scheduler.step()

            probs, trues = predict(model, val_loader, device)
            m = metrics(trues, probs.argmax(1))
            pc = m["per_class"]
            row = {"epoch": epoch, "train_loss": loss, "lr": scheduler.get_last_lr()[0],
                   "val": m, "seconds": time.time() - t0}
            history.append(row)

            star = ""
            if m["anxious_recall"] > best["anxious_recall"]:
                best = {"epoch": epoch, "anxious_recall": m["anxious_recall"]}
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_metrics": m}, os.path.join(ck_dir, "best.pth"))
                star = "  <- best"
            torch.save({"model": model.state_dict(), "epoch": epoch},
                       os.path.join(ck_dir, "last.pth"))

            print(f"  {epoch:>3} {loss:>11.5f} {m['accuracy']:>8.4f} "
                  f"{m['anxious_recall']:>8.4f} {pc['happy']['precision']:>9.4f} "
                  f"{pc['neutral']['precision']:>9.4f} {pc['embarrassed']['precision']:>9.4f} "
                  f"{m['macro_f1']:>8.4f} {row['seconds']:>6.1f}{star}", flush=True)

            if epoch - best["epoch"] >= args.patience:
                print(f"\n  early stop: no anxious-recall improvement in "
                      f"{args.patience} epochs (best epoch {best['epoch']})")
                break

    # ---- restore best ----
    ck_path = os.path.join(ck_dir, "best.pth")
    assert os.path.exists(ck_path), "no checkpoint to evaluate; run training first"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"\n  restored best checkpoint (epoch {ck['epoch']}, "
          f"val anxious recall {ck['val_metrics']['anxious_recall']:.4f})")

    # ---- tau on VAL, reported on TEST ----
    val_probs, val_true = predict(model, val_loader, device)
    tau, tau_curve = choose_tau(val_probs, val_true, val_probs.argmax(1))
    at_tau = next(c for c in tau_curve if abs(c["tau"] - tau) < 1e-9)
    print(f"\n  tau chosen on VAL = {tau:.2f}  "
          f"(coverage {at_tau['coverage']:.3f}, accepted accuracy "
          f"{at_tau['accepted_accuracy']:.4f})")

    # ---- test: fine-tuned, fine-tuned without degradation, and the original ----
    test_probs, test_true = predict(model, test_loader, device)
    ft = metrics(test_true, test_probs.argmax(1))

    raw_probs, raw_true = predict(model, test_raw_loader, device)
    ft_raw = metrics(raw_true, raw_probs.argmax(1))

    print("\n  evaluating the ORIGINAL 7-class model on the same test set...")
    orig = load_original7(device)
    base_probs7, base_true = predict(orig, test_loader, device, n_out=7)
    # Restrict to our four columns; no renormalisation (spec §5).
    base4 = base_probs7[:, ORIG_COLS]
    base = metrics(base_true, base4.argmax(1))
    other_rate = float((base_probs7.argmax(1)[:, None] != np.array(ORIG_COLS)[None, :])
                       .all(axis=1).mean())
    np.save(os.path.join(dv_dir, "baseline_preds.npy"), base_probs7)

    conf_tau = test_probs.max(1) >= tau
    test_at_tau = {
        "tau": tau,
        "coverage": float(conf_tau.mean()),
        "accepted_accuracy": float((test_probs.argmax(1) == test_true)[conf_tau].mean()),
        "accepted_anxious_recall": float(
            ((test_probs.argmax(1) == ANXIOUS) & conf_tau & (test_true == ANXIOUS)).sum()
            / max((test_true == ANXIOUS).sum(), 1)),
    }

    # ---- report ----
    print("\n" + "=" * 76)
    print("TEST SET (A) -- fixed degradation, person-disjoint from train")
    print("=" * 76)
    print(f"  {'metric':<28}{'original 7c':>14}{'fine-tuned':>14}{'delta':>10}")
    rows = [("accuracy", "accuracy", None), ("macro F1", "macro_f1", None)]
    for c in CLASSES:
        rows.append((f"{c} recall", c, "recall"))
    for c in CLASSES:
        rows.append((f"{c} precision", c, "precision"))
    for label, key, sub in rows:
        b = base[key] if sub is None else base["per_class"][key][sub]
        f = ft[key] if sub is None else ft["per_class"][key][sub]
        flag = "  *" if "anxious recall" in label else ""
        print(f"  {label:<28}{b:>14.4f}{f:>14.4f}{f - b:>+10.4f}{flag}")
    print(f"\n  * spec §6 priority-1 metric")
    print(f"  original model's top-1 falls outside the 4 classes on "
          f"{other_rate * 100:.1f}% of test frames (D5's old 'other' branch)")
    print(f"  fine-tuned, no degradation applied: accuracy {ft_raw['accuracy']:.4f}, "
          f"anxious recall {ft_raw['anxious_recall']:.4f}  (domain-gap reference)")
    print(f"\n  at tau={tau:.2f}: coverage {test_at_tau['coverage']:.3f}, "
          f"accepted accuracy {test_at_tau['accepted_accuracy']:.4f}")

    cal_ft = calibration(test_probs, test_true)
    cal_base = calibration(base4 / base4.sum(1, keepdims=True), base_true)
    print(f"\n  calibration of the top-1 confidence (tau is a threshold on it):")
    print(f"    {'model':<16}{'ECE':>8}{'MCE':>8}{'mean conf':>11}{'accuracy':>10}")
    for name, c in (("original 7c", cal_base), ("fine-tuned", cal_ft)):
        print(f"    {name:<16}{c['ece']:>8.4f}{c['mce']:>8.4f}"
              f"{c['mean_confidence']:>11.4f}{c['accuracy']:>10.4f}")
    print(f"    (original 7c renormalised over the 4 columns so the two are on "
          f"the same scale)")

    print("\n  confusion matrix, fine-tuned (rows = true, cols = predicted):")
    print("      " + "".join(f"{c[:5]:>8}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"  {c[:4]:<5}" + "".join(f"{v:>8}" for v in ft["confusion"][i]))

    # ---- artifacts ----
    torch.save({"model": model.state_dict()},
               os.path.join(dv_dir, "emotionnet_v2.pth"))

    meta = {
        "spec": "ai/docs/emotion-finetune-spec.md",
        "preprocessing": "ai/scripts/preprocess.py; ai/data/processed/_step2_report.json",
        "architecture": {
            "file": "ai/scripts/emotionnet.py (AI-Hub original, verbatim)",
            "class": "EmotionNet(num_classes=4)",
            "change_from_original": "fc3 7 -> 4 outputs only; backbone untouched",
            "fc3_warm_start": {"from_original_columns": ORIG_COLS,
                               "korean": [ORIG_7[i] for i in ORIG_COLS]},
            "output": "LogSoftmax -- same as the original model. Backend applies "
                      "F.softmax() or .exp() (identical: softmax is shift-invariant "
                      "and log_softmax subtracts a per-row constant). See spec §5 note.",
        },
        "input": {"size": [INPUT_SIZE, INPUT_SIZE], "channels": 1,
                  "range": "0-1 (pixel/255)", "crop": "MediaPipe box, 0% margin",
                  "resize": "stretch, bilinear (matches original train.py Resize((48,48)))"},
        "class_order": CLASSES,
        "tau": {"value": tau, "chosen_on": "val", "rule": "argmax Youden J of top-1 "
                "confidence separating correct from incorrect", "test": test_at_tau},
        "training": {"seed": seed, "optimizer": "Adam", "lr": args.lr,
                     "schedule": "CosineAnnealingLR", "batch_size": args.batch_size,
                     "max_epochs": args.epochs, "patience": args.patience,
                     "loss": ("-(soft_target * logprob).sum(1) with §3.4 per-sample "
                              "weights; targets from y_soft.npy (§4c)" if args.soft
                              else "nll_loss with §3.4 per-sample weights"),
                     "soft_labels": bool(args.soft),
                     "selection_metric": "val anxious recall",
                     "best_epoch": ck["epoch"]},
        "calibration": {"test_finetuned": cal_ft, "test_original7": cal_base},
        "augmentation": {"degrade_p": DEGRADE_P, "degrade_px": DEGRADE_PX,
                         "jpeg_q": DEGRADE_JPEG_Q, "hflip_p": FLIP_P,
                         "rotation_deg": ROT_DEG, "brightness_contrast": BRIGHT_CONTRAST,
                         "eval_degradation": {"px": EVAL_PX, "jpeg_q": EVAL_JPEG_Q}},
        "results": {"test_finetuned": ft, "test_original7_restricted": base,
                    "test_finetuned_no_degradation": ft_raw,
                    "original7_top1_outside_4classes_rate": other_rate},
        "set_b_webcam": "NOT MEASURED -- spec §6.1 makes Set B the adoption "
                        "criterion. Do not swap the model in on Set A alone.",
    }
    # Step 4 writes the bias vector and the A' results into this same file, and
    # they are part of the backend contract. Re-running training must not
    # silently delete them -- carry over anything training does not own, and
    # flag it, because a stale bias belongs to the checkpoint it was tuned on.
    meta_path = os.path.join(dv_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            prior_meta = json.load(f)
        carried = [k for k in prior_meta if k not in meta]
        for k in carried:
            meta[k] = prior_meta[k]
        for k, v in prior_meta.get("results", {}).items():
            if k.startswith("step4"):
                meta["results"][k] = v
                carried.append(f"results.{k}")
        if carried:
            print(f"\n  carried over from the previous meta.json: {', '.join(carried)}")
            print("  ** these were tuned on the PREVIOUS checkpoint. If this run "
                  "replaced it,\n  ** re-run scripts/calibrate.py and "
                  "scripts/eval_aprime.py before shipping.")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(dv_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history, "tau_curve": tau_curve},
                  f, ensure_ascii=False, indent=2)

    save_plots(history, tau_curve, tau, ft, base, rp_dir, (cal_base, cal_ft))

    print(f"\n  artifacts -> {os.path.abspath(ART_DIR)}")
    print("    deliverable/emotionnet_v2.pth   {'model': state_dict}, "
          "matching AI-Hub model.pth format")
    print("    deliverable/meta.json, history.json, baseline_preds.npy")
    print("    report/confusion.png, tau_curve.png, history.png")
    print("\n  NOT DONE: evaluation Set B (team webcam). Spec §6.1 makes it the "
          "adoption criterion --\n  Set A alone is not grounds for swapping the model in.")


def save_plots(history, tau_curve, tau, ft, base, out_dir, cals=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if cals:
        fig, ax = plt.subplots(figsize=(5, 4.6))
        ax.plot([0.25, 1], [0.25, 1], color="grey", linewidth=0.8, linestyle=":",
                label="perfect calibration")
        for c, name, style in ((cals[0], "original 7c", "--"), (cals[1], "fine-tuned", "-")):
            pts = [(b["confidence"], b["accuracy"]) for b in c["bins"] if b["n"]]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], style, marker="o",
                    markersize=3, label=f"{name} (ECE {c['ece']:.3f})")
        ax.set_xlabel("top-1 confidence"); ax.set_ylabel("accuracy in bin")
        ax.legend(fontsize=8); ax.set_title("reliability diagram (test)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "reliability.png"), dpi=120)
        plt.close(fig)

    if history:
        ep = [h["epoch"] for h in history]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(ep, [h["train_loss"] for h in history], label="train loss")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("nll loss"); ax[0].legend()
        ax[0].set_title("training loss")
        ax[1].plot(ep, [h["val"]["accuracy"] for h in history], label="val accuracy")
        ax[1].plot(ep, [h["val"]["anxious_recall"] for h in history],
                   label="val anxious recall", linewidth=2)
        ax[1].plot(ep, [h["val"]["macro_f1"] for h in history], label="val macro F1",
                   linestyle="--")
        ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title("validation metrics")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "history.png"), dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    t = [c["tau"] for c in tau_curve]
    ax.plot(t, [c["coverage"] for c in tau_curve], label="coverage")
    ax.plot(t, [c["accepted_accuracy"] for c in tau_curve], label="accuracy of accepted")
    ax.plot(t, [c["youden"] for c in tau_curve], label="Youden J", linestyle="--")
    ax.axvline(tau, color="crimson", linewidth=1, label=f"chosen tau = {tau:.2f}")
    ax.set_xlabel("tau (top-1 probability threshold)"); ax.legend(fontsize=8)
    ax.set_title("tau selection on validation set")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "tau_curve.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, cm, title in ((axes[0], np.array(base["confusion"]), "original 7-class"),
                            (axes[1], np.array(ft["confusion"]), "fine-tuned 4-class")):
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        axis.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        axis.set_xticks(range(len(CLASSES))); axis.set_yticks(range(len(CLASSES)))
        axis.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
        axis.set_yticklabels(CLASSES, fontsize=8)
        axis.set_xlabel("predicted"); axis.set_ylabel("true"); axis.set_title(title)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                axis.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center",
                          va="center", fontsize=7,
                          color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "confusion.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
