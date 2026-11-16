"""
B1.7 Sanity Baseline script.
Trains a plain LogisticRegression model on the last real step of the sequence
plus the static features. Validates that performance matches Vivek's static LR baseline.
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.seq.builder import build_sequences


def run_sanity_baseline():
    print("Building train sequence w8...")
    t0 = time.time()
    seq_tr = build_sequences("train", window=8, target="dropout")
    print(f"Train built in {time.time()-t0:.2f}s: X={seq_tr['X'].shape}")

    print("Building val sequence w8...")
    t1 = time.time()
    seq_va = build_sequences("val", window=8, target="dropout")
    print(f"Val built in {time.time()-t1:.2f}s: X={seq_va['X'].shape}")

    X_tr_last = seq_tr["X"][:, -1, :]
    y_tr = seq_tr["y_dropout"]

    X_va_last = seq_va["X"][:, -1, :]
    y_va = seq_va["y_dropout"]

    print("Fitting LogisticRegression on last real step...")
    t2 = time.time()
    clf = LogisticRegression(C=0.1, max_iter=200, random_state=42)
    clf.fit(X_tr_last, y_tr)
    print(f"Fitted in {time.time()-t2:.2f}s")

    val_probs = clf.predict_proba(X_va_last)[:, 1]
    roc = float(roc_auc_score(y_va, val_probs))
    pr = float(average_precision_score(y_va, val_probs))
    brier = float(brier_score_loss(y_va, val_probs))

    print("=" * 60)
    print("B1.7 SANITY BASELINE RESULTS (LR on last sequence step):")
    print(f"Val ROC-AUC: {roc:.4f} (Vivek 39-feat LR: 0.7370)")
    print(f"Val PR-AUC:  {pr:.4f} (Vivek 39-feat LR: 0.0617)")
    print(f"Val Brier:   {brier:.4f} (Vivek 39-feat LR: 0.0228)")
    print("=" * 60)

    assert abs(roc - 0.7370) < 0.02, f"ROC-AUC {roc} too far from static LR (0.7370)!"
    assert abs(pr - 0.0617) < 0.015, f"PR-AUC {pr} too far from static LR (0.0617)!"
    print("SANITY BASELINE PASSED! Alignment, scaling, and labels confirmed.")
    return roc, pr, brier


if __name__ == "__main__":
    run_sanity_baseline()
