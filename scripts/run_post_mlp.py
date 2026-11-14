"""
Post-MLP pipeline: runs permutation importance for all zoo models,
prints the full Checkpoint B table, and appends the Checkpoint B section
to reports/phase3/phase3_summary.md.

Usage:
    python -m scripts.run_post_mlp
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def step(msg: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}")


def run(cmd: list[str]) -> int:
    """Run a command and return its exit code, streaming output."""
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main() -> int:
    # ── 1. Verify MLP output exists ───────────────────────────────────────────
    step("1/4  Verifying MLP outputs")
    mlp_parquet = Path("predictions/dropout/k3/mlp_val.parquet")
    mlp_params = Path("models/zoo/mlp_best_params.json")
    mlp_model = Path("models/zoo/mlp_seed42.pt")
    zoo_results = Path("reports/zoo_checkpoint_results.json")

    missing = [
        p for p in [mlp_parquet, mlp_params, mlp_model, zoo_results]
        if not p.exists()
    ]
    if missing:
        print(f"ERROR: missing files: {missing}")
        return 1

    with open(zoo_results) as f:
        results = json.load(f)
    if "mlp" not in results:
        print("ERROR: 'mlp' key missing from zoo_checkpoint_results.json")
        return 1

    mlp = results["mlp"]
    print(f"MLP seed_mean_pr = {mlp['seed_mean_pr']:.4f} ± {mlp['seed_std_pr']:.4f}")
    print(f"MLP seed_mean_roc = {mlp['seed_mean_roc']:.4f}")
    print(f"MLP inner_cv_pr_auc = {mlp.get('inner_cv_pr_auc', 'N/A')}")
    print(f"MLP chosen_epochs = {mlp.get('chosen_epochs', 'N/A')}")
    print(f"MLP device = {mlp.get('best_params', {}).get('device', 'N/A')}")
    print(f"MLP train_time_sec = {mlp.get('train_time_sec', 'N/A'):.1f}s")
    print(f"MLP model_size_mb = {mlp.get('model_size_mb', 'N/A'):.2f} MB")

    # ── 2. Permutation importance (all 5 models) ──────────────────────────────
    step("2/4  Computing permutation importance (all 5 models, fixed 5k subsample)")
    perm_out = Path("reports/phase3/permutation_importance.json")
    rc = run([sys.executable, "-m", "scripts.compute_perm_importance"])
    if rc != 0:
        print("ERROR: permutation importance script failed")
        return 1
    if not perm_out.exists():
        print("ERROR: permutation_importance.json not created")
        return 1
    print(f"Saved: {perm_out}")

    # ── 3. Print Checkpoint B table ───────────────────────────────────────────
    step("3/4  Printing Checkpoint B comparison table")
    rc = run([sys.executable, "-m", "scripts.print_checkpoint", "B"])
    if rc != 0:
        print("WARNING: print_checkpoint returned non-zero exit code")

    # ── 4. Append Checkpoint B section to phase3_summary.md ──────────────────
    step("4/4  Appending Checkpoint B section to phase3_summary.md")
    rc = run([sys.executable, "-m", "scripts.write_checkpoint_b_summary"])
    if rc != 0:
        print("WARNING: write_checkpoint_b_summary returned non-zero exit code")

    # ── 5. Run full pytest suite ──────────────────────────────────────────────
    step("5/5  Full pytest suite")
    rc = run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])
    if rc != 0:
        print("ERROR: pytest failed")
        return 1

    print("\nCheckpoint B pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
