"""
End-to-End Orchestrator for OULAD Data Preparation Pipeline.
Executes the full pipeline deterministically from raw CSV files to processed tables:
1. load: CSV -> interim Parquet with explicit schemas
2. audit: Full statistical audit and report generation
3. population: Filtered cohort (learners.parquet)
4. grid: Dense learner x week grid (grid.parquet)
5. labels: Multi-horizon dropout, inactivity onset, and performance targets (labels.parquet)
6. features: behavior, temporal, assessment, curriculum, static, graph_inputs
7. audit_table: Fairness demographics table (audit_demographics.parquet)
8. splits: Temporal presentation splits and features_weekly.parquet consolidation
"""
import time
from pathlib import Path
from src.data.load import convert_csv_to_parquet
from src.data.audit import run_full_audit
from src.data.population import build_learner_population
from src.data.grid import build_week_grid
from src.data.labels import build_labels
from src.data.features.behavior import build_behavior_features
from src.data.features.temporal import build_temporal_features
from src.data.features.assessment import build_assessment_features
from src.data.features.curriculum import build_curriculum_features
from src.data.features.static import build_static_features
from src.data.features.graph_inputs import build_graph_inputs
from src.data.audit_table import build_audit_demographics
from src.data.splits import build_splits_and_consolidate


def build_all():
    """Execute end-to-end OULAD data preparation pipeline."""
    start_time = time.time()
    print("=" * 60)
    print("STARTING OULAD DATA PREPARATION PIPELINE")
    print("=" * 60)

    print("\n[1/8] Converting raw CSVs to interim Parquet with explicit schemas...")
    convert_csv_to_parquet()

    print("\n[2/8] Running data audit and generating documentation...")
    run_full_audit()

    print("\n[3/8] Building filtered learner population...")
    build_learner_population()

    print("\n[4/8] Building dense learner x week grid...")
    build_week_grid()

    print("\n[5/8] Computing dropout, inactivity, and performance targets...")
    build_labels()

    print("\n[6/8] Computing all feature blocks...")
    print("  -> Behavior features...")
    build_behavior_features()
    print("  -> Temporal features...")
    build_temporal_features()
    print("  -> Assessment features...")
    build_assessment_features()
    print("  -> Curriculum features...")
    build_curriculum_features()
    print("  -> Static features...")
    build_static_features()
    print("  -> Graph input sparse matrices...")
    build_graph_inputs()

    print("\n[7/8] Generating fairness demographics audit table...")
    build_audit_demographics()

    print("\n[8/8] Assigning temporal splits and consolidating weekly feature table...")
    build_splits_and_consolidate()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")
    print("=" * 60)


if __name__ == "__main__":
    build_all()

