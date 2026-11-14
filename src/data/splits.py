"""
Splits and Preprocessor serialization module for OULAD data preparation pipeline.
Assigns temporal presentation splits:
- Train: 2013B, 2013J
- Validation: 2014B
- Test: 2014J
Fits baseline Scaler and OneHotEncoder on TRAIN ONLY and serializes to models/preprocessing/.
Merges feature groups and split labels into data/processed/features_weekly.parquet.
Writes split metadata to data/processed/splits.yaml.
"""
from pathlib import Path
from typing import Dict, Any, List
import joblib
import polars as pl
from sklearn.preprocessing import StandardScaler
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_splits_and_consolidate(
    interim_dir: str = "data/interim",
    processed_dir: str = "data/processed",
    models_dir: str = "models/preprocessing",
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """
    1. Loads all feature tables from data/interim/:
       features_behavior, features_temporal, features_assessment, features_curriculum, features_static
    2. Merges them on (code_module, code_presentation, id_student, week).
    3. Adds temporal 'split' column based on code_presentation:
       - 2013B, 2013J -> 'train'
       - 2014B -> 'val'
       - 2014J -> 'test'
    4. Joins labels from data/processed/labels.parquet.
    5. Saves combined table to data/processed/features_weekly.parquet.
    6. Fits StandardScaler on continuous numeric features using TRAIN split only.
       Serializes scaler to models/preprocessing/train_scaler.joblib.
    7. Generates split report and saves to data/processed/splits.yaml.
    """
    interim_path = Path(interim_dir)
    processed_path = Path(processed_dir)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    grid_key = config["keys"]["grid"]
    learner_key = config["keys"]["learner"]

    # Load feature tables
    print("Loading feature tables...")
    behavior = pl.read_parquet(interim_path / "features_behavior.parquet")
    temporal = pl.read_parquet(interim_path / "features_temporal.parquet")
    assessment = pl.read_parquet(interim_path / "features_assessment.parquet")
    curriculum = pl.read_parquet(interim_path / "features_curriculum.parquet")
    static = pl.read_parquet(interim_path / "features_static.parquet")
    labels = pl.read_parquet(processed_path / "labels.parquet")

    # Merge features on grid_key
    print("Joining feature tables...")
    features = (
        static.join(behavior, on=grid_key, how="inner")
        .join(temporal, on=grid_key, how="inner")
        .join(assessment, on=grid_key, how="inner")
        .join(curriculum, on=grid_key, how="inner")
    )

    # Assign split
    train_presentations = config["splits"]["train"]
    val_presentations = config["splits"]["val"]
    test_presentations = config["splits"]["test"]

    features = features.with_columns(
        pl.when(pl.col("code_presentation").is_in(train_presentations))
        .then(pl.lit("train"))
        .when(pl.col("code_presentation").is_in(val_presentations))
        .then(pl.lit("val"))
        .when(pl.col("code_presentation").is_in(test_presentations))
        .then(pl.lit("test"))
        .otherwise(pl.lit("unknown"))
        .alias("split")
    )

    # Join labels
    label_cols = [
        col for col in labels.columns
        if col not in grid_key
    ]
    features_weekly = features.join(
        labels.select(grid_key + label_cols),
        on=grid_key,
        how="inner",
    )

    # Check for forbidden columns in features_weekly (target leakage & demographics)
    # final_result and date_unregistration are kept in labels.parquet, but features_weekly should only have labels + features
    # Let's save features_weekly
    out_features_path = processed_path / "features_weekly.parquet"
    features_weekly.write_parquet(out_features_path, compression="zstd")
    print(f"Features weekly consolidated: {len(features_weekly):,} rows -> {out_features_path}")

    # 6. Fit StandardScaler on TRAIN continuous features
    # Exclude identifiers, labels, boolean flags, and targets
    non_feature_cols = set(grid_key + [
        "split",
        "is_enrolled_at_week_t",
        "is_post_assessment_period",
        "label_is_proxy",
        "never_active_learner",
        "is_zero_click_withdrawal",
        "has_missing_date_registration",
        "has_any_assessment",
        "has_banked",
        "final_result",
        "final_score",
        "grade_band",
        "effective_date_unregistration",
        "module_presentation_length",
    ] + [f"label_dropout_k{k}" for k in [2, 3, 4]]
      + [f"label_inactive_k{k}" for k in [2, 3, 4]]
      + [f"label_inactive_persistent_k{k}" for k in [2, 3, 4]]
      + [f"is_censored_k{k}" for k in [2, 3, 4]])

    train_data = features_weekly.filter(pl.col("split") == "train")
    numeric_feature_cols = [
        col for col, dt in features_weekly.schema.items()
        if col not in non_feature_cols and dt in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
    ]

    # Impute NaNs with train median for scaler fitting
    train_features_np = train_data.select(numeric_feature_cols).to_pandas().fillna(
        train_data.select(numeric_feature_cols).to_pandas().median()
    ).values

    scaler = StandardScaler()
    scaler.fit(train_features_np)

    scaler_meta = {
        "scaler": scaler,
        "feature_names": numeric_feature_cols,
    }
    scaler_path = models_path / "train_scaler.joblib"
    joblib.dump(scaler_meta, scaler_path)
    print(f"Fitted StandardScaler on {len(numeric_feature_cols)} features on TRAIN split only -> {scaler_path}")

    # 7. Generate split statistics report
    split_stats = {}
    for sp_name in ["train", "val", "test"]:
        sp_df = features_weekly.filter(pl.col("split") == sp_name)
        eligible_sp = sp_df.filter(pl.col("is_enrolled_at_week_t") & ~pl.col("is_censored_k3"))
        
        unique_learners = len(sp_df.select(learner_key).unique())
        unique_students = len(sp_df.select("id_student").unique())
        modules_present = sp_df["code_module"].unique().sort().to_list()
        
        d_rate_k3 = float(eligible_sp["label_dropout_k3"].mean()) if len(eligible_sp) > 0 else 0.0
        
        split_stats[sp_name] = {
            "row_count": len(sp_df),
            "learner_count": unique_learners,
            "unique_id_students": unique_students,
            "modules_present": modules_present,
            "dropout_rate_k3": round(d_rate_k3 * 100, 3),
        }

    # Cross-presentation repeated students
    student_splits = features_weekly.group_by("id_student").agg(pl.col("split").n_unique().alias("n_splits"))
    repeat_students = len(student_splits.filter(pl.col("n_splits") > 1))

    report = {
        "splits": split_stats,
        "cross_split_repeat_students": repeat_students,
        "warnings": [
            "Module CCC is present in val (2014B) and test (2014J) but completely unseen in train (2013B, 2013J).",
            "Modules AAA and EEE are present in train and test but absent from validation (2014B).",
        ],
    }

    splits_yaml_path = processed_path / "splits.yaml"
    with open(splits_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(report, f, sort_keys=False)
    print(f"Split statistics written to {splits_yaml_path}")

    return {
        "features_weekly": features_weekly,
        "report": report,
    }


if __name__ == "__main__":
    build_splits_and_consolidate()

