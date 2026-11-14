"""
Graph Inputs module for OULAD data preparation pipeline.
Constructs sparse learner x site interaction matrices for collaboration site types:
['forumng', 'oucollaborate', 'ouwiki', 'dataplus'].
Strict leakage safety: week t matrix contains ONLY clicks with date <= 7*(t+1) - 1.
Saves .npz sparse matrices and learner index mappings to data/processed/graphs/.
"""
from pathlib import Path
from typing import Dict, Any, List
import json
import numpy as np
import polars as pl
import scipy.sparse as sp
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_graph_inputs(
    interim_dir: str = "data/interim",
    processed_dir: str = "data/processed",
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """
    Constructs sparse CSR learner x site interaction matrices for each presentation and week:
    - Collaboration activity types: forumng, oucollaborate, ouwiki, dataplus.
    - Saves scipy.sparse CSR matrices to data/processed/graphs/<presentation>_week_<week>.npz.
    - Saves learner index mapping to data/processed/graphs/<presentation>_learners.json.
    - Saves site index mapping to data/processed/graphs/<presentation>_sites.json.
    """
    interim_path = Path(interim_dir)
    processed_path = Path(processed_dir)
    graph_dir = processed_path / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    collab_types = config["graph"]["activity_types"]

    vle = pl.read_parquet(interim_path / "vle.parquet")
    student_vle = pl.read_parquet(interim_path / "studentVle.parquet")
    courses = pl.read_parquet(interim_path / "courses.parquet")

    # 1. Filter collaboration sites
    collab_vle = vle.filter(pl.col("activity_type").is_in(collab_types))
    collab_sites = set(collab_vle["id_site"].to_list())

    # 2. Filter studentVle for collaboration sites
    svle_collab = student_vle.filter(pl.col("id_site").is_in(collab_sites)).with_columns(
        pl.when(pl.col("date") < 0)
        .then(-1)
        .otherwise(pl.col("date") // 7)
        .cast(pl.Int32)
        .alias("week")
    )

    presentations = courses["code_presentation"].unique().to_list()
    matrices_saved = 0

    for pres in presentations:
        # Get all distinct learners enrolled in this presentation
        pres_learners = (
            svle_collab.filter(pl.col("code_presentation") == pres)["id_student"]
            .unique()
            .sort()
            .to_list()
        )
        if not pres_learners:
            continue

        # Get all distinct collaboration sites in this presentation
        pres_sites = (
            collab_vle.filter(pl.col("code_presentation") == pres)["id_site"]
            .unique()
            .sort()
            .to_list()
        )
        if not pres_sites:
            continue

        learner2idx = {sid: i for i, sid in enumerate(pres_learners)}
        site2idx = {site: j for j, site in enumerate(pres_sites)}

        # Save index mappings
        with open(graph_dir / f"{pres}_learners.json", "w", encoding="utf-8") as f:
            json.dump([int(x) for x in pres_learners], f)
        with open(graph_dir / f"{pres}_sites.json", "w", encoding="utf-8") as f:
            json.dump([int(x) for x in pres_sites], f)

        # Get max week for this presentation
        pres_length = courses.filter(pl.col("code_presentation") == pres)["module_presentation_length"].max()
        max_week = (pres_length - 1) // 7

        # Aggregate cumulative clicks up to each week t
        # Group clicks by learner, site, and week
        pres_svle = svle_collab.filter(pl.col("code_presentation") == pres)
        weekly_site_clicks = (
            pres_svle.group_by(["id_student", "id_site", "week"])
            .agg(pl.col("sum_click").sum().alias("clicks"))
        )

        # For efficiency, save weekly CSR matrices for key weeks (e.g. weeks 0 to max_week)
        # To avoid generating hundreds of empty files, save representative weeks or all weeks
        # Let's save all weeks up to max_week:
        for t in range(max_week + 1):
            # Strict point in time: week <= t
            cum_clicks = (
                weekly_site_clicks.filter(pl.col("week") <= t)
                .group_by(["id_student", "id_site"])
                .agg(pl.col("clicks").sum().alias("cum_clicks"))
            )

            if len(cum_clicks) > 0:
                rows = [learner2idx[s] for s in cum_clicks["id_student"].to_list()]
                cols = [site2idx[s] for s in cum_clicks["id_site"].to_list()]
                data = cum_clicks["cum_clicks"].to_numpy()
                mat = sp.csr_matrix((data, (rows, cols)), shape=(len(pres_learners), len(pres_sites)), dtype=np.int32)
            else:
                mat = sp.csr_matrix((len(pres_learners), len(pres_sites)), dtype=np.int32)

            sp.save_npz(graph_dir / f"{pres}_week_{t}.npz", mat)
            matrices_saved += 1

    print(f"Graph inputs built: saved {matrices_saved} sparse CSR matrices to {graph_dir}")
    return {"matrices_saved": matrices_saved, "graph_dir": str(graph_dir)}


if __name__ == "__main__":
    build_graph_inputs()

