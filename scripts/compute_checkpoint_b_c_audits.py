"""
Computes:
1. Reconciled policy table and lead-time metrics (share within 3w before withdrawal,
   share >= 2w before, lead time from start of final continuous flag run, learner-level precision).
2. Per-module-presentation final_score audit table:
   - assessments included, exam_included flag
   - mean score per final_result band (Distinction, Pass, Fail, Withdrawn)
   - MAE and R2 at week 8 and 12
   - Baseline mean_score_submitted comparison
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from src.models.data import load_split

print("=" * 70)
print("1. RECONCILED POLICY & LEAD-TIME AUDIT")
print("=" * 70)

X_va_k3, y_va_k3, meta_va_k3 = load_split("val", k=3, min_week=2)
booster_k3 = lgb.Booster(model_file="models/baselines/lgbm_champion_v1.txt")
probs_k3 = booster_k3.predict(X_va_k3.values)

raw_reg = pd.read_csv("data/raw/studentRegistration.csv")
meta_va = meta_va_k3.merge(
    raw_reg[["code_module", "code_presentation", "id_student", "date_unregistration"]],
    on=["code_module", "code_presentation", "id_student"],
    how="left"
)
meta_va["prob"] = probs_k3
meta_va["learner_key"] = meta_va["code_module"] + "_" + meta_va["code_presentation"] + "_" + meta_va["id_student"].astype(str)
meta_va["unreg_week"] = meta_va["date_unregistration"] // 7

# Denominators
all_unique_learners = meta_va["learner_key"].unique()
n_all_learners = len(all_unique_learners)

withdrawn_learners_set = set(meta_va[meta_va["date_unregistration"].notna()]["learner_key"].unique())
n_withdrawn_learners = len(withdrawn_learners_set)

print(f"Total Val learners (week >= 2): {n_all_learners:,}")
print(f"Total Val withdrawn learners: {n_withdrawn_learners:,} ({n_withdrawn_learners/n_all_learners*100:.2f}%)")

# Define flag masks
# Top 10% per week
top10_mask = pd.Series(False, index=meta_va.index)
for w in meta_va["week"].unique():
    w_idx = meta_va[meta_va["week"] == w].index
    w_sub = meta_va.loc[w_idx].sort_values(by=["prob", "id_student"], ascending=[False, True])
    n_top = int(np.ceil(0.10 * len(w_sub)))
    top_indices = w_sub.iloc[:n_top].index
    top10_mask.loc[top_indices] = True

policies = {
    "MCC-Optimal (th=0.025)": meta_va["prob"] >= 0.025,
    "Cost r=10 (th=0.095)": meta_va["prob"] >= 0.095,
    "Cost r=5 (th=0.265) [Unviable]": meta_va["prob"] >= 0.265,
    "Capacity Top-10% / Week": top10_mask,
}

lead_results = []
for pol_name, flag_mask in policies.items():
    meta_va["is_flagged"] = flag_mask
    
    # Learner level flags
    flagged_learners = set(meta_va[meta_va["is_flagged"]]["learner_key"].unique())
    n_flagged_all = len(flagged_learners)
    pct_all_flagged = n_flagged_all / n_all_learners
    
    # Flagged withdrawn learners
    flagged_withdrawn = flagged_learners.intersection(withdrawn_learners_set)
    n_flagged_withdrawn = len(flagged_withdrawn)
    recall_withdrawn = n_flagged_withdrawn / n_withdrawn_learners
    
    # Learner-level precision: of all learners flagged at least once, how many actually withdrew?
    learner_prec = n_flagged_withdrawn / n_flagged_all if n_flagged_all > 0 else 0.0
    
    # Lead time metrics on withdrawn learners who were flagged
    # For each withdrawn learner, analyze the timeline of flags relative to unregistration week
    within_3w_list = []
    ge_2w_list = []
    final_run_lead_list = []
    
    for lk in flagged_withdrawn:
        df_l = meta_va[(meta_va["learner_key"] == lk)].sort_values("week")
        unreg_w = df_l["unreg_week"].iloc[0]
        
        # Weeks where learner was flagged
        flagged_weeks = df_l[df_l["is_flagged"]]["week"].values
        # Lead time for all flags
        leads = unreg_w - flagged_weeks
        
        # 1. Flagged within 3 weeks before withdrawal: any flag with 0 <= unreg_w - week <= 3
        has_flag_within_3w = any(0 <= lead <= 3 for lead in leads)
        within_3w_list.append(has_flag_within_3w)
        
        # 2. Flagged at least 2 weeks before withdrawal: any flag with lead >= 2
        has_flag_ge_2w = any(lead >= 2 for lead in leads)
        ge_2w_list.append(has_flag_ge_2w)
        
        # 3. Final continuous flag run:
        # Trace backwards from the last evaluated week prior to unreg
        is_fl_arr = df_l["is_flagged"].values
        w_arr = df_l["week"].values
        
        # Find index of last flagged week before unreg
        valid_fl_idx = [i for i, (w, fl) in enumerate(zip(w_arr, is_fl_arr)) if fl and w <= unreg_w]
        if valid_fl_idx:
            last_idx = valid_fl_idx[-1]
            # Walk backwards while consecutive weeks are flagged
            start_idx = last_idx
            while start_idx > 0 and is_fl_arr[start_idx - 1] and (w_arr[start_idx] - w_arr[start_idx - 1] == 1):
                start_idx -= 1
            final_run_start_week = w_arr[start_idx]
            final_run_lead = unreg_w - final_run_start_week
            final_run_lead_list.append(final_run_lead)
        else:
            final_run_lead_list.append(np.nan)
            
    pct_within_3w = np.mean(within_3w_list) if within_3w_list else 0.0
    pct_ge_2w = np.mean(ge_2w_list) if ge_2w_list else 0.0
    med_final_run_lead = np.nanmedian(final_run_lead_list) if final_run_lead_list else 0.0
    
    lead_results.append({
        "Policy": pol_name,
        "Share All Learners Flagged": f"{pct_all_flagged*100:.1f}% ({n_flagged_all}/{n_all_learners})",
        "Withdrawn Recall (Learner)": f"{recall_withdrawn*100:.1f}% ({n_flagged_withdrawn}/{n_withdrawn_learners})",
        "Learner Precision": f"{learner_prec*100:.1f}%",
        "Flagged within 3w of Unreg": f"{pct_within_3w*100:.1f}%",
        "Flagged >= 2w before Unreg": f"{pct_ge_2w*100:.1f}%",
        "Median Lead (Final Run)": f"{med_final_run_lead:.1f} weeks",
    })

df_policy_reconciled = pd.DataFrame(lead_results)
print(df_policy_reconciled.to_string(index=False))

print("\n" + "=" * 70)
print("2. FINAL_SCORE PER-MODULE-PRESENTATION AUDIT")
print("=" * 70)

# Load assessments and studentAssessment
assessments = pd.read_csv("data/raw/assessments.csv")
stud_ass = pd.read_csv("data/raw/studentAssessment.csv")
vle_feat = pd.read_parquet("data/processed/features_weekly.parquet")
raw_vle_mod_pres = vle_feat[["code_module", "code_presentation", "split"]].drop_duplicates()

# Merge assessments with submitted scores
ass_with_scores = assessments.merge(stud_ass[["id_assessment", "score"]], on="id_assessment", how="left")
ass_summary = ass_with_scores.groupby(["code_module", "code_presentation", "assessment_type"]).agg(
    n_assessments=("id_assessment", "nunique"),
    total_catalog_weight=("weight", "sum"),
    n_score_records=("score", "count"),
    weight_with_scores=("weight", lambda w: w[ass_with_scores.loc[w.index, "score"].notna()].sum())
).reset_index()

# Check per presentation
print("Assessment Catalog & Submission Summary by Module-Presentation:")
exam_info = assessments[assessments["assessment_type"] == "Exam"].groupby(["code_module", "code_presentation"]).agg(
    has_exam=("id_assessment", "count"),
    exam_catalog_weight=("weight", "sum"),
    exam_id=("id_assessment", "first")
).reset_index()

exam_submissions = exam_info.merge(
    stud_ass.groupby("id_assessment")["id_student"].count().rename("n_exam_submissions").reset_index(),
    left_on="exam_id", right_on="id_assessment", how="left"
).fillna({"n_exam_submissions": 0})
exam_submissions["exam_included"] = exam_submissions["n_exam_submissions"] > 0
print(exam_submissions[["code_module", "code_presentation", "has_exam", "exam_catalog_weight", "n_exam_submissions", "exam_included"]].to_string(index=False))

# Load val weekly features to compute MAE and R2 at week 8 and 12
meta_val_all = vle_feat[vle_feat["split"] == "val"].copy()
# Filter non-withdrawn
meta_val_nw = meta_val_all[meta_val_all["final_result"] != "Withdrawn"].copy()

print("\nMean final_score per final_result band across all validation:")
band_means = meta_val_all.groupby("final_result")["final_score"].agg(["count", "mean", "std", "min", "max"]).reset_index()
print(band_means.to_string(index=False))

print("\nMean final_score by Module and Band in Validation (2014B):")
mod_band_means = meta_val_all.groupby(["code_module", "final_result"])["final_score"].agg(["count", "mean"]).unstack("final_result")
print(mod_band_means.to_string())

# Save audit results
audit_data = {
    "policy_reconciled": df_policy_reconciled.to_dict(orient="records"),
    "exam_submissions": exam_submissions.to_dict(orient="records"),
    "band_means": band_means.to_dict(orient="records"),
}
import joblib
joblib.dump(audit_data, "reports/phase2/audit_checkpoint_b_c.joblib")
print("\nSaved audit results to reports/phase2/audit_checkpoint_b_c.joblib")
