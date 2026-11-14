import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from src.models.data import load_split
from src.models.harness import paired_bootstrap

# 1. Paired bootstrap v3 vs v1 on 2014B
X_va, y_va, meta_va = load_split('val', drop_never_active=True)
v1_model = lgb.Booster(model_file='models/baselines/lgbm_champion_v1.txt')
v3_model = lgb.Booster(model_file='models/baselines/lgbm_champion_v3_official.txt')

p_v1 = v1_model.predict(X_va)
v3_feats = [c for c in X_va.columns if c != 'n_skipped_resources']
p_v3 = v3_model.predict(X_va[v3_feats])

roc_v1 = roc_auc_score(y_va, p_v1)
pr_v1 = average_precision_score(y_va, p_v1)
roc_v3 = roc_auc_score(y_va, p_v3)
pr_v3 = average_precision_score(y_va, p_v3)
print(f"2014B Val: v1 ROC={roc_v1:.4f}, PR={pr_v1:.4f} | v3 ROC={roc_v3:.4f}, PR={pr_v3:.4f}")

boot_res = paired_bootstrap(
    y_true=y_va,
    y_prob_a=p_v3,
    y_prob_b=p_v1,
    cluster_ids=meta_va['id_student'],
    n_boot=500,
    seed=42
)
print("Paired bootstrap v3 minus v1 on 2014B:")
print(f"  Delta ROC: {boot_res['delta_roc']:+.4f} [95% CI: {boot_res['delta_roc_ci'][0]:+.4f}, {boot_res['delta_roc_ci'][1]:+.4f}]")
print(f"  Delta PR:  {boot_res['delta_pr']:+.4f} [95% CI: {boot_res['delta_pr_ci'][0]:+.4f}, {boot_res['delta_pr_ci'][1]:+.4f}]")

# 2. Per-fold LOPO for v3 vs v1
X_tr, y_tr, meta_tr = load_split('train', drop_never_active=True)
X_all = pd.concat([X_tr, X_va], ignore_index=True)
y_all = np.concatenate([y_tr, y_va])
meta_all = pd.concat([meta_tr, meta_va], ignore_index=True)

presentations = ['2013B', '2013J', '2014B']
print("\n=== Per-Fold LOPO Results (v3 vs v1) ===")
lopo_records = []
for p in presentations:
    test_idx = (meta_all['code_presentation'] == p).values
    train_idx = ~test_idx
    
    y_train_fold, y_test_fold = y_all[train_idx], y_all[test_idx]
    
    # Train v1 (39 features)
    dtrain_v1 = lgb.Dataset(X_all.iloc[train_idx], label=y_train_fold)
    params_v1 = {'objective': 'binary', 'learning_rate': 0.05, 'num_leaves': 31, 'verbosity': -1, 'random_state': 42}
    clf_v1 = lgb.train(params_v1, dtrain_v1, num_boost_round=285)
    pred_v1 = clf_v1.predict(X_all.iloc[test_idx])
    
    # Train v3 (38 features, without n_skipped_resources)
    dtrain_v3 = lgb.Dataset(X_all.iloc[train_idx][v3_feats], label=y_train_fold)
    params_v3 = {'objective': 'binary', 'learning_rate': 0.0666, 'num_leaves': 50, 'max_depth': 3, 'min_child_samples': 352, 'verbosity': -1, 'random_state': 42}
    clf_v3 = lgb.train(params_v3, dtrain_v3, num_boost_round=463)
    pred_v3 = clf_v3.predict(X_all.iloc[test_idx][v3_feats])
    
    r_v1 = roc_auc_score(y_test_fold, pred_v1)
    p_v1_s = average_precision_score(y_test_fold, pred_v1)
    r_v3 = roc_auc_score(y_test_fold, pred_v3)
    p_v3_s = average_precision_score(y_test_fold, pred_v3)
    
    delta_r = r_v3 - r_v1
    delta_p = p_v3_s - p_v1_s
    print(f"Fold {p}: v1 ROC={r_v1:.4f}, PR={p_v1_s:.4f} | v3 ROC={r_v3:.4f}, PR={p_v3_s:.4f} | Delta ROC={delta_r:+.4f}, Delta PR={delta_p:+.4f}")
    lopo_records.append({'fold': p, 'v1_roc': r_v1, 'v1_pr': p_v1_s, 'v3_roc': r_v3, 'v3_pr': p_v3_s, 'delta_roc': delta_r, 'delta_pr': delta_p})

df_lopo = pd.DataFrame(lopo_records)
print(f"\nLOPO Mean v1: ROC={df_lopo['v1_roc'].mean():.4f}, PR={df_lopo['v1_pr'].mean():.4f}")
print(f"LOPO Mean v3: ROC={df_lopo['v3_roc'].mean():.4f}, PR={df_lopo['v3_pr'].mean():.4f}")
print(f"LOPO Mean Delta (v3 - v1): ROC={df_lopo['delta_roc'].mean():+.4f}, PR={df_lopo['delta_pr'].mean():+.4f}")
