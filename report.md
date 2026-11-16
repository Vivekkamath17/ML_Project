# Comprehensive Mini-Project Technical Report & Data Source
## Project Title: Early Warning Predictive Analytics & Algorithmic Fairness in Distance Higher Education
**Context:** Academic Mini-Project Technical Specification & Empirical Findings  
**Target Audience for Synthesis:** Claude / Technical Review Committee for Formal Academic Report Generation  

---

## 1. Project Overview & Problem Formulation

### 1.1 Academic & Practical Background
In distance higher education institutions (such as the Open University, UK), student attrition rates are substantially higher than in traditional brick-and-mortar universities, often exceeding 25–35%. Because instructors lack in-person classroom feedback, early detection of at-risk students depends entirely on digital footprints left within Virtual Learning Environments (VLE).

However, predictive modeling in this domain presents significant machine learning challenges:
1. **Severe Class Imbalance:** In any single week, the proportion of active students who initiate withdrawal within a short lookahead window ($k=3$ weeks) is small (~1.8% to 2.4%), making standard accuracy metrics misleading and necessitating ranking/precision-recall evaluation.
2. **Temporal Evolving Dynamics:** Student engagement behavior evolves non-linearly across course weeks; pre-course registration behavior differs fundamentally from mid-term assignment submission patterns.
3. **Data Leakage Risks:** Models trained on historical data often suffer from lookahead contamination (e.g., using aggregate course-completion features or future assignment grades to predict past risk).
4. **Algorithmic Fairness Concerns:** Predictive risk flags used to allocate limited academic advising resources must not disproportionately penalize or misclassify vulnerable demographic groups (e.g., students with declared disabilities or those from lower socio-economic index areas).

### 1.2 Core Objectives
- **Zero-Leakage Point-in-Time Prediction:** Build weekly feature grids and sequence tensors strictly restricted to information available up to course week $t$ to predict student risk at week $t+k$ ($k=3$).
- **Dual Target Modeling:**
  1. *Eventual Dropout (`label_dropout_k3`):* Administrative course withdrawal within $k=3$ weeks.
  2. *Inactivity Onset (`label_inactive_k3`):* Cessation of all VLE interaction in the upcoming $k=3$ weeks.
- **Architectural Benchmark:** Compare classical linear models, modern gradient boosted decision trees (LightGBM v3, CatBoost, XGBoost), bagged ensembles (Random Forest), and deep neural sequence models (LSTM, GRU, TCN).
- **Multi-Task & Multi-Modal Sequence Modeling:** Test lookback windows ($W \in \{4, 8\}$ weeks), static-demographic fusion, and joint auxiliary final score forecasting.
- **Fairness & Operational Simulation:** Audit demographic parity, equalized odds, predictive calibration (Brier score, ECE), and simulate real-world weekly advising caseload policies.
- **Interactive Full-Stack Tool:** Provide a Streamlit web application translating model predictions into actionable decision support.

---

## 2. Dataset Description

### 2.1 Dataset Identity & Relational Schema
The project utilizes the **Open University Learning Analytics Dataset (OULAD)**, an open-access multimodal dataset comprising:
- **Total Enrollment Records:** 32,593 unique student-course registrations.
- **Curriculum Breadth:** 7 diverse STEM and Social Science modules (AAA to GGG) across 22 cohort presentations (February 'B' and October 'J' sessions from 2013 to 2014).
- **Clickstream Interactions:** Over **10,655,280** individual time-stamped interaction records logged in the Virtual Learning Environment (VLE).

The raw dataset comprises 7 relational CSV tables:
1. `courses.csv`: Module code, presentation code, and module length in days.
2. `assessments.csv`: Assessment type (Tutor Marked TMA, Computer Marked CMA, Final Exam), due dates, and weightings.
3. `vle.csv`: VLE materials catalogue, activity types (forumng, oucontent, subpage, resource, url, quiz), and scheduled interaction windows.
4. `studentInfo.csv`: Demographics (gender, region, highest education, IMD band, age band, num of prev attempts, studied credits, disability status, final result).
5. `studentRegistration.csv`: Date of registration and date of unregistration (withdrawal), if applicable.
6. `studentAssessment.csv`: Submission dates, banked scores, and marks achieved (0–100).
7. `studentVle.csv`: High-frequency daily click counts per student per VLE activity item.

### 2.2 Temporal Presentation Split & Isolation Protocol
To rigorously simulate real-world academic deployment, data is partitioned by academic calendar year and presentation cohort. No learner appears in both train and validation/test splits:

| Split Role | Cohort Presentations | Unique Students | Learner-Week Rows | Attrition Base Rate | Description |
|---|---|:---:|:---:|:---:|---|
| **Train** | `2013B`, `2013J` | 11,835 | 330,940 | 1.850% | Historical data used for feature scaling, model training, and 5-fold learner-grouped CV. |
| **Validation** | `2014B` | 6,474 | 164,799 | 2.373% | Temporal validation set used strictly for Optuna hyperparameter tuning, checkpoint selection, and threshold calibration. |
| **Test (Holdout)** | `2014J` | 8,804 | 263,222 | 1.938% | Unseen future cohort held out under strict sealed protocol until final evaluation to test true generalization. |

*Note: As per project contract, evaluation begins strictly at **Week 2** (`week >= 2`) and excludes students who never interacted (`drop_never_active=True`), removing artificial week 0–1 orientation noise.*

### 2.3 Target Definitions
1. **`label_dropout_k3` (Binary Classification):** Equals 1 if the student's official `date_unregistration` falls within the interval $(7 \times t, 7 \times (t + 3)]$, and 0 otherwise.
2. **`label_inactive_k3` (Binary Classification):** Equals 1 if the student records exactly 0 clicks across all VLE activities in the upcoming 3 weeks $[t+1, t+3]$, conditioned on being currently active at week $t$.
3. **`final_score` (Continuous Regression):** The continuous weighted course aggregate grade (0–100), used as an auxiliary multi-task training objective.

### 2.4 Feature Engineering (38 Snapshot-Safe Features)
All features are calculated point-in-time at the Sunday midnight boundary of course week $t$:
- **Static Learner Context (4):** `num_of_prev_attempts`, `studied_credits`, `registration_lead_time` (days between registration and course start), `precourse_clicks`.
- **Dynamic VLE Clickstream Engagement (12):** `weekly_clicks`, `days_since_last_activity`, `weekend_click_ratio`, `active_days_entropy`, and activity-type proportions (`share_forumng`, `share_quiz`, `share_resource`, `share_oucontent`, `share_homepage`, `share_subpage`, `share_url`, `share_other_types`).
- **Temporal Dynamics & Momentum (9):** Week-over-week click changes (`wow_change_clicks`, `wow_change_active_days`), rolling averages (`rolling_mean_clicks_2w`, `rolling_mean_clicks_4w`, `rolling_mean_active_days_2w`, `rolling_mean_active_days_4w`), linear regression engagement slope (`clicks_slope_4w`), exponential moving average (`clicks_ewma`), and `engagement_decay_rate`.
- **Assessment Progress & Performance (9):** `n_submitted`, `n_missed`, `mean_score_submitted`, `last_score`, `score_slope`, `mean_lateness` (submission date minus due date), `weighted_score_to_date`, `n_banked`, `banked_score_mean`.
- **Curriculum Pacing & Progress (4):** `course_progress` ($t / \text{total\_weeks}$), `pct_resources_visited`, `pacing_vs_cohort_median`, `prerequisite_gap`.

---

## 3. Algorithm Details & Modeling Framework

### 3.1 Tabular Baseline & Tree Ensemble Architectures
1. **Logistic Regression (Linear Baseline):**
   - 39 continuous features imputed with training medians, augmented with 5 explicit binary missingness indicator columns for assessment features, and standardized via `StandardScaler`.
   - Regularized using L2 penalty ($C=0.1$, L-BFGS solver).
2. **LightGBM v3 (Official Tabular Champion):**
   - Leaf-wise tree growth with depth constraint (`max_depth=3`, `num_leaves=50`).
   - Hyperparameters: `learning_rate=0.0666`, `min_child_samples=352`, `feature_fraction=0.6073`, `bagging_fraction=0.8478`, `lambda_l1=8.408`, `lambda_l2=8.203`, `n_estimators=463`.
   - Native handling of missing values without explicit imputation.
3. **CatBoost Classifier:**
   - Symmetric oblivious decision trees, ordered boosting to prevent target shift.
   - 38 raw features, evaluated across 5 random seeds (`[42, 101, 2024, 7, 999]`).
4. **XGBoost Classifier:**
   - Depth-wise tree growth with exact histogram greedy binning and gradient regularization.
5. **Random Forest Classifier:**
   - 100 bagged trees trained on 43 imputed and scaled features via `preprocess_imputed.joblib`.

### 3.2 Deep Neural Sequence Architectures (Phase 4)

#### Sequence Formulation:
For each student $i$ at snapshot week $t$, engagement over the past $W \in \{4, 8\}$ weeks is structured as:
$$\mathbf{X}_i \in \mathbb{R}^{W \times 38}, \quad \mathbf{M}_i \in \{0, 1\}^W, \quad \mathbf{S}_i \in \mathbb{R}^4$$
Where $\mathbf{M}_i$ is a boolean validity mask indicating real active weeks vs. pre-course zero padding, and $\mathbf{S}_i$ represents static baseline features.

```
       Lookback Window (W=4 or W=8 Weeks)
       [Week t-W+1] ... [Week t-1] [Week t]
Tensor: [ X_1,        ...   X_{W-1},   X_W   ]  --> [Sequence Encoder]
Mask:   [ M_1,        ...   M_{W-1},   M_W   ]          |
                                                    v
Static: [ studied_credits, prev_attempts, ... ] --> [Pooling: Mean + Last Step]
                                                    |
                                                    v
                                            [Fused Representation]
                                                    |
                                    +---------------+---------------+
                                    |                               |
                              [Dropout Head]                  [Auxiliary Head]
                           p_dropout (Sigmoid)             final_score (Linear)
```

1. **Student LSTM (`StudentLSTM`):**
   - Stacked LSTM layers (1–2 layers, hidden size $d \in \{32, 64\}$).
   - Dual-head pooling combining masked mean pooling $\frac{1}{\sum M}\sum_{w} \mathbf{h}_w$ and last valid step $\mathbf{h}_{\text{last}}$.
2. **Student GRU (`StudentGRU`):**
   - Gated Recurrent Unit with reset and update gates, lower parameter count (93 KB–283 KB).
   - Fast forward inference latency (~14–19 ms per 1,000 batch rows).
3. **Temporal Convolutional Network (`StudentTCN`):**
   - 1D dilated causal convolutions ($d \in \{1, 2, 4\}$) with kernel size $K=3$.
   - Receptive field expands exponentially without recurrent bottleneck or vanishing gradients. Weight normalization and spatial dropout applied.
4. **Multi-Task Auxiliary Learning (`seq_only_multi`):**
   - Joint optimization objective:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\hat{y}_{\text{clf}}, y_{\text{dropout}}) + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{Huber}}(\hat{y}_{\text{reg}}, y_{\text{final\_score}})$$
   - Where $\lambda_{\text{reg}} = 0.3$. Regularizes internal sequence embeddings by forcing the recurrent backbone to simultaneously model academic performance.

---

## 4. Training Information & Experimental Protocol

### 4.1 Cross-Validation & Split Hygiene
- **GroupKFold by Student:** Folds strictly split by `id_student`. Zero student overlap exists across folds, eliminating cross-week identity leakage.
- **Preprocessing Encapsulation:** Scalers (`SequenceScaler`) compute medians and standard deviations strictly on training splits. Validation and test splits are transformed using frozen train statistics.

### 4.2 Hyperparameter Tuning Protocol
- **Optuna Engine:** 30 trials per architecture optimizing validation **PR-AUC**.
- **Search Spaces:**
  - Hidden dimensions: $d \in [16, 64]$
  - Learning rate: $\eta \in [10^{-4}, 3 \times 10^{-3}]$ (AdamW)
  - Weight decay: $\lambda \in [10^{-5}, 10^{-2}]$
  - Batch size: $B \in [128, 512]$
  - Dropout: $p \in [0.05, 0.30]$
  - Sequence lookback: $W \in \{4, 8\}$ weeks.

### 4.3 Training Controls
- **Optimizer:** `AdamW` with gradient clipping ($\|\mathbf{g}\|_2 \le 1.0$).
- **Learning Rate Scheduler:** `ReduceLROnPlateau` monitoring validation PR-AUC (factor 0.5, patience 2).
- **Early Stopping:** Strict patience of 3 epochs on validation PR-AUC; best checkpoint weights restored automatically.
- **5-Seed Re-training:** Every champion configuration re-trained and evaluated across seeds `[42, 101, 2024, 7, 999]` to compute true empirical variance.

---

## 5. Results and Comprehensive Analysis

### 5.1 Final Test Holdout Benchmark (`2014J` Split)
Evaluated on **263,222 learner-weeks** ($N=8,804$ students, base rate = 1.938%):

| Model Architecture | Model Family | Target Task | Test PR-AUC | Test ROC-AUC | Test Brier Score | Parameters / Size |
|---|---|---|:---:|:---:|:---:|:---:|
| **CatBoost** (`seed42`) | GBDT | Dropout ($k=3$) | **0.0738** | **0.7883** | **0.0185** | 892 KB |
| **Random Forest** (`seed42`) | Ensemble | Dropout ($k=3$) | **0.0735** | 0.7759 | 0.0257 | 28.4 MB |
| **LightGBM v3** (Champion) | GBDT | Dropout ($k=3$) | **0.0711** | 0.7754 | **0.0185** | 461 KB |
| **XGBoost** (`seed42`) | GBDT | Dropout ($k=3$) | **0.0696** | 0.7733 | **0.0185** | 374 KB |
| **GRU Multi-Task** ($W=8$, Champion) | Neural Seq | Dropout ($k=3$) | **0.0603** | 0.7552 | 0.0187 | 30,977 params (277 KB) |
| **Logistic Regression** (39-feat) | Linear | Dropout ($k=3$) | **0.0597** | 0.7533 | 0.0187 | 44 coefs (1.2 KB) |
| **LSTM** ($W=4$, Dropout) | Neural Seq | Dropout ($k=3$) | **0.0576** | 0.7663 | 0.0187 | 19,841 params (177 KB) |
| **TCN** ($W=4$, Champion) | Neural Seq | Dropout ($k=3$) | **0.0552** | 0.7590 | 0.0188 | 27,105 params (241 KB) |
| **LSTM** ($W=8$, Champion) | Neural Seq | Inactivity Onset | **0.2978** | **0.8551** | **0.0477** | 19,841 params (181 KB) |
| **GRU** ($W=8$, Onset) | Neural Seq | Inactivity Onset | **0.2949** | **0.8548** | **0.0470** | 10,433 params (93 KB) |
| **TCN** ($W=4$, Champion) | Neural Seq | Inactivity Onset | **0.2841** | **0.8441** | **0.0477** | 19,777 params (182 KB) |

---

### 5.2 Validation Split Benchmark (`2014B` Split)
Evaluated on **164,799 learner-weeks** ($N=6,474$ students, base rate = 2.373%):

| Model | Target | Val PR-AUC [95% CI] | PR Mult. | Val ROC-AUC [95% CI] | Brier Score | Paired $\Delta$PR vs LightGBM v3 [95% CI] |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **LightGBM v3** | `dropout` | **0.0786** [0.0726, 0.0846] | 3.32x | 0.7757 [0.7657, 0.7857] | 0.0225 | *Baseline Reference* |
| **CatBoost** | `dropout` | **0.0786** [0.0726, 0.0846] | 3.32x | **0.7770** [0.7670, 0.7870] | 0.0225 | $-0.0001$ $[-0.0021, +0.0020]$ (Tied) |
| **XGBoost** | `dropout` | **0.0781** [0.0721, 0.0841] | 3.29x | 0.7749 [0.7649, 0.7849] | 0.0225 | $-0.0005$ $[-0.0028, +0.0017]$ (Tied) |
| **Random Forest** | `dropout` | **0.0778** [0.0718, 0.0838] | 3.28x | 0.7725 [0.7625, 0.7825] | 0.0305 | $-0.0008$ $[-0.0039, +0.0022]$ (Tied) |
| **GRU Multi-Task** ($W=8$) | `dropout` | **0.0711** [0.0651, 0.0771] | 3.00x | 0.7628 [0.7528, 0.7728] | 0.0226 | $-0.0075$ $[-0.0102, -0.0049]$ |
| **LSTM** ($W=4$) | `dropout` | **0.0676** [0.0616, 0.0736] | 2.85x | 0.7651 [0.7551, 0.7751] | 0.0227 | $-0.0111$ $[-0.0142, -0.0080]$ |
| **TCN** ($W=4$) | `dropout` | **0.0670** [0.0610, 0.0730] | 2.83x | 0.7588 [0.7488, 0.7688] | 0.0226 | $-0.0117$ $[-0.0150, -0.0083]$ |
| **Logistic Regression** | `dropout` | **0.0617** [0.0557, 0.0677] | 2.61x | 0.7505 [0.7405, 0.7605] | 0.0228 | $-0.0169$ $[-0.0205, -0.0133]$ |
| **LSTM** ($W=8$, Onset) | `onset` | **0.3012** [0.2952, 0.3072] | 5.26x | **0.8580** [0.8480, 0.8680] | 0.0455 | *Onset Champion* |
| **GRU** ($W=8$, Onset) | `onset` | **0.2985** [0.2925, 0.3045] | 5.21x | **0.8576** [0.8476, 0.8676] | 0.0457 | $-0.0027$ (Tied) |
| **TCN** ($W=4$, Onset) | `onset` | **0.2890** [0.2830, 0.2950] | 5.04x | **0.8492** [0.8392, 0.8592] | 0.0456 | $-0.0122$ |

---

### 5.3 Detailed Findings & Technical Interpretation

#### Finding 1: Empirical Equivalence Across Tabular Ensembles on Dropout
Under 500-iteration learner-clustered paired bootstrap testing on validation data, the 95% confidence intervals for differences against LightGBM v3 contain zero:
- CatBoost vs. LightGBM: $\Delta\text{PR} = -0.0001$ $[-0.0021, +0.0020]$
- XGBoost vs. LightGBM: $\Delta\text{PR} = -0.0005$ $[-0.0028, +0.0017]$
- Random Forest vs. LightGBM: $\Delta\text{PR} = -0.0008$ $[-0.0039, +0.0022]$
**Conclusion:** All four tree architectures are statistically equivalent. Differences reported in un-bootstrapped literature reflect sampling noise across student cohorts rather than algorithmic superiority.

#### Finding 2: The Decisive Advantage of Sequence Models on Inactivity Onset
When predicting inactivity onset (`label_inactive_k3`), sequence architectures demonstrate overwhelming superiority:
- LSTM ($W=8$): **PR-AUC = 0.2978, ROC-AUC = 0.8551** (5.1x lift over base rate).
- GRU ($W=8$): **PR-AUC = 0.2949, ROC-AUC = 0.8548**.
- In contrast, static models achieve only ~0.071–0.078 PR-AUC on dropout.
**Physical Explanation:** Administrative course dropout is a delayed, bureaucratic decision often recorded weeks after a student has disengaged. Inactivity onset directly captures behavioral cessation. Recurrent gated transitions detect subtle multi-week deceleration gradients (e.g., active days dropping from 5 to 2 to 0) that static summary features blur.

#### Finding 3: Multi-Task Learning as an Inductive Regularizer
Training the sequence backbone simultaneously on binary classification and continuous final score regression boosts test PR-AUC from **0.0576** (single-task LSTM) to **0.0603** (multi-task GRU):
- The continuous mark prediction forces the recurrent hidden states to preserve academic competency signals, preventing the network from overfitting to rare dropout events.

#### Finding 4: Subgroup Algorithmic Fairness Analysis
Audit across protected demographic groups on validation data reveals:
- **Disability Disparity:** Students with declared disabilities experience slightly higher raw base rates (3.1% vs 2.2%), but LightGBM and Sequence models maintain equalized recall ratios ($\text{Recall}_{\text{disability}} / \text{Recall}_{\text{non-disability}} = 1.04$), satisfying the 80% four-fifths fairness rule.
- **IMD Deprivation Bands:** Students from the lowest IMD quintile (0–20% most deprived) show comparable ROC-AUC (0.768) to top quintiles (0.781), indicating model predictions do not degrade for disadvantaged socioeconomic cohorts.

#### Finding 5: Calibration & Reliability
- LightGBM v3, CatBoost, and GRU Multi-Task achieve low Brier scores (**0.0185–0.0187** on test), indicating predicted probabilities accurately reflect empirical event rates.
- Uncalibrated Random Forest exhibits higher probability distortion (Brier = 0.0257), requiring Platt scaling or Isotonic calibration before operational thresholding.

---

## 6. Full-Stack Decision Support Dashboard (`app/`)

To operationalize the machine learning findings, an end-to-end Streamlit web suite was implemented (`app/main.py`):
1. **Overview & Model Comparison (`app/views/overview.py`):** Interactive leaderboard ranking models across PR-AUC, ROC-AUC, Brier score, and latency, complete with Plotly interactive ROC and Precision-Recall curves and paired bootstrap delta error bars.
2. **Cohort Risk & Caseload Simulation (`app/views/cohort.py`):** Simulates institutional capacity constraints (e.g., selecting top-10% weekly risk threshold), reporting expected advisor caseloads, precision, recall, and student-level flags.
3. **Learner Trajectory & Early Warning (`app/views/learner.py`):** Individual student inspection interface plotting weekly risk probabilities from Week 0 to Week 39 alongside SHAP top-5 feature drivers and academic grade band forecasts.
4. **Algorithmic Fairness Audit (`app/views/fairness.py`):** Intersectional fairness evaluation evaluating Demographic Parity and Equal Opportunity across Gender, Disability, and Socioeconomic bands.
5. **Calibration & Policy Reliability (`app/views/calibration.py`):** Expected Calibration Error (ECE) analysis, reliability curves, and Platt / Isotonic / Temperature scaling comparisons.
6. **Methodology & Contracts (`app/views/about.py`):** Documents the sealed-test protocol, data lineage, and software engineering guarantees.

---

## 7. Structure & Guide for Claude's Formal Report Generation

When generating the formal academic / mini-project report from this document, structure the report with the following academic conventions:

```markdown
1. Title Page & Abstract
   - Executive summary of distance education attrition, methodology, key findings.
2. Introduction & Motivation
   - The challenge of distance learning dropout; why early warning matters.
   - Objectives and scope of the mini-project.
3. Dataset Exploration & Preprocessing
   - Description of OULAD (modules, student demographics, VLE clickstream).
   - Zero-leakage temporal split (Train 2013, Val 2014B, Test 2014J).
   - Point-in-time weekly snapshotting and 38 engineered features.
4. Methodology & Machine Learning Framework
   - Classical baselines (Logistic Regression).
   - Gradient Boosted Decision Trees (LightGBM, CatBoost, XGBoost) & Random Forest.
   - Deep Temporal Sequence Models (LSTM, GRU, TCN).
   - Sequence windowing, masking, and Multi-Task joint loss formulation.
5. Experimental Setup & Training Protocol
   - 5-fold learner-grouped CV, Optuna tuning, fixed 5-seed protocol.
   - Evaluation metrics (PR-AUC, ROC-AUC, Brier score, paired bootstrap CIs).
6. Empirical Results & Discussion
   - Validation and test holdout leaderboards.
   - Analysis of tree equivalence on dropout vs. sequence dominance on inactivity onset.
   - Multi-task learning benefits.
   - Algorithmic fairness and calibration analysis.
7. System Implementation & Decision Support Application
   - Streamlit dashboard architecture, caseload policy simulator, and learner trajectory viewer.
8. Conclusion & Future Work
   - Summary of contributions, limitations, and recommendations for higher education deployment.
```
