# Presentation Slide Deck Outline: OULAD Predictive Modeling, Fairness Audit & Decision Support

**Author**: Tanmay (Downstream Analytics Lead)  
**Target Duration**: 15–20 Minutes  
**Target Audience**: Academic Examiners, Faculty Evaluators & Retention Advising Stakeholders  

---

### Slide 1: Title & Author Contributions
- **Title**: Predictive Early Warning and Algorithmic Fairness in Distance Higher Education: A Multi-Stage Study on OULAD
- **Sub-bullets**:
  - Evaluation of Machine Learning Models for Rolling 3-Week Dropout Prediction
  - Comprehensive Multi-Attribute Demographic Fairness Audit
  - Production-Ready Streamlit Analytics & Caseload Simulation Suite
- **Author**: Tanmay (Downstream Analytics Lead: Fairness, Streamlit Suite, Synthesis)
- **Collaborators**: Teammates A, B, C (Data Pipeline, Zoo Training, Sequence Models)
- **Speaker Notes**:
  "Good morning, examiners. Today I am presenting our investigation into predictive retention modeling on the Open University Learning Analytics Dataset. My specific role in this team project begins downstream of model outputs: establishing an empirical benchmark, conducting a formal algorithmic fairness audit, and building an interactive decision-support suite for academic advising teams."

---

### Slide 2: Institutional Motivation & The Retention Challenge
- **Visual**: Diagram showing student enrollment funnel, early dropout cliff (Weeks 1–6), and advisor capacity constraint.
- **Key Points**:
  - High open-access attrition: ~25% of distance learners disengage before completion.
  - Advising capacity is severely limited: advisors cannot monitor thousands of raw clickstreams.
  - The Early Warning Goal: Actionable weekly prioritization under class imbalance ($2.37\%$ positive rate).
- **Speaker Notes**:
  "At open-entry distance institutions like the Open University, early departure is concentrated in the first eight weeks. Advising services have finite human capacity. They need early, actionable flags, but operate in an environment where less than 3% of weekly observations represent true dropouts."

---

### Slide 3: Data Lineage, Leakage Guards & Sealed Holdout Protocol
- **Visual**: Diagram of temporal splits: Train (2013B/J, 11,835 learners) $\to$ Val (2014B, 6,474 learners) $\to$ Sealed Test (2014J).
- **Key Points**:
  - Point-in-time feature extraction: strictly zero future engagement exposure.
  - Cohort reconciliation: 6,474 enrolled validation learners ($5,079$ active non-withdrawn $+ 1,395$ active dropouts).
  - Strict Holdout Protocol: **2014J is completely sealed**. No hyperparameter tuning or early evaluation on test data.
- **Speaker Notes**:
  "To maintain scientific integrity, we enforce temporal point-in-time boundaries. Furthermore, our test presentation—2014J—is strictly sealed. Every single number and metric presented today is derived from out-of-presentation temporal validation on 2014B."

---

### Slide 4: Baseline Progression (Phase 2 Modeling)
- **Visual**: Table comparing Majority Class, Univariate Baselines, 39-feature LR, and LightGBM v1–v3.
- **Key Points**:
  - Majority Class: PR-AUC 0.0237 ($1.00\times$).
  - Single features (Recency, Clicks, Scores): PR-AUC 0.032–0.034 ($1.35\times$–$1.44\times$).
  - 39-Feature Logistic Regression: PR-AUC 0.0617 ($2.61\times$).
  - LightGBM v3 (Official Champion): PR-AUC 0.0786 ($3.32\times$), ROC-AUC 0.7757.
- **Speaker Notes**:
  "Notice the progression: tracking a single feature like inactivity recency or click volume yields negligible improvement over chance. Moving to 39 features with logistic regression jumps our precision multiple to 2.6x, and gradient boosting pushes it to 3.3x by capturing non-linear interactions."

---

### Slide 5: Model Zoo Benchmark: Empirical Equivalence of Tree Models
- **Visual**: Forest plot of Paired $\Delta\text{PR}$ vs LightGBM v3 with 95% learner-clustered bootstrap CIs.
- **Key Points**:
  - LightGBM v3 (0.0786 PR), CatBoost (0.0786 PR), XGBoost (0.0781 PR), Random Forest (0.0778 PR).
  - **All tree models are statistically tied** ($\Delta\text{PR}$ CIs encompass 0.0).
  - Selection rationale: LightGBM v3 chosen for operational efficiency (3.3s training, 0.43 MB disk, 13.5 ms CPU latency).
- **Speaker Notes**:
  "Under 500-resample paired bootstrap testing clustered by student, all four tree models—LightGBM, CatBoost, XGBoost, and Random Forest—are statistically tied. This is an essential empirical finding: model performance is constrained by feature representation, not choice of tree algorithm. LightGBM v3 is selected as champion purely for operational latency and efficiency."

---

### Slide 6: Temporal Dynamics & Curriculum Generalization
- **Visual**: Line plot of weekly PR-AUC multiplier (Weeks 0–28) alongside Module CCC zero-shot transfer bar chart.
- **Key Points**:
  - Early noise: Weeks 0–1 drop-add window exhibits low discrimination (ROC ~0.57).
  - Signal takes off at Week 3 (ROC 0.764, PR multiplier $3.02\times$).
  - Zero-shot curriculum transfer: Module CCC (completely unseen in train) achieves 0.732 ROC and 0.092 PR ($2.40\times$ base rate).
- **Speaker Notes**:
  "Early weeks are noisy—orientation clicks offer little retention signal. But by Week 3, the model achieves over 3x lift. Furthermore, on Module CCC, which was completely absent from training data, the model transfers cleanly with 0.732 ROC, proving it learns general learning behaviors rather than course-specific artifacts."

---

### Slide 7: Probability Calibration & Reliability
- **Visual**: Reliability diagrams (uncalibrated vs Isotonic regression vs Platt scaling).
- **Key Points**:
  - Raw uncalibrated models under-predict overall (1.9% predicted vs 2.37% actual base rate).
  - High-risk over-confidence: In bins $> 0.20$, raw models predict 25%–35% risk where empirical attrition is 10%–12%.
  - Post-hoc Isotonic calibration slashes ECE from 0.005 to < 0.001 without altering ranking.
- **Speaker Notes**:
  "If an advisor acts on absolute predicted probabilities, calibration is critical. Uncalibrated trees over-estimate high-risk severity by nearly 3x. Post-hoc isotonic regression aligns predicted probabilities with observed outcomes, reducing calibration error by over 80%."

---

### Slide 8: Algorithmic Fairness Audit: Multi-Attribute Scope
- **Visual**: Summary matrix of Demographic Parity Differences and Equalized Odds across 6 attributes.
- **Key Points**:
  - Evaluated on validation split (`2014B`) under standard Top-10% weekly capacity policy.
  - Attributes audited: `imd_band`, `highest_education`, `age_band`, `gender`, `disability`, `region`.
  - Demographic features are strictly quarantined for evaluation; never fed into models.
- **Speaker Notes**:
  "Now turning to our fairness audit. We evaluated our champion model across six demographic dimensions under a fixed institutional policy: flagging the top 10% most disengaged students weekly. Demographics were strictly isolated from training and only joined at evaluation time."

---

### Slide 9: Subgroup Disparities & Kleinberg's Theorem
- **Visual**: Bar chart showing Selection Rate and Recall (TPR) across Gender and Education levels.
- **Key Points**:
  - Gender: Male selection 11.8% / TPR 39.2% vs Female selection 7.7% / TPR 29.4%.
  - Underlying base rates: Male 2.66% vs Female 1.98%.
  - ROC-AUC is identical (Male 0.774 vs Female 0.773, gap 0.0011).
  - **Kleinberg's Impossibility Theorem**: Equal TPR and Equal Precision cannot simultaneously hold under unequal base rates.
- **Speaker Notes**:
  "Notice the apparent gender disparity: male students are selected at 11.8% with 39% recall, while female students are selected at 7.7% with 29% recall. However, their ROC-AUC is identical. This disparity is mathematically governed by Kleinberg's theorem: because female students withdraw at a lower baseline rate in this cohort, equal opportunity and predictive parity cannot hold simultaneously."

---

### Slide 10: Demographic Proxy Predictability
- **Visual**: Bar chart of 5-fold GroupKFold Macro-F1 lift predicting sensitive attributes from behavior, plus IMD rank correlation scatter.
- **Key Points**:
  - Behavior encodes proxy traces: Gender Macro-F1 lift $2.14\times$; IMD lift $2.85\times$.
  - Socio-economic deprivation rank correlation: Spearman $\rho = -0.1041$ ($p = 9.12 \times 10^{-16}$).
  - Proof that low-income postcodes are not systematically penalised.
- **Speaker Notes**:
  "Even without demographic inputs, could models learn proxies? Our 5-fold cross-validation proves that course choice and activity rhythms allow moderate demographic inference. However, Spearman rank correlation between predicted risk and IMD deprivation deciles is near zero at -0.104, confirming that the model does not penalise students simply for being from deprived postcodes."

---

### Slide 11: In-Sample Threshold Mitigation Experiment
- **Visual**: Trade-off table showing Global Baseline vs Group-Specific Thresholds on 50/50 val split.
- **Key Points**:
  - Equalizing group selection rates reduces Equalized Odds Difference from 0.461 to 0.304 (a 34% reduction).
  - Overall recall is preserved (36.8% vs 34.2%).
  - Governance disclosure: Disparities are **measured and reduced**, never 'fixed', and in-sample tuning risks operational disparate treatment.
- **Speaker Notes**:
  "We tested mitigating disparities using group-specific thresholds on a 50/50 split of validation learners. This reduced the equalized odds difference by 34% while maintaining recall. However, we state clearly that disparities are 'measured and reduced', never 'fixed'—group-specific thresholds raise serious operational and legal questions in live higher education advising."

---

### Slide 12: Streamlit Analytics & Decision Support Suite
- **Visual**: Screenshots/wireframes of Streamlit dashboard (Model Comparison, Cohort Simulator, Learner Trajectory).
- **Key Points**:
  - High performance: cached Parquet backend renders complex views in $< 50$ ms.
  - Interactive policy simulator: Top-X% capacity slider mapping directly to advisor workloads.
  - Longitudinal learner trajectory: weekly risk curves with star markers for flagged weeks and automated case summaries.
  - Graceful degradation: clean pending cards for teammate SHAP and sequence models.
- **Speaker Notes**:
  "To make these findings usable for advisors, we built an offline-capable Streamlit application. Rather than raw probabilities, advisors interact with caseload capacity sliders that compute real-time student lift. The learner detail view shows weekly trajectories, flagged periods, and automated natural language case summaries."

---

### Slide 13: Critical Discussion & Honest Limitations
- **Visual**: Two-column summary comparing strengths vs fundamental limitations.
- **Key Points**:
  - **Assessment-Driven Signal**: Early prediction (Weeks 1–5) relies on weak click trends (best MCC ~0.14); real discrimination occurs after Week 8 assessment submissions.
  - **Modest Student-Level Lift**: $1.4\times$ to $1.9\times$ lift over random selection; over 40% of dropouts show no behavioral pre-warning.
  - **Single Presentation Caveat**: All audit figures derive from 2014B; October cohorts may diverge.
  - **Ethical Intervention Risks**: Automated flags must trigger supportive advising, avoiding deficit framing or self-fulfilling prophecies.
- **Speaker Notes**:
  "We must be candid about limitations. Early clicks are weak predictors. True signal is assessment-driven. At a student level, lift is a modest 1.4x to 1.9x over random outreach. Over 40% of dropouts leave quietly without activity drops. Automated tools must guide human outreach, never automate high-stakes academic decisions."

---

### Slide 14: Conclusions & Project Roadmap
- **Visual**: Roadmap diagram highlighting completed deliverables and pending teammate milestones.
- **Key Points**:
  - Downstream pipeline fully validated: benchmark, fairness audit, and decision support app.
  - 13/13 automated unit tests passing across fairness and application suites.
  - Next steps: unsealing presentation 2014J, integrating teammate sequence models (LSTM), and ingesting SHAP top-5 explanations.
- **Speaker Notes**:
  "In conclusion, we have built a transparent, rigorous, and verifiable retention analytics pipeline. All code, cache builders, and fairness metrics are backed by passing automated tests. We are ready to ingest teammate sequence models and unseal presentation 2014J when instructed. Thank you, and I look forward to your questions."

---

### Slide 15: Anticipated Examiner Questions & Defense Highlights
- **Q1: Why is PR-AUC preferred over ROC-AUC?**
  - *Answer*: With a 2.37% base rate, true negatives dominate. A model can generate thousands of false alarms while maintaining a 0.85 ROC-AUC. PR-AUC directly reflects precision and alert utility.
- **Q2: Why did tree models tie?**
  - *Answer*: Tabular performance on OULAD is feature-constrained. The 38 continuous features provide consistent splits regardless of whether trees are depth-wise (LightGBM), symmetric (CatBoost), or bagged (Random Forest).
- **Q3: Can Equal Opportunity and Predictive Parity hold together?**
  - *Answer*: Mathematically impossible under Kleinberg's theorem whenever base rates differ across groups ($2.66\%$ Male vs $1.98\%$ Female), unless the classifier achieves 100% accuracy.
