# 5-Minute Live Demonstration Script: Streamlit Analytics & Decision Support Suite

**Presenter**: Tanmay (Downstream Analytics Lead)  
**Total Target Time**: Exactly 5 Minutes  
**Terminal Launch Command**:  
```bash
cd ML_Project
.venv\Scripts\streamlit run app/main.py
```

---

### Minute 0:00 – 0:45 | Introduction & Sealed Holdout Protocol Banner
- **Action**:
  - Open browser window at `http://localhost:8501`.
  - Maximize the browser window to show the clean, widescreen layout.
  - Direct the examiner's attention to the persistent yellow alert banner at the top of the application.
- **Spoken Narration**:
  > *"Good morning, examiners. Welcome to our interactive Decision Support Suite for student retention and algorithmic fairness. Before exploring the pages, I want to highlight the protocol banner at the top: our final test split, presentation 2014J, is strictly sealed. Every metric, curve, and cohort simulation you see today runs entirely on held-out temporal validation data from 2014B across 6,474 enrolled students."*
- **Action**:
  - Point to the sidebar metadata: show that the app dynamically discovered 62 prediction files, targets, and horizons from `predictions/` without hardcoding.

---

### Minute 0:45 – 1:45 | Page 1: Model Comparison Leaderboard & Statistical Ties
- **Action**:
  - Navigate to **"1. Overview & Model Comparison"** in the sidebar.
  - Scroll through the formal validation leaderboard table.
  - Hover over the Plotly Precision-Recall (PR) and ROC curves.
  - Point out the green badge: **"Statistically Tied (*)"**.
- **Spoken Narration**:
  > *"This centerpiece leaderboard summarizes our model zoo. While logistic regression achieves a 2.6x lift over base rate, our gradient boosted trees achieve 3.32x lift with a validation PR-AUC of 0.0786 and ROC-AUC of 0.7757.*
  >
  > *Notice this crucial green indicator: under 500-iteration learner-clustered paired bootstrap testing, CatBoost, XGBoost, and Random Forest are all statistically tied with LightGBM v3. The 95% confidence intervals for their paired deltas encompass zero. We selected LightGBM v3 purely for operational reasons: as shown in our cost table, it trains in 3.3 seconds and executes batch inference in just 13.5 milliseconds per 1,000 rows on a standard CPU."*

---

### Minute 1:45 – 2:45 | Page 2: Cohort Caseload Simulation & Exact Denominators
- **Action**:
  - Click **"2. Cohort Risk & Caseload Policy"** in the sidebar.
  - Point to the four metric cards at the top displaying the verified ground-truth denominators:
    - **6,474 Total Enrolled Learners**
    - **1,395 Unique Eventual Dropouts**
  - Interact with the **Weekly Caseload Capacity Slider**: drag it from 10% down to 5%, then up to 15%.
  - Show the live updating precision, recall, and student catch rate.
- **Spoken Narration**:
  > *"In a real university advising center, counselors do not think in terms of abstract probability thresholds; they have fixed weekly capacity. Page 2 translates machine learning into operational reality.*
  >
  > *Notice our exact verified population denominators: 6,474 enrolled learners and 1,395 eventual dropouts. As I adjust this weekly capacity slider, the app recalculates the live precision-recall trade-off. At our default 10% capacity cut, the system catches 55.7% of all students who will eventually withdraw, delivering a 1.59x lift over random selection at the human student level."*

---

### Minute 2:45 – 3:45 | Page 3: Learner Trajectory, Early Warning & Graceful Degradation
- **Action**:
  - Click **"3. Learner Trajectory & Early Warning"** in the sidebar.
  - Filter by **Module BBB**, and select an **Eventual Dropout** student (e.g. `id_student = 27891` or similar).
  - Hover over the weekly trajectory line plot showing predicted risk spiking over time.
  - Point out the pink star markers: **"Flagged (Top-10% Caseload)"**.
  - Read the **Automated Case Synthesis** box.
  - Scroll down to the SHAP and forecasting boxes. Point out the clean **"Pending Teammate Submission"** card.
- **Spoken Narration**:
  > *"Now let's zoom in on an individual student's educational journey. For student 27891 in Module BBB, who ultimately withdrew, you can see their predicted risk trajectory steadily climbing above the threshold in weeks 3, 4, and 5. The pink star markers highlight the exact weeks when an advisor alert was triggered.*
  >
  > *Below the chart, our system generates an automated clinical synthesis summarizing the case. Furthermore, notice our software engineering discipline: teammate SHAP explanations and deep sequence models are under active development. Rather than crashing or using mocked data, the application displays a clean, transparent pending card conforming to our documented team contract."*

---

### Minute 3:45 – 4:30 | Page 4: Algorithmic Fairness Audit & Kleinberg's Theorem
- **Action**:
  - Click **"4. Algorithmic Fairness Audit"** in the sidebar.
  - View the headline disparity table across all 6 demographic attributes.
  - Select **"gender"** in the dropdown.
  - Point to the Male vs Female selection rates (11.8% vs 7.7%) and recall rates (39.2% vs 29.4%).
  - Point out that ROC-AUC is identical ($0.774$ vs $0.773$).
  - Scroll down to the **Demographic Proxy Analysis** and **Spearman Rank Correlation** with IMD deciles ($\rho = -0.1041$).
- **Spoken Narration**:
  > *"Page 4 presents our algorithmic fairness audit. Evaluating our champion model across 6 demographic attributes, we observe an apparent disparity in gender: male recall is 39% while female recall is 29%.*
  >
  > *However, look at the ROC-AUC: it is identical at 0.774 for men and 0.773 for women. This is Kleinberg's theorem in action: because female students in this cohort withdraw at a lower baseline rate—1.98% versus 2.66%—equal recall and equal precision cannot both hold.*
  >
  > *Crucially, our proxy analysis shows that the rank correlation between predicted dropout risk and socio-economic deprivation is essentially zero at -0.104, confirming that the model does not penalise students from low-income postcodes."*

---

### Minute 4:30 – 5:00 | Page 5 & 6: Calibration & Concluding Defense
- **Action**:
  - Click **"5. Calibration & Policy Reliability"**. Point out the reliability curve showing uncalibrated over-confidence vs Isotonic calibration.
  - Click **"6. Methodology, Contracts & Governance"**. Point out the documented team contracts and explicit limitations list.
- **Spoken Narration**:
  > *"Finally, on Page 5, our reliability curves reveal that raw tree models overestimate risk above 20%, but Isotonic calibration reduces calibration error by 83%. Page 6 provides full governance documentation, team contracts, and our honest limitations list.*
  >
  > *In summary, our downstream pipeline provides a verifiable benchmark, an exhaustive fairness audit, and an offline decision-support tool backed by 13 passing unit tests. Thank you, and I look forward to your questions."*
