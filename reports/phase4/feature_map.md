# Phase 4 Feature Map & Sequence Construction (`reports/phase4/feature_map.md`)

## 1. Feature Grain & Architecture Overview
- **Raw Grid Grain:** Every learner is indexed on a contiguous weekly grid `(code_module, code_presentation, id_student, week)` starting at week 0 up to presentation end (weeks 30-38).
- **Official 38 Features:** Derived from `config/features.yaml` (39 sanctioned features minus `n_skipped_resources` due to module mix drift).
- **Time-Varying vs Static Partition:**
  - **Static Features (4 features):** Learner-level demographic/prior variables that do not evolve during the presentation (`num_of_prev_attempts`, `studied_credits`, `registration_lead_time`, `precourse_clicks`). In sequence tensors, they are repeated across active steps (and zeroed at padded steps) so sequence models have full information access, and are also isolated into static tensor $S$ ($F_{static} = 4$) for the B4 hybrid branch.
  - **Time-Varying Features (34 features):** Trajectory variables computed strictly point-in-time up to the end of week $t$ (`cutoff_day = 7*(t+1) - 1`). Includes 1 progress metric (`course_progress`), 12 behavioral click/entropy metrics, 9 temporal slope/rolling metrics, 9 assessment submission/score metrics, and 3 normalized curriculum pacing/gap metrics.
- **Window Lengths ($W \in \{4, 8\}$):**
  - For prediction at week $t$: sequence covers weeks $t - W + 1 \dots t$ inclusive.
  - No information after week $t$ is ever observed.
  - Left-padding with zeros for weeks before week 0; mask $M[i, s] = \text{False}$ for padded steps, $\text{True}$ for real steps.

---

## 2. Exhaustive 38-Feature Table

| Index | Feature Name | Group | Type | Weekly Construction | Point-in-Time Safety Note |
| :---: | :--- | :--- | :---: | :--- | :--- |
| 1 | `num_of_prev_attempts` | static | Static | Constant across all weeks from student registration profile. | Known prior to presentation start. |
| 2 | `studied_credits` | static | Static | Constant across all weeks from course enrollment. | Known prior to presentation start. |
| 3 | `registration_lead_time` | static | Static | Days between registration date and course Day 0. | Negative or 0; known at registration. |
| 4 | `precourse_clicks` | static | Static | Sum of VLE clicks logged strictly before Day 0 (`date < 0`). | Frozen at Day 0. |
| 5 | `course_progress` | static | Time-Varying | Ratio of current week $t$ to presentation length. Monotonically increases. | Dependent solely on calendar week $t$. |
| 6 | `weekly_clicks` | behavior | Time-Varying | Sum of clicks logged in week $t$ (`7*t <= date <= 7*(t+1)-1`). | Strict weekly boundary. |
| 7 | `days_since_last_activity` | behavior | Time-Varying | Days from cutoff day $7*(t+1)-1$ back to latest recorded click date. | Recency computed at week $t$ cutoff. |
| 8 | `weekend_click_ratio` | behavior | Time-Varying | Fraction of week $t$ clicks occurring on Saturday/Sunday. | Strict weekly boundary. |
| 9 | `active_days_entropy` | behavior | Time-Varying | Shannon entropy of click distribution across the 7 days of week $t$. | Strict weekly boundary. |
| 10 | `share_forumng` | behavior | Time-Varying | Proportion of week $t$ clicks directed to discussion forums. | Strict weekly boundary. |
| 11 | `share_quiz` | behavior | Time-Varying | Proportion of week $t$ clicks directed to practice quizzes. | Strict weekly boundary. |
| 12 | `share_resource` | behavior | Time-Varying | Proportion of week $t$ clicks directed to static document resources. | Strict weekly boundary. |
| 13 | `share_oucontent` | behavior | Time-Varying | Proportion of week $t$ clicks directed to core online textbook materials. | Strict weekly boundary. |
| 14 | `share_homepage` | behavior | Time-Varying | Proportion of week $t$ clicks directed to course home landing page. | Strict weekly boundary. |
| 15 | `share_subpage` | behavior | Time-Varying | Proportion of week $t$ clicks directed to submodule index subpages. | Strict weekly boundary. |
| 16 | `share_url` | behavior | Time-Varying | Proportion of week $t$ clicks directed to external hyperlinks. | Strict weekly boundary. |
| 17 | `share_other_types` | behavior | Time-Varying | Proportion of week $t$ clicks directed to remaining activity types. | Strict weekly boundary. |
| 18 | `wow_change_clicks` | temporal | Time-Varying | Week-over-week difference in clicks: $clicks(t) - clicks(t-1)$. | Computed using current and prior week. |
| 19 | `wow_change_active_days` | temporal | Time-Varying | Week-over-week difference in active days: $days(t) - days(t-1)$. | Computed using current and prior week. |
| 20 | `rolling_mean_clicks_2w` | temporal | Time-Varying | Rolling average of clicks over weeks $t-1 \dots t$. | Point-in-time rolling window. |
| 21 | `rolling_mean_clicks_4w` | temporal | Time-Varying | Rolling average of clicks over weeks $t-3 \dots t$. | Point-in-time rolling window. |
| 22 | `rolling_mean_active_days_2w`| temporal | Time-Varying | Rolling average of active days over weeks $t-1 \dots t$. | Point-in-time rolling window. |
| 23 | `rolling_mean_active_days_4w`| temporal | Time-Varying | Rolling average of active days over weeks $t-3 \dots t$. | Point-in-time rolling window. |
| 24 | `clicks_slope_4w` | temporal | Time-Varying | Linear regression slope of clicks over past 4 weeks ($t-3 \dots t$). | Point-in-time rolling window. |
| 25 | `clicks_ewma` | temporal | Time-Varying | Exponentially weighted moving average of clicks ($\alpha=0.3$). | Causal filter, no future data. |
| 26 | `engagement_decay_rate` | temporal | Time-Varying | Ratio of 2-week rolling clicks to 4-week rolling clicks. | Causal ratio, zero-division safe. |
| 27 | `n_submitted` | assessment | Time-Varying | Cumulative assessments submitted with submission date $\le$ cutoff day. | Truncated strictly at cutoff day. |
| 28 | `n_missed` | assessment | Time-Varying | Cumulative assessments with due date $\le$ cutoff day and no submission. | Truncated strictly at cutoff day. |
| 29 | `mean_score_submitted` | assessment | Time-Varying | Mean score across assessments submitted on or before cutoff day. | Truncated strictly at cutoff day. |
| 30 | `last_score` | assessment | Time-Varying | Score on most recently submitted assessment on or before cutoff day. | Truncated strictly at cutoff day. |
| 31 | `score_slope` | assessment | Time-Varying | Trajectory slope across sequentially submitted assessment scores. | Truncated strictly at cutoff day. |
| 32 | `mean_lateness` | assessment | Time-Varying | Mean days between submission date and due date for past assessments. | Truncated strictly at cutoff day. |
| 33 | `weighted_score_to_date` | assessment | Time-Varying | Module-weight normalized score sum earned up to cutoff day. | Truncated strictly at cutoff day. |
| 34 | `n_banked` | assessment | Time-Varying | Count of previously transferred/banked assessment credits. | Evaluated at week $t$. |
| 35 | `banked_score_mean` | assessment | Time-Varying | Mean score across banked assessment credits. | Evaluated at week $t$. |
| 36 | `pct_resources_visited` | curriculum | Time-Varying | Cumulative unique VLE resource IDs clicked up to cutoff day / module catalog size. | Truncated strictly at cutoff day. |
| 37 | `pacing_vs_cohort_median` | curriculum | Time-Varying | Cumulative clicks of student up to week $t$ divided by cohort median at week $t$. | Computed against point-in-time cohort. |
| 38 | `prerequisite_gap` | curriculum | Time-Varying | Difference between student's initial engagement and expected curriculum pacing. | Truncated strictly at cutoff day. |

*(Note: `n_skipped_resources` is excluded from the 38-feature set per the Phase 2 baseline specification due to cross-presentation catalog size drift).*
