# does-late-delivery-cause-dissatisfaction
End-to-end analytics project on Brazilian e-commerce data. K-Means clustering surfaces a Dissatisfied Segment not explained by behavior alone. A leakage-safe delivery delay model investigates further, and propensity score matching provides causal evidence that lateness, not confounders, drives the dissatisfaction. Includes Tableau dashboards.


# Olist Brazilian E-Commerce Analytics

**Does late delivery actually cause customer dissatisfaction, or is it just correlated with it?**

An end-to-end analytics project on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle) — ~99,000 orders across 9 relational tables — that starts with an unsupervised clustering mystery, builds a predictive model to investigate it, and resolves it with a rigorous causal inference analysis.

**Live dashboard:** [Tableau Public link — add once published]

---

## The story, in short

K-Means clustering on customer behavior surfaced five segments, two of which looked nearly identical: same spending, same recency, same category diversity, same freight cost. The only difference was satisfaction — one segment averaged 4.75 stars, the other 1.79. Nothing in the clustering features explained the gap.

That unresolved question drove the rest of the project. A leakage-safe classification model was built to predict delivery delay risk at the moment of purchase, both as a standalone operational tool and to establish a clean, well-understood delay variable. That variable was then used in a propensity score matching analysis to test, causally, whether delivery lateness explains the satisfaction gap.

**It does, substantially.** The naive difference in review score between late and on-time orders was 1.735 stars. After matching on distance, price, freight, seller reliability, category, and payment behavior, the gap barely narrowed, to 1.671 stars, and held up across every time window tested. This is strong evidence that lateness has a real, largely direct effect on satisfaction, not just a coincidental association with other factors.

---

## Why this project is built the way it is

Most portfolio projects stop at prediction. This one is built around a different idea: test whether a standard analytical framework actually fits the data before applying it, investigate surprising results instead of accepting or dismissing them, and be honest about what didn't work. Each phase builds on the last, so the project reads as one continuous investigation rather than four disconnected exercises.

This structure is meant to speak to data analyst, AI data analyst, and data scientist roles where causal reasoning about interventions matters as much as prediction: fintech (risk scoring, explainability), healthtech (utilization and intervention analysis), and consumer marketplaces or logistics platforms (retention, delivery reliability).

---

## Key findings

**Customer segmentation.** Standard RFM segmentation assumes meaningful repeat-purchase behavior; this dataset's repeat rate is only 3.12%, which required adapting the framework (Frequency as a binary flag, not a scored quintile) rather than applying it blindly. K-Means clustering on purely behavioral features (k=5) surfaced the Dissatisfied Segment finding described above. Two rejected clustering iterations, one distorted by payment-type dominance, one by a data quality bug, are documented rather than hidden.

**Delivery risk model.** A logistic regression model predicting delivery delay at checkout showed a suspicious train/test performance gap (0.71 vs. 0.49 AUC) on an initial split. Rather than accept or dismiss this, a full diagnostic investigation traced it to a real, externally-corroborated cause: a documented 2018 Brazilian truck drivers' strike combined with a shift in how delivery-date estimates scaled with distance over time, not a bug, not noise, and not simply "new sellers." Walk-forward validation across 13 months produced a mean AUC of 0.578, consistent with independently published research on this same dataset. A precision/recall analysis showed the model isn't precise enough (4-6%) for individual-order triage, but is realistically usable as an aggregate, seller-level risk-ranking signal, a limitation reported directly rather than smoothed over.

**Causal inference.** Propensity score matching, validated with a covariate balance check (all confounders within 6% between matched groups) and a caliper-based sensitivity check, found delivery lateness has a robust, largely direct causal effect on review score. The effect held across four separate time windows spanning the full dataset, a sharp contrast to the delivery model's core predictive relationship, which broke down in the same later period. Rigor was applied evenly to both analyses; in this case, it confirmed the finding rather than undermining it.

**Dashboards.** Two Tableau dashboards translate these findings into business-facing visuals. Building them surfaced two more small-sample-size artifacts (a product category and a state each showing misleadingly extreme late rates based on very few orders), caught and corrected with the same statistical discipline applied throughout the analytical work.

---

## Documentation

Full week-by-week write-ups, including rejected approaches, diagnostic steps, and all supporting analysis, are in [`/docs`](./docs):

- [Week 1: Segmentation & Clustering](./docs/week1_segmentation.md)
- [Week 2: Delivery Delay Risk Model](./docs/week2_delivery_risk.md)
- [Week 3: Causal Inference](./docs/week3_causal_inference.md)
- [Week 4: Tableau Dashboards](./docs/week4_dashboards.md)

## Tools used

`pandas`, `numpy`, `scikit-learn` (KMeans, LogisticRegression, RandomForestClassifier, NearestNeighbors, StandardScaler), `scipy` (point-biserial correlation, bootstrap resampling), Haversine distance calculations, walk-forward time-series validation, Tableau.

## What this project is not

This is a static historical dataset, not a live production system. Feature engineering and evaluation follow production-grade discipline (leakage control, time-aware validation), but there is no real-time infrastructure, live courier data, or continuous retraining here.
