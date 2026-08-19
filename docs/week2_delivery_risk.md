# Olist Brazilian E-Commerce Analytics — Week 2: Delivery Delay Risk Model

## Overview

Week 1's clustering surfaced an unresolved puzzle: a **Dissatisfied Segment** (15.5% of customers) that looked statistically identical to the largest, happiest customer segment on every feature used in clustering, except review score. Week 2 builds a leakage-safe classification model predicting delivery delay risk at the moment of purchase, both as a standalone operational tool and as groundwork for testing, causally, whether delivery lateness explains that unresolved gap (Week 3).

**The question:** can delivery delay risk be predicted at checkout, using only information available at that moment, well enough to be operationally useful?

---

## Target variable and feature set

**Target:** `is_late` — `order_delivered_customer_date > order_estimated_delivery_date`, computed only for delivered orders (non-delivered orders represent a different failure mode and are excluded). Late rate: **8.11%** (7,826 of 96,478 delivered orders) — imbalanced, but workable.

**Features**, all knowable at checkout:
- Seller-to-customer distance (Haversine formula on averaged zip-prefix geolocation)
- Product weight, volume, category
- Order timing (day of week, month)
- Freight value, item count, payment installments
- **Seller historical on-time rate** — the most important feature, computed strictly from data prior to the prediction point to avoid leakage, with a fallback to the population average for sellers with fewer than 5 prior orders (plus a binary flag marking those thin-history sellers)

Extensive data validation preceded modeling: confirmed no row duplication from merges, investigated and confirmed free-shipping orders were a legitimate pattern (not a data error) with a lower-than-average late rate, confirmed category and state both showed real, meaningful variation in late rate (validating those features), and confirmed zero category mismatch between train and test periods.

---

## The central diagnostic finding

An initial 80/20 chronological train/test split produced a striking, suspicious result: **train AUC of 0.71, test AUC of 0.49 (chance level)**. Rather than accept or dismiss this, a full investigation was carried out:

1. Verified individual features had real, statistically significant correlations with the target — ruling out "the features are simply useless."
2. Verified the model's coefficients were directionally sensible — ruling out a broken training process.
3. Found the strongest feature's correlation with the target collapsed between train and test (−0.21 → −0.02).
4. Found dramatic late-rate spikes in November 2017, February 2018, and March 2018 — and **confirmed via web search** that these align with two independently documented real-world events: the 2018 Brazilian truck drivers' strike and a Black Friday demand surge.
5. Ran 5-fold rolling time-series cross-validation: 4 of 5 folds showed real, workable AUC (0.57–0.61); only the fold matching the original test window looked broken — proving the single split, not the model, was the problem.
6. Investigated that window further and found a **dataset export cutoff artifact** — order volume tapered unnaturally to near-zero in the final week of data collection. Trimmed (677 rows, 0.7%).
7. **Trimming the artifact did not fix the problem** — an important negative result that prevented a false conclusion and forced deeper investigation.
8. Split the poor-performing window by seller history status: even sellers with sufficient history predicted no better than chance, ruling out "it's just new sellers."
9. Compared every feature's correlation with the target across a "good" window and the "bad" window: **seller-customer distance and freight both flipped sign** — a coherent multi-feature reversal.
10. Actively tested for a bug rather than accepting this at face value: checked for duplicate rows, confirmed distance distributions hadn't shifted, ran a 20-sample bootstrap confirming the flipped correlation was stable (not noise), and restricted the comparison to only sellers present in *both* windows — the flip persisted, ruling out a change in seller composition.
11. Broke delivery time into its three component stages (purchase→approval, approval→carrier, carrier→customer) and found **actual transit time's relationship with distance stayed stable** across both windows (~0.52–0.54) — the physical reality of delivery didn't change.
12. Found the real mechanism: the relationship between distance and the **delivery-date estimate buffer** shifted between windows — long-distance orders were given comparatively more generous buffers relative to short-distance orders in the later period than earlier. `is_late` depends on the gap between actual and *promised* delivery, so this changed what "late" meant relative to distance, without any change in real delivery speed.

---

## Final evaluation: walk-forward validation

Given the demonstrated instability of a single fixed split, walk-forward (rolling-origin) validation was adopted as the primary evaluation method across 13 months. Result: **mean AUC 0.578 (std 0.054)**, with the one exception (final month) fully explained by the mechanism above.

A Random Forest model, evaluated under the same framework, performed comparably (mean AUC 0.576) and showed the same weakness in the same month — reinforcing that the limitation is rooted in the target's construction, not model capacity. Logistic regression was retained as the primary model for its interpretability.

A follow-up attempt to add seller-level stage-specific timing features (approval speed, transit speed, separate from the blended on-time rate) produced **no improvement** (mean AUC 0.575) — an honest negative result, reported rather than hidden, that further confirmed the issue is the target's stability over time, not insufficient feature granularity.

### External validation
A search for related work found a recent academic paper studying Olist delay prediction under leakage-controlled rolling-origin evaluation, reporting an R² of approximately −0.02 for delay severity — essentially unlearnable beyond a naive baseline. A public practitioner project doing the same classification task reported comparably modest results (53% recall, 19% precision). This dataset's difficulty is a documented, shared characteristic, not a sign of an error in this pipeline.

---

## Business usability: threshold and precision analysis

AUC measures ranking quality, not whether a model is deployable. Precision and recall were computed across a range of thresholds:

| Threshold | Precision | Recall | Orders Flagged |
|---|---|---|---|
| 0.3 | 5.2% | 87.6% | 16,600 |
| 0.5 | 4.4% | 36.7% | 8,249 |
| 0.8 | 6.3% | 5.9% | 939 |

Precision stays in the single digits at every threshold. **This model, as built, is not precise enough to support individual-order human-reviewed triage** — the false-positive volume would overwhelm any team acting on every flag. It is realistically better suited as an **aggregate, seller-level or regional risk-ranking signal** rather than a per-order decision tool. This limitation is reported directly, not smoothed over.

---

## What this project is not

This is a static historical dataset, not a live system. Real production delay-prediction systems (e.g., at Instacart, DoorDash) use live courier capacity, real routing data, and continuously retrained models; this project demonstrates leakage-safe methodology and honest evaluation, not production infrastructure.

## Tools used

`pandas`, `numpy`, `scikit-learn` (LogisticRegression, RandomForestClassifier, StandardScaler, precision_recall_curve), `scipy` (point-biserial correlation, bootstrap resampling), Haversine distance calculations, walk-forward time-series validation, web search for external validation.

## Related documents
See `Week2_Interview_Prep.md` (repo root) for a full reasoning log and 28 anticipated interview questions with answers, covering this analysis in depth.

## Next steps
Week 3 uses `is_late` directly to test, causally, whether delivery lateness explains Week 1's Dissatisfied Segment finding.
