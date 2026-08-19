# Olist Brazilian E-Commerce Analytics — Week 4: Tableau Dashboards

## Overview

Week 4 translates the findings from Weeks 1-3 into two business-facing Tableau dashboards, built from purpose-exported CSVs rather than raw joins. The goal was not to visualize everything, but to make the project's two strongest findings, the Dissatisfied Segment mystery and its causal resolution, immediately legible to a non-technical viewer.

**Live dashboard:** [Tableau Public link — add once published]

---

## Data exports

Four CSVs were exported from the project's Python analysis for use in Tableau:

| File | Grain | Rows | Purpose |
|---|---|---|---|
| `customer_segments.csv` | Customer | 92,754 | Dashboard 1 |
| `order_delivery_risk.csv` | Order | 94,670 | Dashboard 2 |
| `seller_performance.csv` | Seller | 2,932 | Reference/future use |
| `regional_summary.csv` | State | 27 | Reference/future use |

All exports were validated for missing values and joined correctly against the confirmed final Week 1 cluster assignments before use.

---

## Dashboard 1: Customer Segments & Value

Three worksheets:

1. **Segment Sizes** — customer count by segment, sorted descending, with the Dissatisfied Segment highlighted. Establishes that this segment (15.5% of customers) is large enough to matter, not a statistical footnote.
2. **Segment Profiles** — a per-metric comparison grid (recency, monetary value, category diversity, installments, review score) across all five segments, each on its own scale. Visually shows that Dissatisfied Segment and Satisfied Core are nearly identical on every metric except review score.
3. **Dissatisfied vs. Satisfied Spotlight** — a focused two-bar comparison isolating just the review score gap between these two segments (1.79 vs. 4.75), the clearest possible statement of the Week 1 finding.

---

## Dashboard 2: Delivery Risk & Causal Impact

Three worksheets:

1. **Late Rate by Category** (filtered to categories with 300+ orders, after finding and excluding small-sample categories that produced misleading spikes, e.g., a 19% rate based on only 21 orders). Audio, home comfort, and food show the highest legitimate late rates (~10-13%).
2. **Late Rate by State** (filtered to states with 1,000+ orders, for the same reason). Confirms the Week 2 Python finding: Ceará, Bahia, and Rio de Janeiro show meaningfully elevated late rates (12-15%) compared to Paraná and Minas Gerais (5-6%).
3. **Naive vs. Matched Causal Effect** — the dashboard's centerpiece. A grouped bar chart built from the Week 3 propensity score matching results, showing that the review-score gap between late and on-time orders barely narrows after controlling for distance, price, freight, seller reliability, category, and payment behavior (naive: 4.293 vs. 2.558; matched: 4.229 vs. 2.558). This is the visual proof that delivery lateness has a real, largely direct causal effect on satisfaction.

---

## A data-quality discipline worth noting

Two small-sample artifacts were caught and corrected during dashboard construction rather than shipped as misleading charts: a category showing an inflated ~19% late rate based on just 21 orders, and a state showing ~23% based on 393 orders. Both were identified by checking order counts behind extreme values, then addressed with minimum-sample-size filters, consistent with the same statistical discipline applied throughout Weeks 1-3 (e.g., the 5-order minimum threshold used for the seller on-time rate feature in Week 2).

---

## Scope note

A third, optional dashboard covering regional/GTM-oriented views (using the already-built `seller_performance.csv` and `regional_summary.csv`) was considered but deliberately not built. Dashboards 1 and 2 already carry the project's core narrative; a third dashboard would have added breadth without adding to the central finding.

## Tools used

Tableau Public/Desktop, connected to CSVs exported from the project's Python analysis.

## Project status

All four planned weeks are complete: EDA and RFM-adapted clustering (Week 1), a leakage-safe delivery delay risk model with a rigorous diagnostic investigation (Week 2), causal inference on delivery lateness and satisfaction (Week 3), and this dashboard (Week 4).
