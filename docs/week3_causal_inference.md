# Olist Brazilian E-Commerce Analytics — Week 3: Causal Inference on Delivery Punctuality

## Overview

Week 1's clustering surfaced an unresolved puzzle: a Dissatisfied Segment (15.5% of customers) statistically identical to the largest, happiest segment on every clustering feature except review score. Week 2 built a leakage-safe model predicting delivery delay risk, establishing `is_late` as a rigorously defined variable. Week 3 uses that variable to test the natural follow-up question directly: **does delivery lateness causally affect customer satisfaction, or is the association just a coincidence of other factors that happen to correlate with both?**

---

## The question and why correlation isn't enough

A simple groupby shows late orders average far lower review scores than on-time orders. But that comparison alone can't distinguish two explanations: lateness itself upsets customers (a direct causal effect), or late orders and low-review orders share some other cause (cheaper products, less reliable sellers, farther shipping distances) that isn't lateness itself. Only the first explanation supports the claim that motivated this analysis.

---

## Method: Propensity Score Matching

**Confounders controlled for:** seller-customer distance, order value and shipping cost, seller reliability, payment installments, purchase timing, and product category.

*Note: `seller_on_time_rate` was computed strictly from training-period data in Week 2 to avoid leakage in a predictive model. Here, the goal is descriptive matching, not prediction of an unknown future outcome, so it's computed from the full dataset, a deliberate, documented difference in how the same feature is used correctly in two different contexts.*

**Approach:**
1. A logistic regression propensity model predicted `P(is_late)` using only the confounders.
2. Common support was checked before trusting any matching: 99.2% of late orders fell within the on-time group's propensity score range.
3. Each late order was matched to its nearest-neighbor on-time order by propensity score (1:1, with replacement). Average match distance was negligible (0.0002).
4. Review scores were compared within matched pairs.

---

## Results

| Comparison | Difference (late minus on-time) |
|---|---|
| Naive (no adjustment) | **−1.735** stars |
| Matched (propensity score, all pairs) | **−1.671** stars |
| Matched (caliper, best-quality pairs only) | **−1.695** stars |

The gap barely narrows after matching. If confounders were driving most of the naive association, the matched estimate would have shrunk substantially; instead, all three estimates cluster tightly together.

**Validation performed:**
- **Covariate balance check:** every confounder's mean value differed by less than 6% between the late and matched on-time groups after matching, well under the standard 10-25% threshold used to judge whether matches are genuinely comparable.
- **Caliper sensitivity check:** restricting to only the highest-quality matches (99.2% already met this stricter threshold) produced a nearly identical estimate.

## Temporal stability check

Following the same rigor applied in Week 2, the causal estimate was re-tested across four time windows (quartiles of the dataset's span, ~23,700 orders each):

| Window | Date Range | Naive Difference | Matched Difference |
|---|---|---|---|
| Q1 | Oct 2016 to Sep 2017 | −1.622 | −1.611 |
| Q2 | Sep 2017 to Jan 2018 | −1.821 | −1.699 |
| Q3 | Jan 2018 to May 2018 | −1.983 | −1.855 |
| Q4 | May 2018 to Aug 2018 | −1.102 | −1.099 |

The effect is directionally stable and substantial in every window, no sign reversals, no window where the effect disappears. Q4's smaller effect size is a plausible echo of the Week 2 finding that delivery-date estimate buffers became more generous for some orders in this later period; if "late" represented a smaller, less jarring delay on average, a softer satisfaction impact would be consistent with that.

This is a meaningfully stronger result than Week 2's predictive model, whose core relationship (distance and lateness) broke down and reversed sign in the same later period. The same scrutiny applied to a different question produced a stable, robust finding here.

## Conclusion

Delivery lateness has a large (~1.7 star), robust effect on review score that survives controlling for distance, price, freight, seller reliability, payment behavior, category, and purchase timing, and that holds up across the full span of the dataset. This is strong evidence, not proof, that Week 1's Dissatisfied Segment finding is substantially explained by delivery lateness specifically.

## Limitations, stated explicitly

Propensity score matching only controls for **observed** confounders. It cannot account for product quality or defects on arrival, individual customer expectations, whether the order was time-sensitive (a gift with a deadline), or post-purchase customer service interactions. This analysis rests on the standard ignorability (unconfoundedness) assumption, that once observed confounders are controlled for, no other systematic difference remains. This cannot be directly verified with observational data. Given the depth of the confounder set and the strength of the balance and sensitivity checks, the result is well-supported, but it is not the same standard of evidence a randomized experiment would provide.

## Scope note

NLP/sentiment analysis on review text was originally planned as a second component of Week 3, intended to corroborate the causal finding using an independent signal. It was deliberately deprioritized: the causal inference work above already provides rigorous, well-validated evidence on its own, and NLP would have added corroboration rather than a new independent finding.

## Tools used

`scikit-learn` (LogisticRegression, NearestNeighbors), `pandas`, confounder balance and caliper-based sensitivity analysis.

## Next steps

Week 4 translates this finding, along with Weeks 1-2, into business-facing Tableau dashboards.
