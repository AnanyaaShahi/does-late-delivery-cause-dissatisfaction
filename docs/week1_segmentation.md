# Olist Brazilian E-Commerce Analytics — Week 1: EDA, RFM Segmentation & Customer Clustering

## Overview

This is Week 1 of a 4-week analytics project built on the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle), covering ~99,441 orders across 9 relational tables (customers, orders, order items, payments, reviews, products, sellers, geolocation, category translations).

This week covers three stages:
1. Data validation and exploratory analysis
2. RFM (Recency, Frequency, Monetary) segmentation, adapted to fit this dataset's actual shape rather than applying the textbook framework blindly
3. K-Means clustering on an enriched behavioral feature set, extended beyond bare RFM to demonstrate unsupervised learning earlier in the portfolio

---

## 1. Data Validation

All 9 CSVs were loaded and validated against publicly documented row counts:

| Table | Rows |
|---|---|
| customers | 99,441 |
| orders | 99,441 |
| order_items | 112,650 |
| payments | 103,886 |
| reviews | 99,224 |
| products | 32,951 |
| sellers | 3,095 |
| geolocation | 1,000,163 |
| category_translation | 71 |

**Join integrity:** zero duplicate primary keys and zero orphaned foreign keys were found across the orders/customers/order_items/payments/reviews relationships (checked payments to orders, order_items to orders, and reviews to orders directions).

**Missingness findings:**
- `order_delivered_customer_date` is missing for 2,965 orders, matching the count of non-"delivered" order statuses almost exactly (cancelled, still in transit, unavailable, etc.), confirming this is expected missingness rather than a data quality issue.
- `review_comment_message` is missing for 58,247 of 99,224 reviews (~59%). The usable text sample is closer to ~41,000 rows. `review_score` is complete for all rows, so satisfaction can be measured dataset-wide even without comment text.

---

## 2. RFM Segmentation, Adapted for This Dataset

### The `customer_id` vs. `customer_unique_id` distinction

Olist assigns a new `customer_id` for every order, even from the same real-world customer. The persistent identity lives in `customer_unique_id`. All RFM and clustering work here uses `customer_unique_id`.

### Finding: a 3.12% repeat purchase rate

Grouping by `customer_unique_id` reveals that only 3.12% of customers (2,997 of 96,096) ever placed more than one order. Standard RFM assumes enough spread in Frequency to build a meaningful 5-way quintile score; with Frequency this flat, that would produce arbitrary, non-meaningful buckets.

**Adaptation made:** Recency and Monetary are scored normally; Frequency is instead captured as a binary `is_repeat_customer` flag, a deliberate design choice made after checking the data's actual shape.

### RFM base table

Built from delivered orders only, using `customer_unique_id`, with `payment_value` summed per order to handle orders with multiple payment records.

- **Recency:** ranges 1-714 days (median 219)
- **Monetary:** right-skewed, median ~$108, mean ~$165, max ~$13,664 (log-transformed before clustering)

**Data cleanup:** one customer had `monetary = 0.00`, traced to a single order with no matching payment record. Excluded. Final table: 93,357 customers.

---

## 3. Enriched Clustering (K-Means)

### Why go beyond bare RFM

With Frequency essentially flat, clustering on R/F/M alone would really only cluster on two variables. The feature set was enriched with category diversity, average freight-to-price ratio, average payment installments, and average review score given. Delivery punctuality and geographic region were deliberately excluded from clustering inputs and reserved as post-hoc profiling lenses, since punctuality is the treatment variable for Week 3's causal analysis and region is better used to interpret clusters than to form them.

### Iteration 1: payment type dominated the clusters

An initial pass one-hot encoded payment type alongside the behavioral features. This produced strong-looking separation (silhouette ~0.37 at k=3), but each cluster mapped almost perfectly onto a single payment type. The clustering had simply rediscovered payment type, not found a genuinely new segmentation. Payment type was removed from clustering inputs.

### Iteration 2: a data quality catch

Re-running on purely behavioral features produced a cluster with `category_diversity = 0`. Every order in that cluster had a missing `product_category_name`. This was a data artifact, not a real segment. Nulls were filled with "unknown" before recomputing, which also shifted the optimal k (from a silhouette peak at k=7 pre-fix to k=5 post-fix).

### Choosing k

Both the elbow method and silhouette score were used, since inertia decreases monotonically with more clusters almost by definition. Silhouette score was treated as the more trustworthy signal.

| k | Score |
|---|---|
| 2 | 0.190 |
| 3 | 0.208 |
| 4 | 0.223 |
| **5** | **0.256** |
| 6 | 0.253 |
| 7 | 0.232 |

k=5 was selected as the clear peak.

### Final Segments

| Segment | Size | % | Description |
|---|---|---|---|
| **Satisfied Core** | 43,147 | 46.5% | Largest segment. Average spend and freight burden, but notably high satisfaction (4.75 avg review score). |
| **Low-Spend, High-Freight** | 18,829 | 20.3% | Lowest average spend, but by far the highest freight-to-price ratio. Still relatively satisfied (4.47). |
| **Dissatisfied Segment** | 14,418 | 15.5% | The standout finding. Nearly identical to the Satisfied Core on spend, recency, and freight, but review score averages just 1.79. Motivating question for Week 3's causal analysis. |
| **High-Value Installment Payers** | 14,047 | 15.1% | Highest spend, heavy installment use, 99.9% credit card. |
| **Repeat / Multi-Category Buyers** | 2,313 | 2.5% | The dataset's true repeat customers, consistent with the 3.12% repeat-purchase finding. |

---

## Key Takeaways

- Applying a standard framework without checking whether its assumptions hold would have produced a misleading segmentation here.
- A first clustering pass that looked strong (silhouette ~0.37) was actually a weaker result than it appeared, since it had rediscovered a categorical variable already visible via a simple groupby.
- A data quality issue materially changed the optimal number of clusters once fixed, a reminder that clustering results are highly sensitive to feature construction.
- The clustering surfaced a genuinely actionable finding, a Dissatisfied Segment indistinguishable from the largest satisfied segment on every feature used, setting up the causal inference work in Week 3.

## Tools Used

`pandas`, `numpy`, `scikit-learn` (KMeans, StandardScaler, silhouette_score), `matplotlib`

## Next Steps (Week 2)

A delivery delay risk model, using the High-Value Installment Payers and Dissatisfied Segment clusters as reference points for what "high value" and "at-risk" customers look like in this dataset.
