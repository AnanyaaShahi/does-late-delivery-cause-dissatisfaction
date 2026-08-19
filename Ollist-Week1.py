#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 13:07:12 2026

@author: ananyaashahi
"""

import pandas as pd
import os

data_path = "/Users/ananyaashahi/Desktop/Ollist/"

files = {
    "customers": "olist_customers_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv"
}

dfs = {}
for name, filename in files.items():
    dfs[name] = pd.read_csv(os.path.join(data_path, filename))
    print(f"{name}: {dfs[name].shape}")
    
#%%
# making sure order_id, customer_id, seller_id, product_id are actually unique where they should be
print("Duplicate order_ids in orders:", dfs["orders"]["order_id"].duplicated().sum())
print("Duplicate customer_ids in customers:", dfs["customers"]["customer_id"].duplicated().sum())
print("Duplicate seller_ids in sellers:", dfs["sellers"]["seller_id"].duplicated().sum())
print("Duplicate product_ids in products:", dfs["products"]["product_id"].duplicated().sum())

# every order should trace back to a real customer - flagging any that don't
orphan_orders = ~dfs["orders"]["customer_id"].isin(dfs["customers"]["customer_id"])
print("\nOrders with no matching customer:", orphan_orders.sum())

# same idea for order_items -> orders
orphan_items = ~dfs["order_items"]["order_id"].isin(dfs["orders"]["order_id"])
print("Order items with no matching order:", orphan_items.sum())

# payments -> orders
orphan_payments = ~dfs["payments"]["order_id"].isin(dfs["orders"]["order_id"])
print("Payments with no matching order:", orphan_payments.sum())

# reviews -> orders
orphan_reviews = ~dfs["reviews"]["order_id"].isin(dfs["orders"]["order_id"])
print("Reviews with no matching order:", orphan_reviews.sum())

# expecting all of these to be 0 - if not, need to decide how to handle before moving to RFM 




#%%

# checking missing values across the orders table - delivery dates matter a lot later for the causal piece
print("Missing values in orders:")
print(dfs["orders"].isnull().sum())

# checking missing values in reviews - comment text especially, since NLP depends on it
print("\nMissing values in reviews:")
print(dfs["reviews"].isnull().sum())

# quick look at order status breakdown - delivery date will only exist for delivered orders
print("\nOrder status breakdown:")
print(dfs["orders"]["order_status"].value_counts())


#%%


# using customer_unique_id since customer_id resets per order - need the real repeat-purchase picture
merged = dfs["orders"].merge(dfs["customers"][["customer_id", "customer_unique_id"]], on="customer_id", how="left")

orders_per_customer = merged.groupby("customer_unique_id")["order_id"].nunique()

print("Total unique customers:", orders_per_customer.shape[0])
print("\nDistribution of orders per customer:")
print(orders_per_customer.value_counts().sort_index())

repeat_rate = (orders_per_customer > 1).sum() / orders_per_customer.shape[0]
print(f"\nRepeat purchase rate: {repeat_rate:.2%}")



#%%

# building the base customer-level table using customer_unique_id, not customer_id
# (customer_id resets per order, customer_unique_id is the real persistent identity)

# summing payment_value per order first, since one order can have multiple payment rows
order_payments = dfs["payments"].groupby("order_id")["payment_value"].sum().reset_index()

# merging orders with customer_unique_id and the order-level payment total
orders_customers = dfs["orders"].merge(
    dfs["customers"][["customer_id", "customer_unique_id"]], on="customer_id", how="left"
).merge(
    order_payments, on="order_id", how="left"
)

# only using delivered orders for recency/monetary - undelivered orders don't reflect real purchase behavior
delivered = orders_customers[orders_customers["order_status"] == "delivered"].copy()
delivered["order_purchase_timestamp"] = pd.to_datetime(delivered["order_purchase_timestamp"])

# setting a reference date - one day after the latest purchase in the dataset, standard practice for recency calcs
reference_date = delivered["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

rfm = delivered.groupby("customer_unique_id").agg(
    recency_days=("order_purchase_timestamp", lambda x: (reference_date - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("payment_value", "sum")
).reset_index()

# repeat-purchase flag instead of a scored frequency, since frequency has almost no spread
rfm["is_repeat_customer"] = rfm["frequency"] > 1

print(rfm.shape)
print(rfm.head())
print("\nRecency stats:\n", rfm["recency_days"].describe())
print("\nMonetary stats:\n", rfm["monetary"].describe())

#%%

# checking who these zero-payment customers actually are - could be legit (free promo items) or a data quirk
zero_monetary = rfm[rfm["monetary"] == 0]
print("Number of customers with $0 monetary:", zero_monetary.shape[0])
print(zero_monetary)

# pulling the actual orders behind these to see what's going on
zero_customer_ids = zero_monetary["customer_unique_id"].tolist()
zero_orders = orders_customers[orders_customers["customer_unique_id"].isin(zero_customer_ids)]
print("\nOrder details for these customers:")
print(zero_orders[["order_id", "customer_unique_id", "order_status", "payment_value"]])

#%%
# single order with no matching payment record (not a real $0 purchase) - dropping before clustering/scaling
rfm = rfm[rfm["customer_unique_id"] != "830d5b7aaa3b6f1e9ad63703bec97d23"].copy()
print(rfm.shape)

#%%

# category diversity - how many distinct product categories has this customer bought from
order_items_products = dfs["order_items"].merge(
    dfs["products"][["product_id", "product_category_name"]], on="product_id", how="left"
)
order_items_customers = order_items_products.merge(
    dfs["orders"][["order_id", "customer_id"]], on="order_id", how="left"
).merge(
    dfs["customers"][["customer_id", "customer_unique_id"]], on="customer_id", how="left"
)

category_diversity = order_items_customers.groupby("customer_unique_id")["product_category_name"].nunique().reset_index()
category_diversity.columns = ["customer_unique_id", "category_diversity"]

# freight-to-price ratio - avg freight_value / price per item, per customer
order_items_customers["freight_ratio"] = order_items_customers["freight_value"] / order_items_customers["price"]
avg_freight_ratio = order_items_customers.groupby("customer_unique_id")["freight_ratio"].mean().reset_index()
avg_freight_ratio.columns = ["customer_unique_id", "avg_freight_ratio"]

# avg installments used - from payments table
payments_customers = dfs["payments"].merge(
    dfs["orders"][["order_id", "customer_id"]], on="order_id", how="left"
).merge(
    dfs["customers"][["customer_id", "customer_unique_id"]], on="customer_id", how="left"
)
avg_installments = payments_customers.groupby("customer_unique_id")["payment_installments"].mean().reset_index()
avg_installments.columns = ["customer_unique_id", "avg_installments"]

# avg review score given - from reviews table
reviews_customers = dfs["reviews"].merge(
    dfs["orders"][["order_id", "customer_id"]], on="order_id", how="left"
).merge(
    dfs["customers"][["customer_id", "customer_unique_id"]], on="customer_id", how="left"
)
avg_review_score = reviews_customers.groupby("customer_unique_id")["review_score"].mean().reset_index()
avg_review_score.columns = ["customer_unique_id", "avg_review_score"]

# merging all enrichment features onto the rfm base table
rfm_enriched = rfm.merge(category_diversity, on="customer_unique_id", how="left") \
                   .merge(avg_freight_ratio, on="customer_unique_id", how="left") \
                   .merge(avg_installments, on="customer_unique_id", how="left") \
                   .merge(avg_review_score, on="customer_unique_id", how="left")

print(rfm_enriched.shape)
print(rfm_enriched.head())
print("\nMissing values after merge:\n", rfm_enriched.isnull().sum())


#%%

# payment type - categorical, low cardinality, one-hot encoding is reasonable here
# taking the most frequently used payment type per customer (in case someone used more than one type across orders)
preferred_payment = payments_customers.groupby("customer_unique_id")["payment_type"].agg(
    lambda x: x.mode()[0] if not x.mode().empty else None
).reset_index()
preferred_payment.columns = ["customer_unique_id", "preferred_payment_type"]

print(preferred_payment["preferred_payment_type"].value_counts())

# merging onto the enriched table
rfm_enriched = rfm_enriched.merge(preferred_payment, on="customer_unique_id", how="left")

print(rfm_enriched.shape)
print(rfm_enriched["preferred_payment_type"].isnull().sum())

#%%

# dropping the 603 customers with no review score - better than imputing a satisfaction signal we don't actually know
rfm_clustering_ready = rfm_enriched.dropna(subset=["avg_review_score"]).copy()
print(rfm_clustering_ready.shape)

# one-hot encoding payment type - folding the 2 "not_defined" into a fallback category to avoid a near-empty column
rfm_clustering_ready["preferred_payment_type"] = rfm_clustering_ready["preferred_payment_type"].replace("not_defined", "boleto")
payment_dummies = pd.get_dummies(rfm_clustering_ready["preferred_payment_type"], prefix="payment")
rfm_clustering_ready = pd.concat([rfm_clustering_ready, payment_dummies], axis=1)

print(rfm_clustering_ready.columns.tolist())

#%%

from sklearn.preprocessing import StandardScaler
import numpy as np

# log-transforming monetary and freight ratio since both are right-skewed - keeps outliers from dominating distance calcs
rfm_clustering_ready["monetary_log"] = np.log1p(rfm_clustering_ready["monetary"])
rfm_clustering_ready["avg_freight_ratio_log"] = np.log1p(rfm_clustering_ready["avg_freight_ratio"])

# selecting the features going into clustering - numeric + log-transformed + payment dummies
cluster_features = [
    "recency_days", "monetary_log", "category_diversity",
    "avg_freight_ratio_log", "avg_installments", "avg_review_score",
    "payment_boleto", "payment_credit_card", "payment_debit_card", "payment_voucher"
]

X = rfm_clustering_ready[cluster_features].copy()

# checking for any inf/nan introduced by the log transform (log(0) issues, division by zero in freight ratio, etc.)
print(X.isnull().sum())
print(np.isinf(X).sum())


#%%

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# scaling everything to mean 0, std 1 - so no single feature dominates just because of its raw magnitude
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# elbow method - checking how much "within-cluster spread" drops as we add more clusters
# looking for the point where adding another cluster stops giving a meaningful improvement
inertia = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

import matplotlib.pyplot as plt
plt.plot(list(k_range), inertia, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("Elbow Method for Optimal k")
plt.show()

#%%

from sklearn.metrics import silhouette_score

# silhouette score: measures how well-separated clusters are (higher = better separation, ranges -1 to 1)
# cross-checking against the elbow's soft bend around k=5
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k={k}: silhouette score = {score:.4f}")

plt.plot(list(k_range), silhouette_scores, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by k")
plt.show()

#%%

# going with k=3 based on silhouette score (tied with k=4, but 3 is simpler/more interpretable, and elbow's inertia curve doesn't override this since inertia always decreases with more k)
final_k = 3
kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
rfm_clustering_ready["cluster"] = kmeans_final.fit_predict(X_scaled)

print(rfm_clustering_ready["cluster"].value_counts())

# profiling each cluster - what does each group actually look like on the original (unscaled) features
cluster_profile = rfm_clustering_ready.groupby("cluster")[
    ["recency_days", "monetary", "frequency", "category_diversity", "avg_freight_ratio", "avg_installments", "avg_review_score"]
].mean()
print("\nCluster profiles (mean values):\n", cluster_profile)

#%%

# printing without truncation so we can see every feature per cluster
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(cluster_profile)

# also useful: cluster sizes as % of total, and payment type breakdown per cluster
print("\nCluster sizes (%):")
print(rfm_clustering_ready["cluster"].value_counts(normalize=True) * 100)

print("\nPayment type distribution by cluster:")
print(pd.crosstab(rfm_clustering_ready["cluster"], rfm_clustering_ready["preferred_payment_type"], normalize="index") * 100)

#%%

# dropping payment type dummies - they were dominating distance calcs and just re-surfacing payment_type itself
# keeping payment type out of clustering inputs, will bring it back later purely for profiling/interpretation
cluster_features_v2 = [
    "recency_days", "monetary_log", "category_diversity",
    "avg_freight_ratio_log", "avg_installments", "avg_review_score"
]

X_v2 = rfm_clustering_ready[cluster_features_v2].copy()

scaler_v2 = StandardScaler()
X_v2_scaled = scaler_v2.fit_transform(X_v2)

# re-running elbow method on the reduced, purely behavioral feature set
inertia_v2 = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_v2_scaled)
    inertia_v2.append(km.inertia_)

plt.plot(list(k_range), inertia_v2, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method (v2 - no payment type)")
plt.show()

# re-running silhouette score too
silhouette_scores_v2 = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_v2_scaled)
    score = silhouette_score(X_v2_scaled, labels)
    silhouette_scores_v2.append(score)
    print(f"k={k}: silhouette score = {score:.4f}")

plt.plot(list(k_range), silhouette_scores_v2, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score (v2 - no payment type)")
plt.show()
#%%
final_k_v2 = 7
kmeans_v2 = KMeans(n_clusters=final_k_v2, random_state=42, n_init=10)
rfm_clustering_ready["cluster_v2"] = kmeans_v2.fit_predict(X_v2_scaled)

print(rfm_clustering_ready["cluster_v2"].value_counts())

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

cluster_profile_v2 = rfm_clustering_ready.groupby("cluster_v2")[
    ["recency_days", "monetary", "frequency", "category_diversity", "avg_freight_ratio", "avg_installments", "avg_review_score"]
].mean()
print("\nCluster profiles (v2, k=7):\n", cluster_profile_v2)

print("\nPayment type by cluster (v2):")
print(pd.crosstab(rfm_clustering_ready["cluster_v2"], rfm_clustering_ready["preferred_payment_type"], normalize="index") * 100)

#%%

# investigating cluster 5's zero category diversity - checking for missing category names
cluster_5_customers = rfm_clustering_ready[rfm_clustering_ready["cluster_v2"] == 5]["customer_unique_id"]
cluster_5_items = order_items_customers[order_items_customers["customer_unique_id"].isin(cluster_5_customers)]
print("Missing product_category_name in cluster 5's orders:", cluster_5_items["product_category_name"].isnull().sum())
print("Total order_items rows for cluster 5:", cluster_5_items.shape[0])
#%%
# fixing the root cause: nulls in product_category_name were making category_diversity artificially 0
# filling nulls with "unknown" so these customers get counted as having bought from 1 (unknown) category, not 0
order_items_customers["product_category_name"] = order_items_customers["product_category_name"].fillna("unknown")

# rebuilding category_diversity with the fix
category_diversity_fixed = order_items_customers.groupby("customer_unique_id")["product_category_name"].nunique().reset_index()
category_diversity_fixed.columns = ["customer_unique_id", "category_diversity"]

# re-merging the corrected feature back onto our clustering-ready table
rfm_clustering_ready = rfm_clustering_ready.drop(columns=["category_diversity"]).merge(
    category_diversity_fixed, on="customer_unique_id", how="left"
)

print(rfm_clustering_ready["category_diversity"].describe())
print("Any zeros left?", (rfm_clustering_ready["category_diversity"] == 0).sum())

#%%

# rebuilding X with the corrected category_diversity, log-transforming monetary/freight ratio again
X_v2 = rfm_clustering_ready[[
    "recency_days", "monetary_log", "category_diversity",
    "avg_freight_ratio_log", "avg_installments", "avg_review_score"
]].copy()

scaler_v2 = StandardScaler()
X_v2_scaled = scaler_v2.fit_transform(X_v2)

# re-running silhouette check across k=2 to 10 with the corrected data
silhouette_scores_v3 = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_v2_scaled)
    score = silhouette_score(X_v2_scaled, labels)
    silhouette_scores_v3.append(score)
    print(f"k={k}: silhouette score = {score:.4f}")

plt.plot(list(k_range), silhouette_scores_v3, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score (v3 - category diversity fixed)")
plt.show()


#%%

final_k_v3 = 5
kmeans_v3 = KMeans(n_clusters=final_k_v3, random_state=42, n_init=10)
rfm_clustering_ready["cluster_final"] = kmeans_v3.fit_predict(X_v2_scaled)

print(rfm_clustering_ready["cluster_final"].value_counts())

cluster_profile_final = rfm_clustering_ready.groupby("cluster_final")[
    ["recency_days", "monetary", "frequency", "category_diversity", "avg_freight_ratio", "avg_installments", "avg_review_score"]
].mean()
print("\nFinal cluster profiles (k=5):\n", cluster_profile_final)

print("\nPayment type by cluster (final):")
print(pd.crosstab(rfm_clustering_ready["cluster_final"], rfm_clustering_ready["preferred_payment_type"], normalize="index") * 100)

#%%
####WEEk 2 - Our Model ######


# defining lateness using only delivered orders - "late" doesn't apply to orders that never arrived
delivered_orders = dfs["orders"][dfs["orders"]["order_status"] == "delivered"].copy()

delivered_orders["order_estimated_delivery_date"] = pd.to_datetime(delivered_orders["order_estimated_delivery_date"])
delivered_orders["order_delivered_customer_date"] = pd.to_datetime(delivered_orders["order_delivered_customer_date"])

delivered_orders["is_late"] = delivered_orders["order_delivered_customer_date"] > delivered_orders["order_estimated_delivery_date"]

print("Total delivered orders:", delivered_orders.shape[0])
print("Late orders:", delivered_orders["is_late"].sum())
print("Late rate:", delivered_orders["is_late"].mean())


#%%
# starting from delivered_orders (already has is_late), building out order-level features

# order timing features - day of week and month of purchase, possible seasonality effect on delays
delivered_orders["order_purchase_timestamp"] = pd.to_datetime(dfs["orders"].set_index("order_id").loc[delivered_orders["order_id"], "order_purchase_timestamp"].values)
delivered_orders["purchase_dow"] = delivered_orders["order_purchase_timestamp"].dt.dayofweek  # 0=Monday
delivered_orders["purchase_month"] = delivered_orders["order_purchase_timestamp"].dt.month

# aggregating order_items to order level - some orders have multiple items/sellers
# using primary seller (first seller_id) for simplicity, and summing item count/weight/freight
order_items_agg = dfs["order_items"].merge(
    dfs["products"][["product_id", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]],
    on="product_id", how="left"
)

order_level_features = order_items_agg.groupby("order_id").agg(
    item_count=("order_item_id", "count"),
    total_freight=("freight_value", "sum"),
    avg_price=("price", "mean"),
    total_weight_g=("product_weight_g", "sum"),
    primary_seller_id=("seller_id", "first")  # using first seller as primary for distance calc
).reset_index()

# merging onto delivered_orders
delivered_orders = delivered_orders.merge(order_level_features, on="order_id", how="left")

print(delivered_orders.shape)
print(delivered_orders[["order_id", "purchase_dow", "purchase_month", "item_count", "total_freight", "total_weight_g"]].head())
print("\nMissing values:\n", delivered_orders.isnull().sum())

#%%

# checking how many orders actually have more than 1 seller - determines if "primary_seller_id" simplification is safe
sellers_per_order = dfs["order_items"].groupby("order_id")["seller_id"].nunique()
print("Orders with more than 1 seller:", (sellers_per_order > 1).sum())
print("Total orders:", sellers_per_order.shape[0])
print("% multi-seller:", (sellers_per_order > 1).mean() * 100)

#%%
# checking what payments actually look like - specifically orders with more than 1 payment row
payment_counts = dfs["payments"].groupby("order_id").size()
print("Orders with more than 1 payment row:", (payment_counts > 1).sum())
print("Total orders in payments table:", payment_counts.shape[0])

# pulling a few real examples of multi-payment orders to see the pattern
multi_payment_order_ids = payment_counts[payment_counts > 1].index[:5]
print("\nExample multi-payment orders:")
print(dfs["payments"][dfs["payments"]["order_id"].isin(multi_payment_order_ids)].sort_values("order_id"))

#%%

# checking payment_type breakdown specifically within multi-payment orders
multi_payment_orders = dfs["payments"][dfs["payments"]["order_id"].isin(payment_counts[payment_counts > 1].index)]
print(multi_payment_orders["payment_type"].value_counts())

# checking if any order has more than 1 credit_card row (the case where .max() logic might get murkier)
credit_card_counts = multi_payment_orders[multi_payment_orders["payment_type"] == "credit_card"].groupby("order_id").size()
print("\nOrders with multiple credit_card rows:", (credit_card_counts > 1).sum())


#%%
# looking at a few examples where an order has multiple credit_card rows - does .max() still make sense here?
multi_cc_order_ids = credit_card_counts[credit_card_counts > 1].index[:5]
print(dfs["payments"][dfs["payments"]["order_id"].isin(multi_cc_order_ids)].sort_values(["order_id", "payment_sequential"]))

#%%

# rebuilding order_items_agg to include product_category_name this time (it was missed in the original merge)
order_items_agg = dfs["order_items"].merge(
    dfs["products"][["product_id", "product_category_name", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]],
    on="product_id", how="left"
)

# now rebuilding volume and the category merge on top of the corrected order_items_agg
order_items_agg["volume_cm3"] = (
    order_items_agg["product_length_cm"] * order_items_agg["product_height_cm"] * order_items_agg["product_width_cm"]
)

order_items_with_category = order_items_agg.merge(
    dfs["category_translation"], on="product_category_name", how="left"
)
order_items_with_category["product_category_name_english"] = order_items_with_category["product_category_name_english"].fillna(
    order_items_with_category["product_category_name"]
)

order_level_features = order_items_with_category.groupby("order_id").agg(
    item_count=("order_item_id", "count"),
    total_freight=("freight_value", "sum"),
    avg_price=("price", "mean"),
    total_weight_g=("product_weight_g", "sum"),
    total_volume_cm3=("volume_cm3", "sum"),
    primary_seller_id=("seller_id", "first"),
    primary_category=("product_category_name_english", "first")
).reset_index()

order_installments = dfs["payments"].groupby("order_id")["payment_installments"].max().reset_index()
order_installments.columns = ["order_id", "payment_installments"]

delivered_orders = delivered_orders.drop(columns=["item_count", "total_freight", "avg_price", "total_weight_g", "primary_seller_id"], errors="ignore")
delivered_orders = delivered_orders.merge(order_level_features, on="order_id", how="left")
delivered_orders = delivered_orders.merge(order_installments, on="order_id", how="left")

print(delivered_orders.shape)
print(delivered_orders[["order_id", "total_volume_cm3", "primary_category", "payment_installments"]].head())
print("\nMissing values:\n", delivered_orders.isnull().sum())

#%%

# filling missing category with "unknown" - same known data quirk from Week 1 (some products lack a category label)
delivered_orders["primary_category"] = delivered_orders["primary_category"].fillna("unknown")

# checking the 1 missing payment_installments row - confirming it's the same known missing-payment order from Week 1
print(delivered_orders[delivered_orders["payment_installments"].isnull()][["order_id", "customer_id"]])

#%%
# same known missing-payment order from Week 1 - dropping here too for consistency
delivered_orders = delivered_orders[delivered_orders["order_id"] != "bfbd0f9bdef84302105ad712db648a6c"].copy()

print(delivered_orders.shape)
print("Remaining missing values:\n", delivered_orders.isnull().sum().sum(), "total missing values left (excluding date columns we don't use as features)")
#%%

# geolocation has multiple lat/long entries per zip prefix - averaging to get one representative point per prefix
geo_avg = dfs["geolocation"].groupby("geolocation_zip_code_prefix").agg(
    lat=("geolocation_lat", "mean"),
    lng=("geolocation_lng", "mean")
).reset_index()

print(geo_avg.shape)

# checking match coverage before committing to this approach
print("Unique seller zip prefixes:", dfs["sellers"]["seller_zip_code_prefix"].nunique())
print("Unique customer zip prefixes:", dfs["customers"]["customer_zip_code_prefix"].nunique())
print("Sellers matchable to geo_avg:", dfs["sellers"]["seller_zip_code_prefix"].isin(geo_avg["geolocation_zip_code_prefix"]).sum(), "/", dfs["sellers"].shape[0])
print("Customers matchable to geo_avg:", dfs["customers"]["customer_zip_code_prefix"].isin(geo_avg["geolocation_zip_code_prefix"]).sum(), "/", dfs["customers"].shape[0])

#%%

import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    # haversine formula - great-circle distance between two lat/long points, in km
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# merging seller lat/long onto delivered_orders via primary_seller_id -> seller_zip_code_prefix
sellers_geo = dfs["sellers"].merge(geo_avg, left_on="seller_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left")
delivered_orders = delivered_orders.merge(
    sellers_geo[["seller_id", "lat", "lng"]].rename(columns={"lat": "seller_lat", "lng": "seller_lng"}),
    left_on="primary_seller_id", right_on="seller_id", how="left"
)

# merging customer lat/long onto delivered_orders via customer_id -> customer_zip_code_prefix
customers_geo = dfs["customers"].merge(geo_avg, left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left")
delivered_orders = delivered_orders.merge(
    customers_geo[["customer_id", "lat", "lng"]].rename(columns={"lat": "customer_lat", "lng": "customer_lng"}),
    on="customer_id", how="left"
)

# calculating distance
delivered_orders["seller_customer_distance_km"] = haversine_distance(
    delivered_orders["seller_lat"], delivered_orders["seller_lng"],
    delivered_orders["customer_lat"], delivered_orders["customer_lng"]
)

print(delivered_orders["seller_customer_distance_km"].describe())
print("\nMissing distances:", delivered_orders["seller_customer_distance_km"].isnull().sum())

#%%
# dropping rows with missing distance - small enough (0.5%) that dropping is cleaner than imputing a fake distance
delivered_orders = delivered_orders.dropna(subset=["seller_customer_distance_km"]).copy()
print(delivered_orders.shape)

# quick sanity check on the max distance - is 8677 km plausible for Brazil, or a data anomaly?
print(delivered_orders.nlargest(3, "seller_customer_distance_km")[["order_id", "seller_lat", "seller_lng", "customer_lat", "customer_lng", "seller_customer_distance_km"]])
#%%

# checking the customer_state for these specific "implausible" orders - a real cross-border order would likely
# still have a valid Brazilian state code if it went through Olist's normal system, or a clearly different pattern
suspicious_order_ids = ["8ad3f1d0f96992e43566c4c82c9f6c58", "acdbc7396e191931c263db11af241d62", "4d5abe7999d76d1fb6237d3677706af0"]
suspicious_customers = delivered_orders[delivered_orders["order_id"].isin(suspicious_order_ids)]["customer_id"]

print(dfs["customers"][dfs["customers"]["customer_id"].isin(suspicious_customers)][["customer_id", "customer_city", "customer_state", "customer_zip_code_prefix"]])


#%%

# confirmed data quality issue in geolocation table (not real cross-border orders) - dropping these implausible-coordinate rows
delivered_orders = delivered_orders[
    ~((delivered_orders["customer_lat"] < -34) | (delivered_orders["customer_lat"] > 6) |
      (delivered_orders["customer_lng"] < -75) | (delivered_orders["customer_lng"] > -33) |
      (delivered_orders["seller_lat"] < -34) | (delivered_orders["seller_lat"] > 6) |
      (delivered_orders["seller_lng"] < -75) | (delivered_orders["seller_lng"] > -33))
].copy()

print(delivered_orders.shape)
print(delivered_orders["seller_customer_distance_km"].describe())

#%%

# sorting chronologically before splitting - train on earlier orders, test on more recent ones
delivered_orders = delivered_orders.sort_values("order_purchase_timestamp").reset_index(drop=True)

split_index = int(len(delivered_orders) * 0.8)
split_date = delivered_orders.iloc[split_index]["order_purchase_timestamp"]

print("Split date:", split_date)
print("Train date range:", delivered_orders["order_purchase_timestamp"].min(), "to", split_date)
print("Test date range:", split_date, "to", delivered_orders["order_purchase_timestamp"].max())

train = delivered_orders[delivered_orders["order_purchase_timestamp"] < split_date].copy()
test = delivered_orders[delivered_orders["order_purchase_timestamp"] >= split_date].copy()

print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)
print("Train late rate:", train["is_late"].mean())
print("Test late rate:", test["is_late"].mean())
#%%

# 1. confirming no row duplication happened from any of our merges - should still be 1 row per order_id
print("Duplicate order_ids in delivered_orders:", delivered_orders["order_id"].duplicated().sum())

# 2. checking for $0 or negative price/freight - possible data errors, not real promotional patterns
print("\nOrders with price <= 0:", (delivered_orders["avg_price"] <= 0).sum())
print("Orders with freight <= 0:", (delivered_orders["total_freight"] <= 0).sum())

# 3. multi-category orders - how often does "primary_category" (first item) misrepresent the order?
order_items_with_category_check = order_items_with_category.groupby("order_id")["product_category_name_english"].nunique()
print("\nOrders spanning multiple categories:", (order_items_with_category_check > 1).sum(), "out of", order_items_with_category_check.shape[0])

# 4. late rate by category - checking if lateness is suspiciously concentrated in one category (proxy risk)
print("\nLate rate by category (top 10 by order count):")
category_late = delivered_orders.groupby("primary_category").agg(orders=("is_late", "count"), late_rate=("is_late", "mean")).sort_values("orders", ascending=False)
print(category_late.head(10))

# 5. late rate by customer state - same check, geographically
print("\nLate rate by top 10 states:")
state_late = delivered_orders.merge(dfs["customers"][["customer_id", "customer_state"]], on="customer_id", how="left")
print(state_late.groupby("customer_state").agg(orders=("is_late", "count"), late_rate=("is_late", "mean")).sort_values("orders", ascending=False).head(10))

# 6. category overlap between train and test sets (relevant once we one-hot encode)
train_categories = set(train["primary_category"].unique())
test_categories = set(test["primary_category"].unique())
print("\nCategories in test but not train:", test_categories - train_categories)
print("Categories in train but not test:", len(train_categories - test_categories))

# 7. sellers in test set with zero training history - will need a fallback on-time rate for these
train_sellers = set(train["primary_seller_id"].unique())
test_sellers = set(test["primary_seller_id"].unique())
unseen_sellers_in_test = test_sellers - train_sellers
print("\nSellers in test with NO training history:", len(unseen_sellers_in_test), "out of", len(test_sellers), "test sellers")

#%%

# checking what these $0 freight orders actually look like - legit free shipping, or a data issue?
zero_freight = delivered_orders[delivered_orders["total_freight"] <= 0]
print(zero_freight[["order_id", "avg_price", "total_freight", "primary_category", "is_late"]].head(10))
print("\nLate rate for zero-freight orders:", zero_freight["is_late"].mean())
print("Late rate for all orders:", delivered_orders["is_late"].mean())


#%%


# minimum threshold for trusting a seller's own on-time rate
MIN_ORDERS_THRESHOLD = 5

# seller stats computed ONLY from training data - leakage-safe
seller_stats = train.groupby("primary_seller_id")["is_late"].agg(
    seller_total_orders="count",
    seller_late_rate="mean"
).reset_index()
seller_stats["seller_on_time_rate_raw"] = 1 - seller_stats["seller_late_rate"]

# overall training average on-time rate - fallback for new/thin-history sellers
overall_on_time_rate = 1 - train["is_late"].mean()
print("Overall training on-time rate (fallback value):", overall_on_time_rate)

# flagging which sellers have enough history to trust their own rate
seller_stats["has_sufficient_history"] = seller_stats["seller_total_orders"] >= MIN_ORDERS_THRESHOLD

# final seller on-time rate: own rate if sufficient history, else fallback to overall average
seller_stats["seller_on_time_rate"] = np.where(
    seller_stats["has_sufficient_history"],
    seller_stats["seller_on_time_rate_raw"],
    overall_on_time_rate
)

# merging onto both train and test - sellers not seen in training at all get the fallback + flagged as new
def merge_seller_feature(df, seller_stats, overall_rate):
    df = df.merge(
        seller_stats[["primary_seller_id", "seller_on_time_rate", "has_sufficient_history"]],
        on="primary_seller_id", how="left"
    )
    # sellers with zero training history entirely (not in seller_stats at all) get the fallback
    df["seller_on_time_rate"] = df["seller_on_time_rate"].fillna(overall_rate)
    df["has_sufficient_history"] = df["has_sufficient_history"].fillna(False)
    df["is_new_or_thin_history_seller"] = ~df["has_sufficient_history"]
    return df.drop(columns=["has_sufficient_history"])

train = merge_seller_feature(train, seller_stats, overall_on_time_rate)
test = merge_seller_feature(test, seller_stats, overall_on_time_rate)

print("\nTrain seller_on_time_rate summary:\n", train["seller_on_time_rate"].describe())
print("\nTest seller_on_time_rate summary:\n", test["seller_on_time_rate"].describe())
print("\n% flagged as new/thin-history in test:", test["is_new_or_thin_history_seller"].mean())

#%%

# quick fix for the FutureWarning - explicit dtype handling instead of relying on fillna's default behavior
def merge_seller_feature(df, seller_stats, overall_rate):
    df = df.merge(
        seller_stats[["primary_seller_id", "seller_on_time_rate", "has_sufficient_history"]],
        on="primary_seller_id", how="left"
    )
    df["seller_on_time_rate"] = df["seller_on_time_rate"].fillna(overall_rate)
    df["has_sufficient_history"] = df["has_sufficient_history"].astype("boolean").fillna(False).astype(bool)
    df["is_new_or_thin_history_seller"] = ~df["has_sufficient_history"]
    return df.drop(columns=["has_sufficient_history"])

# re-running with the fix
train = train.drop(columns=["seller_on_time_rate", "is_new_or_thin_history_seller"], errors="ignore")
test = test.drop(columns=["seller_on_time_rate", "is_new_or_thin_history_seller"], errors="ignore")

train = merge_seller_feature(train, seller_stats, overall_on_time_rate)
test = merge_seller_feature(test, seller_stats, overall_on_time_rate)

print("No warning this time, confirming shapes:", train.shape, test.shape)

#%%

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# baseline feature set - numeric only, no category, keeping this one interpretable
baseline_features = [
    "seller_customer_distance_km", "total_weight_g", "total_volume_cm3",
    "total_freight", "avg_price", "item_count", "payment_installments",
    "seller_on_time_rate", "is_new_or_thin_history_seller",
    "purchase_dow", "purchase_month"
]

X_train = train[baseline_features].copy()
y_train = train["is_late"]
X_test = test[baseline_features].copy()
y_test = test["is_late"]

# converting the boolean flag to int so scaling doesn't choke on it
X_train["is_new_or_thin_history_seller"] = X_train["is_new_or_thin_history_seller"].astype(int)
X_test["is_new_or_thin_history_seller"] = X_test["is_new_or_thin_history_seller"].astype(int)

# scaling - logistic regression benefits from features being on comparable scales
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # using train's fit, not re-fitting on test - avoids leakage here too

# class_weight="balanced" - correcting for the 8% imbalance rather than letting the model default to "always predict on-time"
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

y_pred = log_reg.predict(X_test_scaled)
y_pred_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#%%

# checking each numeric feature's individual relationship with is_late - point-biserial correlation
from scipy.stats import pointbiserialr

for col in baseline_features:
    if col == "is_new_or_thin_history_seller":
        continue
    corr, pval = pointbiserialr(train["is_late"], train[col])
    print(f"{col}: correlation={corr:.4f}, p-value={pval:.4g}")

# checking the binary flag separately
late_rate_new_seller = train.groupby("is_new_or_thin_history_seller")["is_late"].mean()
print("\nLate rate by new/thin-history seller flag:\n", late_rate_new_seller)

#%%



# checking the logistic regression's learned coefficients - do they match the correlations we just found?
coef_df = pd.DataFrame({
    "feature": baseline_features,
    "coefficient": log_reg.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)
print(coef_df)

#%%

# checking training performance vs test performance - is this a generalization problem?
y_train_pred_proba = log_reg.predict_proba(X_train_scaled)[:, 1]
print("Train ROC-AUC:", roc_auc_score(y_train, y_train_pred_proba))
print("Test ROC-AUC:", roc_auc_score(y_test, y_pred_proba))

# double-checking the test AUC calculation directly, isolated
print("\nRe-checking test AUC directly:")
test_proba_check = log_reg.predict_proba(X_test_scaled)[:, 1]
print("Test ROC-AUC (recomputed):", roc_auc_score(y_test, test_proba_check))
print("Any NaN in test predictions?", np.isnan(test_proba_check).sum())

#%%

# checking if the relationship between seller_on_time_rate (our strongest feature) and is_late holds similarly in both periods
print("Correlation in train:", pointbiserialr(train["is_late"], train["seller_on_time_rate"])[0])
print("Correlation in test:", pointbiserialr(test["is_late"], test["seller_on_time_rate"])[0])

# checking late rate by month across the whole dataset - is there a visible seasonal pattern?
delivered_orders_temp = pd.concat([train, test])
monthly_late_rate = delivered_orders_temp.groupby(delivered_orders_temp["order_purchase_timestamp"].dt.to_period("M"))["is_late"].agg(["mean", "count"])
print("\nLate rate by month:\n", monthly_late_rate)


#%%


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# combining back into one dataset, sorted chronologically, to build multiple rolling splits from scratch
full_data = pd.concat([train, test]).sort_values("order_purchase_timestamp").reset_index(drop=True)

# dropping the old seller_on_time_rate / is_new_or_thin_history_seller - we'll recompute per fold to avoid leakage each time
full_data = full_data.drop(columns=["seller_on_time_rate", "is_new_or_thin_history_seller"], errors="ignore")

# defining 5 rolling time-based folds - each fold trains on everything before a cutoff, tests on the next chunk after it
n_folds = 5
fold_size = len(full_data) // (n_folds + 1)  # +1 because first chunk is only ever used for training

results = []

for fold in range(1, n_folds + 1):
    train_end = fold_size * fold
    test_end = fold_size * (fold + 1)

    fold_train = full_data.iloc[:train_end].copy()
    fold_test = full_data.iloc[train_end:test_end].copy()

    # recomputing seller_on_time_rate fresh, using ONLY this fold's training data - leakage-safe per fold
    seller_stats_fold = fold_train.groupby("primary_seller_id")["is_late"].agg(
        seller_total_orders="count", seller_late_rate="mean"
    ).reset_index()
    seller_stats_fold["seller_on_time_rate"] = 1 - seller_stats_fold["seller_late_rate"]
    overall_rate_fold = 1 - fold_train["is_late"].mean()
    seller_stats_fold["has_sufficient_history"] = seller_stats_fold["seller_total_orders"] >= 5
    seller_stats_fold.loc[~seller_stats_fold["has_sufficient_history"], "seller_on_time_rate"] = overall_rate_fold

    def apply_seller_feature(df):
        df = df.merge(seller_stats_fold[["primary_seller_id", "seller_on_time_rate", "has_sufficient_history"]], on="primary_seller_id", how="left")
        df["seller_on_time_rate"] = df["seller_on_time_rate"].fillna(overall_rate_fold)
        df["is_new_or_thin_history_seller"] = (~df["has_sufficient_history"].fillna(False).astype(bool)).astype(int)
        return df.drop(columns=["has_sufficient_history"])

    fold_train = apply_seller_feature(fold_train)
    fold_test = apply_seller_feature(fold_test)

    feats = baseline_features
    Xtr, ytr = fold_train[feats], fold_train["is_late"]
    Xte, yte = fold_test[feats], fold_test["is_late"]

    scaler_fold = StandardScaler()
    Xtr_scaled = scaler_fold.fit_transform(Xtr)
    Xte_scaled = scaler_fold.transform(Xte)

    model_fold = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model_fold.fit(Xtr_scaled, ytr)
    proba = model_fold.predict_proba(Xte_scaled)[:, 1]
    auc = roc_auc_score(yte, proba)

    fold_date_range = (fold_test["order_purchase_timestamp"].min(), fold_test["order_purchase_timestamp"].max())
    fold_late_rate = yte.mean()

    results.append({"fold": fold, "test_start": fold_date_range[0], "test_end": fold_date_range[1], "test_late_rate": fold_late_rate, "auc": auc})
    print(f"Fold {fold}: test window {fold_date_range[0].date()} to {fold_date_range[1].date()}, late_rate={fold_late_rate:.3f}, AUC={auc:.4f}")

results_df = pd.DataFrame(results)
print("\n", results_df)

#%%

# checking if fold 5 (the problematic window) looks structurally different - e.g. fewer orders, different seller mix
fold5_start = pd.Timestamp("2018-06-13")
fold5_data = full_data[full_data["order_purchase_timestamp"] >= fold5_start]

print("Fold 5 order count:", fold5_data.shape[0])
print("Unique sellers in fold 5:", fold5_data["primary_seller_id"].nunique())
print("% of fold 5 sellers with training-period history:", fold5_data["primary_seller_id"].isin(full_data[full_data["order_purchase_timestamp"] < fold5_start]["primary_seller_id"]).mean())

# checking overall order volume trend near the end of the dataset - is it tapering off unusually?
weekly_volume = full_data.set_index("order_purchase_timestamp").resample("W")["order_id"].count()
print("\nWeekly order volume, last 10 weeks of data:\n", weekly_volume.tail(10))
#%%


# checking exactly where the data cuts off, and how much of the tail is genuinely truncated/incomplete
print("Last order date in dataset:", full_data["order_purchase_timestamp"].max())

# looking at the last 3 weeks in more detail to decide a sensible cutoff
print("\nDaily order volume, last 21 days:")
daily_volume = full_data.set_index("order_purchase_timestamp").resample("D")["order_id"].count()
print(daily_volume.tail(21))

#%%

# trimming to the last date with stable, representative volume - excluding the dataset's export-cutoff tail
cutoff_date = pd.Timestamp("2018-08-22")
full_data_trimmed = full_data[full_data["order_purchase_timestamp"] <= cutoff_date].copy()

print("Rows before trimming:", full_data.shape[0])
print("Rows after trimming:", full_data_trimmed.shape[0])
print("Rows removed:", full_data.shape[0] - full_data_trimmed.shape[0])

#%%

# rebuilding the standard 80/20 chronological split on the trimmed (artifact-free) dataset
full_data_trimmed = full_data_trimmed.sort_values("order_purchase_timestamp").reset_index(drop=True)
split_index = int(len(full_data_trimmed) * 0.8)
split_date_trimmed = full_data_trimmed.iloc[split_index]["order_purchase_timestamp"]

train_v2 = full_data_trimmed[full_data_trimmed["order_purchase_timestamp"] < split_date_trimmed].copy()
test_v2 = full_data_trimmed[full_data_trimmed["order_purchase_timestamp"] >= split_date_trimmed].copy()

print("New split date:", split_date_trimmed)
print("Train shape:", train_v2.shape, "| Train late rate:", train_v2["is_late"].mean())
print("Test shape:", test_v2.shape, "| Test late rate:", test_v2["is_late"].mean())

# recomputing seller_on_time_rate fresh on this new split - same leakage-safe logic as before
seller_stats_v2 = train_v2.groupby("primary_seller_id")["is_late"].agg(
    seller_total_orders="count", seller_late_rate="mean"
).reset_index()
seller_stats_v2["seller_on_time_rate"] = 1 - seller_stats_v2["seller_late_rate"]
overall_rate_v2 = 1 - train_v2["is_late"].mean()
seller_stats_v2["has_sufficient_history"] = seller_stats_v2["seller_total_orders"] >= 5
seller_stats_v2.loc[~seller_stats_v2["has_sufficient_history"], "seller_on_time_rate"] = overall_rate_v2

def apply_seller_feature_v2(df):
    df = df.merge(seller_stats_v2[["primary_seller_id", "seller_on_time_rate", "has_sufficient_history"]], on="primary_seller_id", how="left")
    df["seller_on_time_rate"] = df["seller_on_time_rate"].fillna(overall_rate_v2)
    df["is_new_or_thin_history_seller"] = (~df["has_sufficient_history"].fillna(False).astype(bool)).astype(int)
    return df.drop(columns=["has_sufficient_history"])

train_v2 = apply_seller_feature_v2(train_v2)
test_v2 = apply_seller_feature_v2(test_v2)

# re-running the logistic regression baseline on the cleaned split
X_train_v2 = train_v2[baseline_features].copy()
X_test_v2 = test_v2[baseline_features].copy()
y_train_v2 = train_v2["is_late"]
y_test_v2 = test_v2["is_late"]

scaler_v2_model = StandardScaler()
X_train_v2_scaled = scaler_v2_model.fit_transform(X_train_v2)
X_test_v2_scaled = scaler_v2_model.transform(X_test_v2)

log_reg_v2 = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg_v2.fit(X_train_v2_scaled, y_train_v2)

y_pred_proba_v2 = log_reg_v2.predict_proba(X_test_v2_scaled)[:, 1]
print("\nTrain AUC:", roc_auc_score(y_train_v2, log_reg_v2.predict_proba(X_train_v2_scaled)[:, 1]))
print("Test AUC:", roc_auc_score(y_test_v2, y_pred_proba_v2))
#%%

# comparing fold 4 (good AUC, Apr-Jun) vs fold 5 (bad AUC, Jun-Aug) more directly
fold4_window = full_data_trimmed[(full_data_trimmed["order_purchase_timestamp"] >= "2018-04-01") & (full_data_trimmed["order_purchase_timestamp"] < "2018-06-13")]
fold5_window = full_data_trimmed[(full_data_trimmed["order_purchase_timestamp"] >= "2018-06-13") & (full_data_trimmed["order_purchase_timestamp"] <= "2018-08-22")]

print("Fold 4 window - late rate:", fold4_window["is_late"].mean(), "| seller_on_time_rate corr:", pointbiserialr(fold4_window["is_late"], fold4_window.merge(seller_stats_v2[["primary_seller_id","seller_on_time_rate"]], on="primary_seller_id", how="left")["seller_on_time_rate"])[0])
print("Fold 5 window - late rate:", fold5_window["is_late"].mean(), "| seller_on_time_rate corr:", pointbiserialr(fold5_window["is_late"], fold5_window.merge(seller_stats_v2[["primary_seller_id","seller_on_time_rate"]], on="primary_seller_id", how="left")["seller_on_time_rate"])[0])

# checking % of new sellers in each window
print("\nFold 4 % new/thin sellers:", fold4_window["primary_seller_id"].isin(seller_stats_v2[seller_stats_v2["has_sufficient_history"]]["primary_seller_id"]).mean())
print("Fold 5 % new/thin sellers:", fold5_window["primary_seller_id"].isin(seller_stats_v2[seller_stats_v2["has_sufficient_history"]]["primary_seller_id"]).mean())

#%%

# adding explicit fallback fill after merge - sellers not in seller_stats_v2 at all (never seen in training) need the fallback too
fold4_merged = fold4_window.merge(seller_stats_v2[["primary_seller_id","seller_on_time_rate"]], on="primary_seller_id", how="left")
fold4_merged["seller_on_time_rate"] = fold4_merged["seller_on_time_rate"].fillna(overall_rate_v2)

fold5_merged = fold5_window.merge(seller_stats_v2[["primary_seller_id","seller_on_time_rate"]], on="primary_seller_id", how="left")
fold5_merged["seller_on_time_rate"] = fold5_merged["seller_on_time_rate"].fillna(overall_rate_v2)

print("Fold 4 window - late rate:", fold4_window["is_late"].mean(), "| seller_on_time_rate corr:", pointbiserialr(fold4_merged["is_late"], fold4_merged["seller_on_time_rate"])[0])
print("Fold 5 window - late rate:", fold5_window["is_late"].mean(), "| seller_on_time_rate corr:", pointbiserialr(fold5_merged["is_late"], fold5_merged["seller_on_time_rate"])[0])

# checking % of new/thin-history sellers in each window
sufficient_history_sellers = set(seller_stats_v2[seller_stats_v2["has_sufficient_history"]]["primary_seller_id"])
print("\nFold 4 % sellers WITH sufficient training history:", fold4_window["primary_seller_id"].isin(sufficient_history_sellers).mean())
print("Fold 5 % sellers WITH sufficient training history:", fold5_window["primary_seller_id"].isin(sufficient_history_sellers).mean())

#%%

# splitting test_v2 predictions by whether the seller had sufficient training history or not
test_v2_with_proba = test_v2.copy()
test_v2_with_proba["predicted_proba"] = y_pred_proba_v2

sufficient_history_mask = test_v2_with_proba["is_new_or_thin_history_seller"] == 0
thin_history_mask = test_v2_with_proba["is_new_or_thin_history_seller"] == 1

print("Test orders with sufficient seller history:", sufficient_history_mask.sum())
print("Test orders with new/thin seller history:", thin_history_mask.sum())

auc_sufficient = roc_auc_score(
    test_v2_with_proba.loc[sufficient_history_mask, "is_late"],
    test_v2_with_proba.loc[sufficient_history_mask, "predicted_proba"]
)
auc_thin = roc_auc_score(
    test_v2_with_proba.loc[thin_history_mask, "is_late"],
    test_v2_with_proba.loc[thin_history_mask, "predicted_proba"]
)

print(f"\nAUC for sellers WITH sufficient history: {auc_sufficient:.4f}")
print(f"AUC for sellers with NEW/thin history: {auc_thin:.4f}")

print("\nLate rate, sufficient history group:", test_v2_with_proba.loc[sufficient_history_mask, "is_late"].mean())
print("Late rate, thin history group:", test_v2_with_proba.loc[thin_history_mask, "is_late"].mean())

#%%

# comparing every feature's correlation with is_late across fold 4 (good AUC) vs fold 5 (bad AUC)
comparison_features = [f for f in baseline_features if f != "is_new_or_thin_history_seller"]

print(f"{'Feature':<30} {'Fold 4 corr':>12} {'Fold 5 corr':>12}")
for col in comparison_features:
    corr4, _ = pointbiserialr(fold4_window["is_late"], fold4_window[col])
    corr5, _ = pointbiserialr(fold5_window["is_late"], fold5_window[col])
    print(f"{col:<30} {corr4:>12.4f} {corr5:>12.4f}")

# also checking the seller rate feature we already computed, formatted the same way
corr4_seller, _ = pointbiserialr(fold4_merged["is_late"], fold4_merged["seller_on_time_rate"])
corr5_seller, _ = pointbiserialr(fold5_merged["is_late"], fold5_merged["seller_on_time_rate"])
print(f"{'seller_on_time_rate':<30} {corr4_seller:>12.4f} {corr5_seller:>12.4f}")

#%%

# isolating the two features that seem to have caused the break
for col in ["purchase_dow", "purchase_month"]:
    corr4, _ = pointbiserialr(fold4_window["is_late"], fold4_window[col])
    corr5, _ = pointbiserialr(fold5_window["is_late"], fold5_window[col])
    print(f"{col:<30} {corr4:>12.4f} {corr5:>12.4f}")

#%%

# checking basic data integrity first - are fold4_window and fold5_window free of duplicates/issues?
print("Fold 4 window duplicate order_ids:", fold4_window["order_id"].duplicated().sum())
print("Fold 5 window duplicate order_ids:", fold5_window["order_id"].duplicated().sum())
print("Fold 4 window size:", fold4_window.shape[0])
print("Fold 5 window size:", fold5_window.shape[0])

# checking distance distribution - did something change in the distance values themselves, not just their relationship to lateness?
print("\nFold 4 distance stats:\n", fold4_window["seller_customer_distance_km"].describe())
print("\nFold 5 distance stats:\n", fold5_window["seller_customer_distance_km"].describe())

# bootstrap-style check: is the fold 5 distance correlation stable, or does it swing wildly with subsampling?
np.random.seed(42)
boot_corrs = []
for i in range(20):
    sample = fold5_window.sample(frac=0.7, replace=True)
    c, _ = pointbiserialr(sample["is_late"], sample["seller_customer_distance_km"])
    boot_corrs.append(c)
print("\nFold 5 distance correlation across 20 bootstrap samples:")
print("Mean:", np.mean(boot_corrs), "| Std:", np.std(boot_corrs), "| Range:", min(boot_corrs), "to", max(boot_corrs))

#%%

# checking seller overlap between fold 4 and fold 5 - are we largely looking at the same sellers, or a different mix?
fold4_sellers = set(fold4_window["primary_seller_id"])
fold5_sellers = set(fold5_window["primary_seller_id"])
overlap = fold4_sellers & fold5_sellers
print("Sellers in fold 4:", len(fold4_sellers))
print("Sellers in fold 5:", len(fold5_sellers))
print("Sellers in both:", len(overlap))

# checking category mix shift between windows
fold4_cat_dist = fold4_window["primary_category"].value_counts(normalize=True).head(10)
fold5_cat_dist = fold5_window["primary_category"].value_counts(normalize=True).head(10)
print("\nFold 4 top categories:\n", fold4_cat_dist)
print("\nFold 5 top categories:\n", fold5_cat_dist)

# the key test: restricting to ONLY sellers present in both windows, does the distance correlation still flip?
fold4_common = fold4_window[fold4_window["primary_seller_id"].isin(overlap)]
fold5_common = fold5_window[fold5_window["primary_seller_id"].isin(overlap)]

corr4_common, _ = pointbiserialr(fold4_common["is_late"], fold4_common["seller_customer_distance_km"])
corr5_common, _ = pointbiserialr(fold5_common["is_late"], fold5_common["seller_customer_distance_km"])
print(f"\nDistance correlation, SAME sellers only - Fold 4: {corr4_common:.4f} | Fold 5: {corr5_common:.4f}")
print("Orders in this restricted comparison - Fold 4:", fold4_common.shape[0], "| Fold 5:", fold5_common.shape[0])

   #%%
   
import numpy as np

# breaking delivery into its 3 legs using timestamps already in the dataset
for df_name, df in [("fold4_window", fold4_window), ("fold5_window", fold5_window)]:
    df["order_approved_at"] = pd.to_datetime(df["order_approved_at"])
    df["order_delivered_carrier_date"] = pd.to_datetime(df["order_delivered_carrier_date"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])

    # stage 1: purchase -> approval (how fast payment/order gets approved)
    df["stage1_purchase_to_approval_hrs"] = (df["order_approved_at"] - df["order_purchase_timestamp"]).dt.total_seconds() / 3600
    # stage 2: approval -> carrier handoff (how fast seller ships it)
    df["stage2_approval_to_carrier_hrs"] = (df["order_delivered_carrier_date"] - df["order_approved_at"]).dt.total_seconds() / 3600
    # stage 3: carrier -> customer (actual transit/delivery time - most likely tied to distance)
    df["stage3_carrier_to_customer_hrs"] = (df["order_delivered_customer_date"] - df["order_delivered_carrier_date"]).dt.total_seconds() / 3600

# comparing average time per stage between the two windows
for stage in ["stage1_purchase_to_approval_hrs", "stage2_approval_to_carrier_hrs", "stage3_carrier_to_customer_hrs"]:
    print(f"{stage} - Fold 4 mean: {fold4_window[stage].mean():.1f} hrs | Fold 5 mean: {fold5_window[stage].mean():.1f} hrs")

# now checking distance's correlation with EACH stage separately, not just overall lateness
from scipy.stats import pearsonr
for stage in ["stage1_purchase_to_approval_hrs", "stage2_approval_to_carrier_hrs", "stage3_carrier_to_customer_hrs"]:
    valid4 = fold4_window.dropna(subset=[stage, "seller_customer_distance_km"])
    valid5 = fold5_window.dropna(subset=[stage, "seller_customer_distance_km"])
    corr4, _ = pearsonr(valid4[stage], valid4["seller_customer_distance_km"])
    corr5, _ = pearsonr(valid5[stage], valid5["seller_customer_distance_km"])
    print(f"\nDistance vs {stage} - Fold 4: {corr4:.4f} | Fold 5: {corr5:.4f}")
#%%

# checking if the ESTIMATED delivery window (buffer) itself correlates differently with distance between windows
for df in [fold4_window, fold5_window]:
    df["estimated_delivery_window_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400

corr4_est, _ = pearsonr(fold4_window["estimated_delivery_window_days"], fold4_window["seller_customer_distance_km"])
corr5_est, _ = pearsonr(fold5_window["estimated_delivery_window_days"], fold5_window["seller_customer_distance_km"])
print(f"Distance vs estimated delivery window - Fold 4: {corr4_est:.4f} | Fold 5: {corr5_est:.4f}")

print("\nFold 4 avg estimated window (days):", fold4_window["estimated_delivery_window_days"].mean())
print("Fold 5 avg estimated window (days):", fold5_window["estimated_delivery_window_days"].mean())    
    
 #%%

# calculating actual "slack" - the buffer between promised delivery and actual delivery, directly
for df in [fold4_window, fold5_window]:
    df["actual_total_delivery_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.total_seconds() / 86400
    df["slack_days"] = df["estimated_delivery_window_days"] - df["actual_total_delivery_days"]
    # slack < 0 means late (arrived after estimate) - should match is_late exactly, good sanity check too

# sanity check: does slack_days < 0 match our is_late flag?
print("Fold 4 - slack<0 matches is_late:", ((fold4_window["slack_days"] < 0) == fold4_window["is_late"]).mean())
print("Fold 5 - slack<0 matches is_late:", ((fold5_window["slack_days"] < 0) == fold5_window["is_late"]).mean())

# the real test: how does distance relate to slack (the actual margin) in each window?
corr4_slack, _ = pearsonr(fold4_window["slack_days"], fold4_window["seller_customer_distance_km"])
corr5_slack, _ = pearsonr(fold5_window["slack_days"], fold5_window["seller_customer_distance_km"])
print(f"\nDistance vs slack_days - Fold 4: {corr4_slack:.4f} | Fold 5: {corr5_slack:.4f}")

print("\nFold 4 avg slack (days):", fold4_window["slack_days"].mean())
print("Fold 5 avg slack (days):", fold5_window["slack_days"].mean())   
    
#%%

# checking for NaNs before computing correlation - likely culprit for the error
print("Fold 4 NaNs in slack_days:", fold4_window["slack_days"].isnull().sum())
print("Fold 5 NaNs in slack_days:", fold5_window["slack_days"].isnull().sum())
print("Fold 4 NaNs in seller_customer_distance_km:", fold4_window["seller_customer_distance_km"].isnull().sum())
print("Fold 5 NaNs in seller_customer_distance_km:", fold5_window["seller_customer_distance_km"].isnull().sum())

# dropping any NaN rows before correlation, to be safe
fold4_clean = fold4_window.dropna(subset=["slack_days", "seller_customer_distance_km"])
fold5_clean = fold5_window.dropna(subset=["slack_days", "seller_customer_distance_km"])

corr4_slack, _ = pearsonr(fold4_clean["slack_days"], fold4_clean["seller_customer_distance_km"])
corr5_slack, _ = pearsonr(fold5_clean["slack_days"], fold5_clean["seller_customer_distance_km"])
print(f"\nDistance vs slack_days - Fold 4: {corr4_slack:.4f} | Fold 5: {corr5_slack:.4f}")

print("\nFold 4 avg slack (days):", fold4_clean["slack_days"].mean())
print("Fold 5 avg slack (days):", fold5_clean["slack_days"].mean())
    
#%%

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# walk-forward validation: expanding training window, testing on each subsequent month
# this simulates realistic deployment - retrain periodically, always predicting forward into unseen time

full_data_trimmed = full_data_trimmed.sort_values("order_purchase_timestamp").reset_index(drop=True)
full_data_trimmed["year_month"] = full_data_trimmed["order_purchase_timestamp"].dt.to_period("M")

# starting the walk-forward from a point where we have enough training history to be meaningful
all_months = sorted(full_data_trimmed["year_month"].unique())
start_idx = 9  # skip the first ~9 months so training always has a reasonable base to learn from

walk_forward_results = []

for i in range(start_idx, len(all_months)):
    test_month = all_months[i]
    train_cutoff = all_months[i]  # train on everything strictly before this month

    wf_train = full_data_trimmed[full_data_trimmed["year_month"] < train_cutoff].copy()
    wf_test = full_data_trimmed[full_data_trimmed["year_month"] == test_month].copy()

    if wf_test["is_late"].sum() < 5 or len(wf_test) < 50:
        continue  # skip months with too few orders or too few late cases to evaluate meaningfully

    # recomputing seller_on_time_rate fresh for this fold - leakage-safe
    seller_stats_wf = wf_train.groupby("primary_seller_id")["is_late"].agg(
        seller_total_orders="count", seller_late_rate="mean"
    ).reset_index()
    seller_stats_wf["seller_on_time_rate"] = 1 - seller_stats_wf["seller_late_rate"]
    overall_rate_wf = 1 - wf_train["is_late"].mean()
    seller_stats_wf["has_sufficient_history"] = seller_stats_wf["seller_total_orders"] >= 5
    seller_stats_wf.loc[~seller_stats_wf["has_sufficient_history"], "seller_on_time_rate"] = overall_rate_wf

    def apply_seller_wf(df):
        df = df.merge(seller_stats_wf[["primary_seller_id", "seller_on_time_rate"]], on="primary_seller_id", how="left")
        df["seller_on_time_rate"] = df["seller_on_time_rate"].fillna(overall_rate_wf)
        return df

    wf_train = apply_seller_wf(wf_train)
    wf_test = apply_seller_wf(wf_test)

    feats = [f for f in baseline_features if f != "is_new_or_thin_history_seller"] + ["seller_on_time_rate"]
    feats = list(dict.fromkeys(feats))  # dedupe in case seller_on_time_rate already there

    Xtr, ytr = wf_train[feats], wf_train["is_late"]
    Xte, yte = wf_test[feats], wf_test["is_late"]

    scaler_wf = StandardScaler()
    Xtr_scaled = scaler_wf.fit_transform(Xtr)
    Xte_scaled = scaler_wf.transform(Xte)

    model_wf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model_wf.fit(Xtr_scaled, ytr)
    proba = model_wf.predict_proba(Xte_scaled)[:, 1]
    auc = roc_auc_score(yte, proba)

    walk_forward_results.append({
        "test_month": str(test_month), "train_size": len(wf_train), "test_size": len(wf_test),
        "test_late_rate": yte.mean(), "auc": auc
    })
    print(f"Test month {test_month}: train_size={len(wf_train)}, test_late_rate={yte.mean():.3f}, AUC={auc:.4f}")

wf_results_df = pd.DataFrame(walk_forward_results)
print("\nOverall walk-forward AUC - mean:", wf_results_df["auc"].mean(), "| std:", wf_results_df["auc"].std())
print(wf_results_df)    
    
  #%%

from sklearn.ensemble import RandomForestClassifier

# same walk-forward structure, now with Random Forest and category included
walk_forward_results_rf = []

# one-hot encoding category once on the full trimmed dataset - consistent encoding across all folds
category_dummies = pd.get_dummies(full_data_trimmed["primary_category"], prefix="cat")
full_data_with_cat = pd.concat([full_data_trimmed, category_dummies], axis=1)
category_cols = category_dummies.columns.tolist()

for i in range(start_idx, len(all_months)):
    test_month = all_months[i]
    train_cutoff = all_months[i]

    wf_train = full_data_with_cat[full_data_with_cat["year_month"] < train_cutoff].copy()
    wf_test = full_data_with_cat[full_data_with_cat["year_month"] == test_month].copy()

    if wf_test["is_late"].sum() < 5 or len(wf_test) < 50:
        continue

    seller_stats_wf = wf_train.groupby("primary_seller_id")["is_late"].agg(
        seller_total_orders="count", seller_late_rate="mean"
    ).reset_index()
    seller_stats_wf["seller_on_time_rate"] = 1 - seller_stats_wf["seller_late_rate"]
    overall_rate_wf = 1 - wf_train["is_late"].mean()
    seller_stats_wf["has_sufficient_history"] = seller_stats_wf["seller_total_orders"] >= 5
    seller_stats_wf.loc[~seller_stats_wf["has_sufficient_history"], "seller_on_time_rate"] = overall_rate_wf

    def apply_seller_wf(df):
        df = df.merge(seller_stats_wf[["primary_seller_id", "seller_on_time_rate"]], on="primary_seller_id", how="left")
        df["seller_on_time_rate"] = df["seller_on_time_rate"].fillna(overall_rate_wf)
        return df

    wf_train = apply_seller_wf(wf_train)
    wf_test = apply_seller_wf(wf_test)

    feats_rf = [f for f in baseline_features if f != "is_new_or_thin_history_seller"]
    feats_rf = list(dict.fromkeys(feats_rf + ["seller_on_time_rate"])) + category_cols

    Xtr, ytr = wf_train[feats_rf], wf_train["is_late"]
    Xte, yte = wf_test[feats_rf], wf_test["is_late"]

    # trees don't need scaling - using class_weight="balanced" for the same imbalance handling
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_model.fit(Xtr, ytr)
    proba = rf_model.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)

    walk_forward_results_rf.append({"test_month": str(test_month), "test_late_rate": yte.mean(), "auc": auc})
    print(f"Test month {test_month}: test_late_rate={yte.mean():.3f}, AUC={auc:.4f}")

wf_results_rf_df = pd.DataFrame(walk_forward_results_rf)
print("\nRandom Forest walk-forward AUC - mean:", wf_results_rf_df["auc"].mean(), "| std:", wf_results_rf_df["auc"].std())
print("\nLogistic Regression comparison - mean:", wf_results_df["auc"].mean(), "| std:", wf_results_df["auc"].std())  
    
 #%%
# building seller-level historical stage-duration features, leakage-safe (training data only per fold)
# using the walk-forward structure we already have, extended with 2 new features per seller:
# - avg approval-to-carrier speed (how fast this seller typically ships once approved)
# - avg carrier-to-customer speed (how fast this seller's shipments typically transit)

def compute_seller_stage_features(train_df):
    train_df = train_df.copy()
    train_df["order_approved_at"] = pd.to_datetime(train_df["order_approved_at"])
    train_df["order_delivered_carrier_date"] = pd.to_datetime(train_df["order_delivered_carrier_date"])
    train_df["order_delivered_customer_date"] = pd.to_datetime(train_df["order_delivered_customer_date"])

    train_df["stage2_hrs"] = (train_df["order_delivered_carrier_date"] - train_df["order_approved_at"]).dt.total_seconds() / 3600
    train_df["stage3_hrs"] = (train_df["order_delivered_customer_date"] - train_df["order_delivered_carrier_date"]).dt.total_seconds() / 3600

    seller_stage_stats = train_df.groupby("primary_seller_id").agg(
        seller_avg_stage2_hrs=("stage2_hrs", "mean"),
        seller_avg_stage3_hrs=("stage3_hrs", "mean"),
        seller_stage_order_count=("stage2_hrs", "count")
    ).reset_index()

    return seller_stage_stats

# quick test on the full training set (train_v2) before wiring into the walk-forward loop
seller_stage_stats_test = compute_seller_stage_features(train_v2)
print(seller_stage_stats_test.describe())
print("\nMissing/null stage stats:", seller_stage_stats_test.isnull().sum())   
    
#%%

# investigating negative stage durations - shouldn't be physically possible, worth understanding before trusting this feature
train_v2_temp = train_v2.copy()
train_v2_temp["order_approved_at"] = pd.to_datetime(train_v2_temp["order_approved_at"])
train_v2_temp["order_delivered_carrier_date"] = pd.to_datetime(train_v2_temp["order_delivered_carrier_date"])
train_v2_temp["order_delivered_customer_date"] = pd.to_datetime(train_v2_temp["order_delivered_customer_date"])

train_v2_temp["stage2_hrs"] = (train_v2_temp["order_delivered_carrier_date"] - train_v2_temp["order_approved_at"]).dt.total_seconds() / 3600
train_v2_temp["stage3_hrs"] = (train_v2_temp["order_delivered_customer_date"] - train_v2_temp["order_delivered_carrier_date"]).dt.total_seconds() / 3600

negative_stage2 = train_v2_temp[train_v2_temp["stage2_hrs"] < 0]
negative_stage3 = train_v2_temp[train_v2_temp["stage3_hrs"] < 0]

print("Orders with negative stage2 (approval after carrier handoff):", negative_stage2.shape[0])
print("Orders with negative stage3 (carrier handoff after customer delivery):", negative_stage3.shape[0])

print("\nSample negative stage2 orders:")
print(negative_stage2[["order_id", "order_approved_at", "order_delivered_carrier_date", "stage2_hrs"]].head())

print("\nSample negative stage3 orders:")
print(negative_stage3[["order_id", "order_delivered_carrier_date", "order_delivered_customer_date", "stage3_hrs"]].head())    
    
#%%

# excluding orders with impossible stage sequencing before computing seller stage averages - 
# these are data logging errors, not real negative durations, and would distort seller-level averages
train_v2_temp_clean = train_v2_temp[(train_v2_temp["stage2_hrs"] >= 0) & (train_v2_temp["stage3_hrs"] >= 0)].copy()

print("Rows before excluding impossible sequencing:", train_v2_temp.shape[0])
print("Rows after:", train_v2_temp_clean.shape[0])
print("Rows excluded:", train_v2_temp.shape[0] - train_v2_temp_clean.shape[0])

# rebuilding seller stage stats on the cleaned data
seller_stage_stats_clean = train_v2_temp_clean.groupby("primary_seller_id").agg(
    seller_avg_stage2_hrs=("stage2_hrs", "mean"),
    seller_avg_stage3_hrs=("stage3_hrs", "mean"),
    seller_stage_order_count=("stage2_hrs", "count")
).reset_index()

print("\n", seller_stage_stats_clean.describe())    
    
    #%%
# extending the walk-forward loop with the 2 new stage-based features, added alongside seller_on_time_rate

def compute_seller_stage_features_clean(train_df):
    df = train_df.copy()
    df["order_approved_at"] = pd.to_datetime(df["order_approved_at"])
    df["order_delivered_carrier_date"] = pd.to_datetime(df["order_delivered_carrier_date"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["stage2_hrs"] = (df["order_delivered_carrier_date"] - df["order_approved_at"]).dt.total_seconds() / 3600
    df["stage3_hrs"] = (df["order_delivered_customer_date"] - df["order_delivered_carrier_date"]).dt.total_seconds() / 3600
    # excluding impossible sequencing before computing seller averages
    df_clean = df[(df["stage2_hrs"] >= 0) & (df["stage3_hrs"] >= 0)]
    stats = df_clean.groupby("primary_seller_id").agg(
        seller_avg_stage2_hrs=("stage2_hrs", "mean"),
        seller_avg_stage3_hrs=("stage3_hrs", "mean")
    ).reset_index()
    return stats

walk_forward_results_v2 = []

for i in range(start_idx, len(all_months)):
    test_month = all_months[i]
    train_cutoff = all_months[i]

    wf_train = full_data_trimmed[full_data_trimmed["year_month"] < train_cutoff].copy()
    wf_test = full_data_trimmed[full_data_trimmed["year_month"] == test_month].copy()

    if wf_test["is_late"].sum() < 5 or len(wf_test) < 50:
        continue

    # existing seller_on_time_rate
    seller_stats_wf = wf_train.groupby("primary_seller_id")["is_late"].agg(
        seller_total_orders="count", seller_late_rate="mean"
    ).reset_index()
    seller_stats_wf["seller_on_time_rate"] = 1 - seller_stats_wf["seller_late_rate"]
    overall_rate_wf = 1 - wf_train["is_late"].mean()
    seller_stats_wf["has_sufficient_history"] = seller_stats_wf["seller_total_orders"] >= 5
    seller_stats_wf.loc[~seller_stats_wf["has_sufficient_history"], "seller_on_time_rate"] = overall_rate_wf

    # new stage features
    stage_stats_wf = compute_seller_stage_features_clean(wf_train)
    overall_stage2 = stage_stats_wf["seller_avg_stage2_hrs"].mean()
    overall_stage3 = stage_stats_wf["seller_avg_stage3_hrs"].mean()

    def apply_features_wf(df):
        df = df.merge(seller_stats_wf[["primary_seller_id", "seller_on_time_rate"]], on="primary_seller_id", how="left")
        df["seller_on_time_rate"] = df["seller_on_time_rate"].fillna(overall_rate_wf)
        df = df.merge(stage_stats_wf, on="primary_seller_id", how="left")
        df["seller_avg_stage2_hrs"] = df["seller_avg_stage2_hrs"].fillna(overall_stage2)
        df["seller_avg_stage3_hrs"] = df["seller_avg_stage3_hrs"].fillna(overall_stage3)
        return df

    wf_train = apply_features_wf(wf_train)
    wf_test = apply_features_wf(wf_test)

    feats_v2 = [f for f in baseline_features if f != "is_new_or_thin_history_seller"] + ["seller_avg_stage2_hrs", "seller_avg_stage3_hrs"]
    feats_v2 = list(dict.fromkeys(feats_v2))

    Xtr, ytr = wf_train[feats_v2], wf_train["is_late"]
    Xte, yte = wf_test[feats_v2], wf_test["is_late"]

    scaler_wf = StandardScaler()
    Xtr_scaled = scaler_wf.fit_transform(Xtr)
    Xte_scaled = scaler_wf.transform(Xte)

    model_wf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model_wf.fit(Xtr_scaled, ytr)
    proba = model_wf.predict_proba(Xte_scaled)[:, 1]
    auc = roc_auc_score(yte, proba)

    walk_forward_results_v2.append({"test_month": str(test_month), "auc": auc})
    print(f"Test month {test_month}: AUC={auc:.4f}")

wf_results_v2_df = pd.DataFrame(walk_forward_results_v2)
print("\nWith stage features - mean AUC:", wf_results_v2_df["auc"].mean(), "| std:", wf_results_v2_df["auc"].std())
print("Original baseline - mean AUC:", wf_results_df["auc"].mean(), "| std:", wf_results_df["auc"].std())    
    
#%%

from sklearn.metrics import precision_recall_curve

# getting precision/recall at every possible threshold from our baseline model's predictions
precisions, recalls, thresholds = precision_recall_curve(y_test_v2, y_pred_proba_v2)

# building a readable table at a few candidate thresholds
import numpy as np
candidate_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'Flagged Orders':>15}")
for t in candidate_thresholds:
    preds_at_t = (y_pred_proba_v2 >= t).astype(int)
    tp = ((preds_at_t == 1) & (y_test_v2 == 1)).sum()
    fp = ((preds_at_t == 1) & (y_test_v2 == 0)).sum()
    fn = ((preds_at_t == 0) & (y_test_v2 == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    flagged = preds_at_t.sum()
    print(f"{t:>10.1f} {precision:>10.3f} {recall:>10.3f} {flagged:>15}")    
    
 #%%

# merging review_score onto our order-level dataset (full_data_trimmed has is_late + all our engineered features)
reviews_for_causal = dfs["reviews"].groupby("order_id")["review_score"].mean().reset_index()

causal_data = full_data_trimmed.merge(reviews_for_causal, on="order_id", how="left")

print("Total orders:", causal_data.shape[0])
print("Orders with a review score:", causal_data["review_score"].notnull().sum())
print("Missing review_score:", causal_data["review_score"].isnull().sum())

# dropping orders without a review - can't include them in this analysis
causal_data = causal_data.dropna(subset=["review_score"]).copy()
print("\nFinal dataset for causal analysis:", causal_data.shape[0])

# THE NAIVE COMPARISON - no adjustment for confounders yet, just raw group means
naive_comparison = causal_data.groupby("is_late")["review_score"].agg(["mean", "std", "count"])
print("\nNaive comparison - review score by lateness (no confounder adjustment):")
print(naive_comparison)

naive_diff = naive_comparison.loc[True, "mean"] - naive_comparison.loc[False, "mean"]
print(f"\nNaive difference (late - on-time): {naive_diff:.3f}")  

#%%

# building the propensity score model - predicts P(is_late) using only the confounders, not the outcome
from sklearn.linear_model import LogisticRegression

confounders = [
    "seller_customer_distance_km", "avg_price", "total_freight",
    "seller_on_time_rate", "payment_installments", "purchase_month"
]

# category needs encoding - one-hot for the propensity model
category_dummies_causal = pd.get_dummies(causal_data["primary_category"], prefix="cat", drop_first=True)
causal_data_encoded = pd.concat([causal_data, category_dummies_causal], axis=1)

# NOTE: causal_data doesn't have seller_on_time_rate yet (that was built per-fold in Week 2's walk-forward loop)
# need to compute it fresh here, using the FULL dataset this time since we're not doing time-based prediction anymore -
# we're just using it as a control variable, not as a leakage-sensitive predictive feature
seller_rate_full = causal_data.groupby("primary_seller_id")["is_late"].mean().reset_index()
seller_rate_full.columns = ["primary_seller_id", "seller_late_rate_full"]
seller_rate_full["seller_on_time_rate"] = 1 - seller_rate_full["seller_late_rate_full"]
causal_data_encoded = causal_data_encoded.merge(seller_rate_full[["primary_seller_id", "seller_on_time_rate"]], on="primary_seller_id", how="left")

print(causal_data_encoded[["order_id", "seller_on_time_rate"]].head())
print("\nMissing seller_on_time_rate:", causal_data_encoded["seller_on_time_rate"].isnull().sum()) 
    
    #%%
from sklearn.linear_model import LogisticRegression

# building the full feature set for the propensity model: numeric confounders + category dummies
propensity_features = confounders + category_dummies_causal.columns.tolist()

X_propensity = causal_data_encoded[propensity_features].copy()
y_treatment = causal_data_encoded["is_late"]

# checking for any missing values before fitting
print("Missing values in propensity features:\n", X_propensity.isnull().sum().sum())

# scaling numeric features (dummies don't need scaling but won't be harmed by it either)
from sklearn.preprocessing import StandardScaler
scaler_prop = StandardScaler()
X_propensity_scaled = scaler_prop.fit_transform(X_propensity)

# fitting the propensity model - predicts P(is_late) using ONLY the confounders
propensity_model = LogisticRegression(max_iter=1000, random_state=42)
propensity_model.fit(X_propensity_scaled, y_treatment)

# generating propensity scores for every order
causal_data_encoded["propensity_score"] = propensity_model.predict_proba(X_propensity_scaled)[:, 1]

print("\nPropensity score distribution:")
print(causal_data_encoded["propensity_score"].describe())

print("\nPropensity score by actual treatment group:")
print(causal_data_encoded.groupby("is_late")["propensity_score"].describe()) 


#%%

import matplotlib.pyplot as plt

# visualizing propensity score overlap between groups - checks if matching is actually viable
plt.figure(figsize=(8,5))
plt.hist(causal_data_encoded[causal_data_encoded["is_late"]==False]["propensity_score"], bins=50, alpha=0.5, label="On-time", density=True)
plt.hist(causal_data_encoded[causal_data_encoded["is_late"]==True]["propensity_score"], bins=50, alpha=0.5, label="Late", density=True)
plt.xlabel("Propensity Score (P(late))")
plt.ylabel("Density")
plt.title("Propensity Score Overlap: Late vs On-Time Orders")
plt.legend()
plt.show()

# quantifying overlap - % of late orders whose propensity score falls within the on-time group's range
ontime_min = causal_data_encoded[causal_data_encoded["is_late"]==False]["propensity_score"].min()
ontime_max = causal_data_encoded[causal_data_encoded["is_late"]==False]["propensity_score"].max()
late_in_range = causal_data_encoded[
    (causal_data_encoded["is_late"]==True) &
    (causal_data_encoded["propensity_score"] >= ontime_min) &
    (causal_data_encoded["propensity_score"] <= ontime_max)
]
print(f"% of late orders within on-time group's propensity range: {len(late_in_range) / (causal_data_encoded['is_late']==True).sum() * 100:.1f}%")

#%%

from sklearn.neighbors import NearestNeighbors

late_orders = causal_data_encoded[causal_data_encoded["is_late"] == True].copy()
ontime_orders = causal_data_encoded[causal_data_encoded["is_late"] == False].copy()

# nearest-neighbor matching on propensity score - finding each late order's closest on-time counterpart
nn = NearestNeighbors(n_neighbors=1)
nn.fit(ontime_orders[["propensity_score"]])
distances, indices = nn.kneighbors(late_orders[["propensity_score"]])

# building the matched dataset
matched_ontime = ontime_orders.iloc[indices.flatten()].reset_index(drop=True)
late_orders_reset = late_orders.reset_index(drop=True)

# checking match quality - how close were the matches on average?
print("Average propensity score distance between matched pairs:", distances.mean())
print("Max distance:", distances.max())

# the matched comparison - review scores between late orders and their matched on-time counterparts
matched_comparison = pd.DataFrame({
    "late_review_score": late_orders_reset["review_score"],
    "matched_ontime_review_score": matched_ontime["review_score"]
})

matched_diff = matched_comparison["late_review_score"].mean() - matched_comparison["matched_ontime_review_score"].mean()
print(f"\nMatched (causal-adjusted) difference: {matched_diff:.3f}")
print(f"Naive (unadjusted) difference, for comparison: {naive_diff:.3f}") 

 #%%
# checking covariate balance - are matched pairs actually similar on individual confounders, not just propensity score?
balance_check = pd.DataFrame({
    "distance_late": late_orders_reset["seller_customer_distance_km"],
    "distance_matched_ontime": matched_ontime["seller_customer_distance_km"],
    "price_late": late_orders_reset["avg_price"],
    "price_matched_ontime": matched_ontime["avg_price"],
    "freight_late": late_orders_reset["total_freight"],
    "freight_matched_ontime": matched_ontime["total_freight"],
    "seller_rate_late": late_orders_reset["seller_on_time_rate"],
    "seller_rate_matched_ontime": matched_ontime["seller_on_time_rate"],
})

print("Covariate balance after matching (mean values, late vs matched on-time):\n")
for col_base in ["distance", "price", "freight", "seller_rate"]:
    late_mean = balance_check[f"{col_base}_late"].mean()
    ontime_mean = balance_check[f"{col_base}_matched_ontime"].mean()
    pct_diff = abs(late_mean - ontime_mean) / late_mean * 100 if late_mean != 0 else 0
    print(f"{col_base:15} late={late_mean:>10.2f}  matched_ontime={ontime_mean:>10.2f}  % diff={pct_diff:.1f}%")
    
#%%
# caliper matching - only keep matches within a maximum propensity score distance (caliper)
# common rule of thumb: caliper = 0.2 * std of propensity scores
caliper = 0.2 * causal_data_encoded["propensity_score"].std()
print("Caliper threshold:", caliper)

# using the distances we already computed from nearest-neighbor matching
within_caliper = distances.flatten() <= caliper
print(f"\nMatches within caliper: {within_caliper.sum()} out of {len(within_caliper)} ({within_caliper.mean()*100:.1f}%)")

# re-running the comparison using only matches that pass the caliper test
late_orders_caliper = late_orders_reset[within_caliper]
matched_ontime_caliper = matched_ontime[within_caliper]

caliper_diff = late_orders_caliper["review_score"].mean() - matched_ontime_caliper["review_score"].mean()
print(f"\nCaliper-matched difference: {caliper_diff:.3f}")
print(f"Original (unrestricted) matched difference: {matched_diff:.3f}")
print(f"Naive difference, for reference: {naive_diff:.3f}")

#%%

# splitting causal_data_encoded into a few time windows to test stability of the causal effect
causal_data_encoded["order_purchase_timestamp"] = pd.to_datetime(
    full_data_trimmed.set_index("order_id").loc[causal_data_encoded["order_id"], "order_purchase_timestamp"].values
)

# defining 4 roughly equal time windows across the dataset's span
causal_data_encoded = causal_data_encoded.sort_values("order_purchase_timestamp").reset_index(drop=True)
causal_data_encoded["time_window"] = pd.qcut(causal_data_encoded["order_purchase_timestamp"].astype(int), 4, labels=["Q1", "Q2", "Q3", "Q4"])

print(causal_data_encoded.groupby("time_window")["order_purchase_timestamp"].agg(["min", "max", "count"]))

#%%

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

quarterly_results = []

for window in ["Q1", "Q2", "Q3", "Q4"]:
    window_data = causal_data_encoded[causal_data_encoded["time_window"] == window].copy()

    # naive difference for this window
    naive_window = window_data.groupby("is_late")["review_score"].mean()
    naive_diff_window = naive_window[True] - naive_window[False]

    # propensity model for this window only
    X_prop_w = window_data[propensity_features]
    y_treat_w = window_data["is_late"]

    scaler_w = StandardScaler()
    X_prop_w_scaled = scaler_w.fit_transform(X_prop_w)

    prop_model_w = LogisticRegression(max_iter=1000, random_state=42)
    prop_model_w.fit(X_prop_w_scaled, y_treat_w)
    window_data["propensity_score_w"] = prop_model_w.predict_proba(X_prop_w_scaled)[:, 1]

    late_w = window_data[window_data["is_late"] == True]
    ontime_w = window_data[window_data["is_late"] == False]

    # matching within this window
    nn_w = NearestNeighbors(n_neighbors=1)
    nn_w.fit(ontime_w[["propensity_score_w"]])
    dist_w, idx_w = nn_w.kneighbors(late_w[["propensity_score_w"]])

    matched_ontime_w = ontime_w.iloc[idx_w.flatten()].reset_index(drop=True)
    late_w_reset = late_w.reset_index(drop=True)

    matched_diff_w = late_w_reset["review_score"].mean() - matched_ontime_w["review_score"].mean()

    quarterly_results.append({
        "window": window,
        "n_late": len(late_w),
        "n_ontime": len(ontime_w),
        "naive_diff": naive_diff_window,
        "matched_diff": matched_diff_w,
        "avg_match_distance": dist_w.mean()
    })

    print(f"{window}: n_late={len(late_w)}, naive={naive_diff_window:.3f}, matched={matched_diff_w:.3f}, avg_match_dist={dist_w.mean():.5f}")

quarterly_results_df = pd.DataFrame(quarterly_results)
print("\n", quarterly_results_df)

#%%

# building order_delivery_risk.csv - the core Tableau export tying together Week 1 segments, Week 2 risk, Week 3 causal work

# starting from causal_data_encoded, which already has is_late, review_score, and all the order-level features
export_orders = causal_data_encoded[[
    "order_id", "customer_id", "primary_seller_id", "primary_category",
    "seller_customer_distance_km", "avg_price", "total_freight", "item_count",
    "payment_installments", "seller_on_time_rate", "is_late", "review_score",
    "order_purchase_timestamp", "purchase_month"
]].copy()

# pulling in customer_state and seller_state for the regional angle
export_orders = export_orders.merge(dfs["customers"][["customer_id", "customer_state"]], on="customer_id", how="left")
export_orders = export_orders.merge(dfs["sellers"][["seller_id", "seller_state"]], left_on="primary_seller_id", right_on="seller_id", how="left")

# pulling in the customer's cluster/segment assignment from Week 1 (rfm_clustering_ready has customer_unique_id + cluster_final)
# need customer_unique_id to join, since that's what Week 1's clustering used
export_orders = export_orders.merge(dfs["customers"][["customer_id", "customer_unique_id"]], on="customer_id", how="left")
export_orders = export_orders.merge(
    rfm_clustering_ready[["customer_unique_id", "cluster_final"]], on="customer_unique_id", how="left"
)

# mapping cluster numbers to segment names for readability in Tableau
segment_names = {
    0: "Dissatisfied Segment",
    1: "Satisfied Core",
    2: "High-Value Installment Payers",
    3: "Repeat/Multi-Category Buyers",
    4: "Low-Spend High-Freight"
}
export_orders["customer_segment"] = export_orders["cluster_final"].map(segment_names)

print(export_orders.shape)
print(export_orders.isnull().sum())
print(export_orders["customer_segment"].value_counts())

#%%
# checking what columns actually exist in rfm_clustering_ready right now
print(rfm_clustering_ready.columns.tolist())

#%%

# rebuilding the final Week 1 clustering result cleanly, to make sure we're exporting the correct, final version

# recomputing category_diversity with the "unknown" fix (same as our final Week 1 version)
order_items_customers["product_category_name"] = order_items_customers["product_category_name"].fillna("unknown")
category_diversity_final = order_items_customers.groupby("customer_unique_id")["product_category_name"].nunique().reset_index()
category_diversity_final.columns = ["customer_unique_id", "category_diversity"]

rfm_clustering_ready = rfm_clustering_ready.drop(columns=["category_diversity"], errors="ignore").merge(
    category_diversity_final, on="customer_unique_id", how="left"
)

# rebuilding the final feature set and clustering (k=5, no payment type - matches our final Week 1 decision)
final_cluster_features = [
    "recency_days", "monetary_log", "category_diversity",
    "avg_freight_ratio_log", "avg_installments", "avg_review_score"
]

X_final = rfm_clustering_ready[final_cluster_features].copy()
scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final)

kmeans_final_rebuild = KMeans(n_clusters=5, random_state=42, n_init=10)
rfm_clustering_ready["cluster_final"] = kmeans_final_rebuild.fit_predict(X_final_scaled)

print(rfm_clustering_ready["cluster_final"].value_counts())
print("\nCluster profiles:\n", rfm_clustering_ready.groupby("cluster_final")[["recency_days", "monetary", "avg_review_score", "category_diversity", "avg_installments"]].mean())
    

#%%

print(rfm_clustering_ready.groupby("cluster_final")["avg_review_score"].mean())

#%%


# building order_delivery_risk.csv - core Tableau export combining Week 1 segments, Week 2 risk, Week 3 causal work
export_orders = causal_data_encoded[[
    "order_id", "customer_id", "primary_seller_id", "primary_category",
    "seller_customer_distance_km", "avg_price", "total_freight", "item_count",
    "payment_installments", "seller_on_time_rate", "is_late", "review_score",
    "order_purchase_timestamp", "purchase_month"
]].copy()

# pulling in customer_state and seller_state for regional analysis
export_orders = export_orders.merge(dfs["customers"][["customer_id", "customer_state"]], on="customer_id", how="left")
export_orders = export_orders.merge(dfs["sellers"][["seller_id", "seller_state"]], left_on="primary_seller_id", right_on="seller_id", how="left")

# pulling in customer_unique_id, then the confirmed cluster assignment
export_orders = export_orders.merge(dfs["customers"][["customer_id", "customer_unique_id"]], on="customer_id", how="left")
export_orders = export_orders.merge(
    rfm_clustering_ready[["customer_unique_id", "cluster_final"]], on="customer_unique_id", how="left"
)

# mapping confirmed cluster numbers to segment names
segment_names = {
    0: "Dissatisfied Segment",
    1: "Satisfied Core",
    2: "High-Value Installment Payers",
    3: "Repeat/Multi-Category Buyers",
    4: "Low-Spend High-Freight"
}
export_orders["customer_segment"] = export_orders["cluster_final"].map(segment_names)

print(export_orders.shape)
print("\nMissing values:\n", export_orders.isnull().sum())
print("\nSegment distribution in export:\n", export_orders["customer_segment"].value_counts())

#%%

export_orders.to_csv("order_delivery_risk.csv", index=False)
print("Saved order_delivery_risk.csv -", export_orders.shape[0], "rows")

#%%

# building customer_segments.csv - customer-level export for Dashboard 1 (Week 1 story)
export_customers = rfm_clustering_ready[[
    "customer_unique_id", "recency_days", "monetary", "frequency", "is_repeat_customer",
    "category_diversity", "avg_freight_ratio", "avg_installments", "avg_review_score",
    "preferred_payment_type", "cluster_final"
]].copy()

export_customers["customer_segment"] = export_customers["cluster_final"].map(segment_names)

# adding customer state for regional cross-referencing
customer_state_lookup = dfs["customers"][["customer_unique_id", "customer_state"]].drop_duplicates(subset="customer_unique_id")
export_customers = export_customers.merge(customer_state_lookup, on="customer_unique_id", how="left")

print(export_customers.shape)
print(export_customers.isnull().sum())

export_customers.to_csv("customer_segments.csv", index=False)
print("\nSaved customer_segments.csv -", export_customers.shape[0], "rows")

#%%

# building seller_performance.csv - seller-level export
seller_performance = export_orders.groupby("primary_seller_id").agg(
    total_orders=("order_id", "count"),
    late_orders=("is_late", "sum"),
    on_time_rate=("is_late", lambda x: 1 - x.mean()),
    avg_review_score=("review_score", "mean"),
    avg_price=("avg_price", "mean"),
    avg_distance_km=("seller_customer_distance_km", "mean")
).reset_index()

# adding seller state
seller_performance = seller_performance.merge(
    dfs["sellers"][["seller_id", "seller_state"]].rename(columns={"seller_id": "primary_seller_id"}),
    on="primary_seller_id", how="left"
)

print(seller_performance.shape)
print(seller_performance.isnull().sum())
print("\n", seller_performance.describe())

seller_performance.to_csv("seller_performance.csv", index=False)
print("\nSaved seller_performance.csv -", seller_performance.shape[0], "rows")

#%%
# building regional_summary.csv - state-level export for the optional regional/GTM dashboard
regional_summary = export_orders.groupby("customer_state").agg(
    total_orders=("order_id", "count"),
    late_rate=("is_late", "mean"),
    avg_review_score=("review_score", "mean"),
    avg_price=("avg_price", "mean"),
    avg_freight=("total_freight", "mean"),
    avg_distance_km=("seller_customer_distance_km", "mean")
).reset_index()

# adding segment mix per state - % of orders from each customer segment
segment_mix = export_orders.groupby(["customer_state", "customer_segment"]).size().unstack(fill_value=0)
segment_mix_pct = segment_mix.div(segment_mix.sum(axis=1), axis=0) * 100
segment_mix_pct.columns = [f"pct_{col.replace(' ', '_')}" for col in segment_mix_pct.columns]
segment_mix_pct = segment_mix_pct.reset_index()

regional_summary = regional_summary.merge(segment_mix_pct, on="customer_state", how="left")

print(regional_summary.shape)
print(regional_summary.sort_values("total_orders", ascending=False).head(10))

regional_summary.to_csv("regional_summary.csv", index=False)
print("\nSaved regional_summary.csv -", regional_summary.shape[0], "rows")



#%%

import os
print(os.getcwd())
print([f for f in os.listdir() if f.endswith(".csv")])


#%%

import shutil

csv_files = ["seller_performance.csv", "customer_segments.csv", "regional_summary.csv", "order_delivery_risk.csv"]
destination = "/Users/ananyaashahi/Desktop/Ollist/"

for f in csv_files:
    shutil.move(f, destination + f)

print("Moved. Confirming:")
import os
print(os.listdir(destination))

#%%
# building a tiny summary CSV specifically for the naive vs matched causal comparison chart in Tableau
causal_summary = pd.DataFrame({
    "comparison_type": ["Naive", "Naive", "Matched", "Matched"],
    "group": ["On-time", "Late", "On-time", "Late"],
    "avg_review_score": [4.293, 2.558, 4.229, 2.558]
})

print(causal_summary)
causal_summary.to_csv("causal_comparison_summary.csv", index=False)
print("\nSaved causal_comparison_summary.csv")

#%%
import os
print(os.getcwd())
print("causal_comparison_summary.csv" in os.listdir())
#%%
import shutil
shutil.move("causal_comparison_summary.csv", "/Users/ananyaashahi/Desktop/Ollist/causal_comparison_summary.csv")





