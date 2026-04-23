import numpy as np
import pandas as pd
import joblib

from sklearn.cluster import DBSCAN

# ✅ Load scaler
scaler = joblib.load("scaler.pkl")

# ✅ Load clustered dataset (from training)
df_clustered = pd.read_csv("creditcard_clustered.csv")

# ✅ Identify fraud-heavy clusters (>10 fraud cases)
fraud_clusters = df_clustered[df_clustered["Class"] == 1]["Cluster"].value_counts()
fraud_clusters = fraud_clusters[fraud_clusters > 10].index.tolist()

print("\nFraud Dominant Clusters:", fraud_clusters)

# ✅ Load new transactions
new_data = pd.read_csv("new_transactions.csv")

# ✅ Feature selection (same as training)
features = [col for col in new_data.columns if "V" in col]
features.append("Amount")

X_new = new_data[features]

# ✅ Scale using SAME scaler
X_new_scaled = scaler.transform(X_new)

# ✅ Apply DBSCAN again (same params)
dbscan = DBSCAN(eps=2, min_samples=5)
new_data["Cluster"] = dbscan.fit_predict(X_new_scaled)

# ⚠️ Note: Cluster numbers may differ from training (DBSCAN limitation)

# ✅ Fraud prediction
new_data["Fraud_Prediction"] = new_data["Cluster"].apply(
    lambda x: "Fraud" if x in fraud_clusters else "Normal"
)

# ✅ Save output
new_data.to_csv("fraud_predictions.csv", index=False)

# ✅ Print sample output
print("\nSample Predictions:\n")
print(new_data[["Amount", "Cluster", "Fraud_Prediction"]].head(10))