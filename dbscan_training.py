import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

# ✅ Step 1: Load dataset (keep CSV in same folder)
df = pd.read_csv("creditcard.csv")

# ✅ Step 2: Clean data
df.columns = df.columns.str.strip()
df = df.dropna()

# ✅ Step 3: Feature selection
features = [col for col in df.columns if "V" in col]
features.append("Amount")

X = df[features]

# ✅ Step 4: Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Save scaler for inference
joblib.dump(scaler, "scaler.pkl")

# ✅ Step 5: DBSCAN clustering
dbscan = DBSCAN(eps=2, min_samples=5)
df["Cluster"] = dbscan.fit_predict(X_scaled)

# ✅ Step 6: Fraud cluster analysis
fraud_labels = df[df["Class"] == 1]["Cluster"].value_counts()
print("\nFraudulent Transactions per Cluster:\n", fraud_labels)

# ✅ Step 7: Cluster summary
cluster_summary = df.groupby("Cluster").agg({
    "Class": ["count", "sum"]
})
print("\nCluster Summary:\n", cluster_summary)

# ✅ Save clustered dataset
df.to_csv("creditcard_clustered.csv", index=False)

# ✅ Step 8: Create sample new transactions for testing
sample_data = df.sample(100)
sample_data.to_csv("new_transactions.csv", index=False)

# ✅ Step 9: t-SNE Visualization
#tsne = TSNE(n_components=2, random_state=42)
#X_embedded = tsne.fit_transform(X_scaled)

#df_tsne = pd.DataFrame(X_embedded, columns=["TSNE1", "TSNE2"])
#df_tsne["Cluster"] = df["Cluster"]

#plt.figure(figsize=(10, 6))
#sns.scatterplot(data=df_tsne, x="TSNE1", y="TSNE2", hue="Cluster", palette="tab10")
#plt.title("DBSCAN Clustering (t-SNE)")
#plt.show()

from sklearn.decomposition import PCA

# Reduce to 2D using PCA (FAST)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["Cluster"] = df["Cluster"]

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Cluster", palette="tab10", legend='full')
plt.title("DBSCAN Clustering (PCA Visualization)")
plt.show()