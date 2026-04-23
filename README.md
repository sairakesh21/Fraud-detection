🕵️‍♂️ Fraud Detection Using DBSCAN and Cluster Matching
📌 Overview
This project identifies potential fraud in new financial transactions by applying DBSCAN clustering and comparing the results with known fraud-dominant clusters from previously clustered data. The approach combines unsupervised learning and pattern recognition for efficient fraud detection in large-scale credit card datasets.

🎯 Features
📦 Scalable Anomaly Detection – Uses DBSCAN to group new transactions into clusters.
📈 Transferable Insights – Maps new data to previously known fraud-prone clusters.
🧪 Unsupervised Learning – No labels required for new data prediction.
🔄 Reusable Preprocessing – Leverages a pre-trained StandardScaler for consistency.
📁 Result Export – Outputs labeled predictions to CSV for reporting and audit trails.

🧠 Model
💡 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

eps: 2

min_samples: 5

Unsupervised clustering based on density of points in feature space.

📦 Installation
Install required libraries:

bash
Copy
Edit
pip install numpy pandas scikit-learn joblib
🚀 Usage
Step-by-Step Inference Flow
python
Copy
Edit
# 1. Load the trained StandardScaler
scaler = joblib.load("scaler.pkl")

# 2. Load historical clustered dataset (with known fraud clusters)
df_clustered = pd.read_csv("creditcard_clustered.csv")

# 3. Identify fraud-dominant clusters
fraud_clusters = df_clustered[df_clustered["Class"] == 1]["Cluster"].value_counts()
fraud_clusters = fraud_clusters[fraud_clusters > 10].index.tolist()

# 4. Load new incoming transaction data
new_data = pd.read_csv("new_transactions.csv")

# 5. Select relevant features (same as training)
features = [col for col in new_data.columns if "V" in col] + ["Amount"]
X_new = new_data[features]

# 6. Standardize with previously trained scaler
X_new_scaled = scaler.transform(X_new)

# 7. Apply DBSCAN with same hyperparameters
dbscan = DBSCAN(eps=2, min_samples=5)
new_data["Cluster"] = dbscan.fit_predict(X_new_scaled)

# 8. Label as Fraud if in known fraud-dominant clusters
new_data["Fraud_Prediction"] = new_data["Cluster"].apply(
    lambda x: "Fraud" if x in fraud_clusters else "Normal"
)

# 9. Save output
new_data.to_csv("fraud_predictions.csv", index=False)
🔍 Sample Output
Amount	Cluster	Fraud_Prediction
82.50	3	Fraud
120.00	-1	Normal
5.99	2	Fraud
...	...	...
🛠 Applications
💳 Credit Card Monitoring – Pre-screen transactions before flagging to analysts.
🛡 Fraud Investigation – Group suspicious transactions based on learned fraud clusters.
🔍 Behavioral Profiling – Detect abnormal customer behavior patterns over time.

📁 Output Files
fraud_predictions.csv – Predictions labeled as Fraud or Normal.

Uses existing:

scaler.pkl – Pre-trained StandardScaler

creditcard_clustered.csv – Historical DBSCAN clusters with labeled fraud

📌 References
🧪 DBSCAN in scikit-learn

📚 Unsupervised Learning for Fraud Detection