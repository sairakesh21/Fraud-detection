from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = FastAPI()

@app.post("/dbscan_anomaly/")
async def dbscan_anomaly(file: UploadFile = File(...)):
    try:
        # ✅ Load CSV
        df = pd.read_csv(file.file)

        # ✅ Strip whitespace from column names
        df.columns = df.columns.str.strip()
        df = df.dropna()

        # ✅ Ensure required columns are present
        pca_features = [col for col in df.columns if "V" in col]
        if 'Amount' not in df.columns or 'Class' not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns like 'Amount' or 'Class'. Columns present: {df.columns.tolist()}"
            )

        # ✅ Features for clustering
        features = pca_features + ['Amount']
        X = df[features]

        # ✅ Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # ✅ DBSCAN Clustering
        dbscan = DBSCAN(eps=2, min_samples=5)
        df['Cluster'] = dbscan.fit_predict(X_scaled)

        # ✅ Cluster summary for fraud
        fraud_labels = df[df['Class'] == 1]['Cluster'].value_counts().to_dict()
        cluster_summary = df.groupby("Cluster").agg({
            "Class": ["count", "sum"]
        }).reset_index()
        cluster_summary.columns = ['Cluster', 'Total_Transactions', 'Fraud_Transactions']
        summary_dict = cluster_summary.to_dict(orient="records")

        # ✅ t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X_scaled)

        df_tsne = pd.DataFrame(X_embedded, columns=['TSNE1', 'TSNE2'])
        df_tsne['Cluster'] = df['Cluster']

        # ✅ t-SNE Plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Cluster', palette='tab10', legend='full')
        plt.title('DBSCAN Clustering of Credit Card Transactions (t-SNE Visualization)')
        plt.xlabel('TSNE Component 1')
        plt.ylabel('TSNE Component 2')
        plt.legend(title='Cluster')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        tsne_img = base64.b64encode(buf.getvalue()).decode('utf-8')

        return {
            "message": "DBSCAN clustering and anomaly detection completed.",
            "fraudulent_transactions_per_cluster": fraud_labels,
            "cluster_summary": summary_dict,
            "tsne_plot": f"data:image/png;base64,{tsne_img}",
            "sample_clusters": df[['V1', 'V2', 'Amount', 'Cluster', 'Class']].head().to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
