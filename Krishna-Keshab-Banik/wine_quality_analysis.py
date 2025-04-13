import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# Ensure Visuals directory exists
if not os.path.exists('Visuals'):
    os.makedirs('Visuals')

# Load dataset
try:
    df = pd.read_csv('Data/WineQT.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Oops! File not found. Check the path and try again.")
    exit()

# Handle missing values using mean
if df.isnull().sum().sum() > 0:
    df.fillna(df.mean(), inplace=True)
    print("Missing values handled.")
else:
    print("No missing values detected.")

# Remove unnecessary columns
if 'Id' in df.columns:
    df.drop(columns=['Id'], inplace=True)

# Convert wine quality to binary classification
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0)

# Features and labels
X = df.drop('quality', axis=1)
y = df['quality']

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
print("Decision Tree training complete.")

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print("Random Forest training complete.")

# Evaluation function
def evaluate_model(true_labels, predictions, model_name):
    print(f"{model_name} Metrics:")
    print(f"Accuracy: {accuracy_score(true_labels, predictions):.4f}")
    print(f"Precision: {precision_score(true_labels, predictions):.4f}")
    print(f"Recall: {recall_score(true_labels, predictions):.4f}")
    print(f"F1 Score: {f1_score(true_labels, predictions):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(true_labels, predictions)}\n")

evaluate_model(y_test, dt_preds, "Decision Tree")
evaluate_model(y_test, rf_preds, "Random Forest")

# K-Means Clustering
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)
print("K-Means clustering completed.")

# PCA for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_clusters, cmap='viridis', alpha=0.7)
plt.title('PCA - K-Means Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar(label='Cluster')
plt.savefig('Visuals/PCA_Clustering.png')
print("PCA plot saved!")
plt.show()

# t-SNE for Visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=kmeans_clusters, cmap='plasma', alpha=0.7)
plt.title('t-SNE - K-Means Clustering')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(label='Cluster')
plt.savefig('Visuals/TSNE_Clustering.png')
print("t-SNE plot saved!")
plt.show()
