import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load datasets
customers_df = pd.read_csv("Customers.csv")
transactions_df = pd.read_csv("Transactions.csv")
products_df = pd.read_csv("Products.csv")

# Step 2: Aggregate transaction data
# Map product categories to transactions
transactions_df = transactions_df.merge(products_df[['ProductID', 'Category']], on='ProductID', how='left')

# Aggregate transaction data by CustomerID and Category
category_interaction = transactions_df.groupby(['CustomerID', 'Category']).agg({'Quantity': 'sum'}).reset_index()
category_pivot = category_interaction.pivot(index='CustomerID', columns='Category', values='Quantity').fillna(0)

# Step 3: Merge customer profile information
customers_df = customers_df.set_index('CustomerID')
final_df = customers_df.join(category_pivot, how='left').fillna(0)

# One-hot encode categorical variables (e.g., Region, SignupDate)
categorical_cols = ['Region']
if all(col in final_df.columns for col in categorical_cols):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_cols = encoder.fit_transform(final_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols), index=final_df.index)
    final_df = pd.concat([final_df.drop(columns=categorical_cols), encoded_df], axis=1)
final_df.drop(columns=["CustomerName",'SignupDate'],inplace=True)
# Step 4: Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(final_df)

# Step 5: Determine optimal number of clusters using the Elbow Method
inertia = []
k_values = range(2, 11)  # Number of clusters between 2 and 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid()
plt.show()

optimal_k = 4  # Based on Elbow method
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Step 7: Evaluate clustering metrics
db_index = davies_bouldin_score(scaled_data, clusters)
silhouette_avg = silhouette_score(scaled_data, clusters)

print(f"Number of Clusters: {optimal_k}")
print(f"Davies-Bouldin Index: {db_index:.4f}")
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Step 8: Visualize clusters using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='viridis', s=50)
plt.title('Customer Segmentation (PCA Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid()
plt.show()

# Step 9: Save clustering results
final_df['Cluster'] = clusters
final_df.to_csv("Customer_Segmentation.csv", index=True)
