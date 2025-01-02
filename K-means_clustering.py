# Importing libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Reading data and creating data frame
data = pd.read_csv('C:/Users/omars/OneDrive/Desktop/Work/Prodigy InfoTech/PRODIGY_ML_02/Mall_Customers.csv')
df = pd.DataFrame(data)

# Taking columns relevant to purchase history
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardizing values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to identify best number of clusters to use
wcss = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
# Elbow method plot    
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

# Best value of k is 5 based on the plot so creating 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plot of the clusters
plt.figure(figsize=(8, 5))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans, cmap='viridis', s=100, alpha=0.7, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()