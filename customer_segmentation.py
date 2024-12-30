import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

#loading the Amazon_Customer_segmentation.csv file into the data frame
df=pd.read_csv('Amazon_Customer_Segmentation.csv')

#printing the information about the dataframe
print(df.info())

#preprocessing the data because k-means clustering only works with numerical data
df['PrimeMember'] = df['PrimeMember'].map({'Yes': 1, 'No': 0})

features = df[['Age', 'AnnualSpend($)', 'PrimeMember']]

#feature scaling to ensure that each feature contributes equally to the algorithm
scaler = StandardScaler()
scaledfeatures = scaler.fit_transform(features)

#applying K-Means Clustering algorithm
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaledfeatures)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='AnnualSpend($)', hue='Cluster', data=df, palette='Set2')
plt.title('Customer Segmentation by Age and Annual Spend')
plt.xlabel('Age')
plt.ylabel('Annual Spend ($)')
plt.legend(title='Cluster', loc='upper left')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y='Age', data=df, palette='Set1')
plt.title('Age Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Age')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x='Cluster', hue='PrimeMember', data=df, palette='Pastel1')
plt.title('Prime Membership Distribution Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaledfeatures)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

print("\nCustomer Segmentation Result:")
print(df[['CustomerID', 'Age', 'AnnualSpend($)', 'PrimeMember', 'Cluster']].head())

#applying DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['Cluster'] = dbscan.fit_predict(scaledfeatures)
df['Cluster'] = df['Cluster'].apply(lambda x: x if x != -1 else 'Noise')

print("\nCustomer Segmentation using DBSCAN:")
print(df[['CustomerID', 'Age', 'AnnualSpend($)', 'PrimeMember', 'Cluster']].head())

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='AnnualSpend($)', hue='Cluster', data=df, palette='Set2')
plt.title('Customer Segmentation by Age and Annual Spend using DBSCAN')
plt.xlabel('Age')
plt.ylabel('Annual Spend ($)')
plt.legend(title='Cluster', loc='upper left')
plt.show()