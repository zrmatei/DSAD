import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA


# 1. Incarcare date
data = pd.read_csv("usa_cars.csv")

# 2. Selectare date pt clustere
numeric_columns = ['price', 'year', 'mileage']
categorical_columns = ['brand', 'state']
data_numeric = data[numeric_columns]
data_categorical = data[categorical_columns]

# 3. Standardizarea datelor numerice
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# 4. Metoda Elbow pt determinare nr optim de clustere
inertia = []
cluster_range = range(1, 11)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    inertia.append(kmeans.inertia_)

# Graficul Elbow
plt.figure(figsize=(8, 5))
plt.plot(cluster_range, inertia, marker='o', linestyle='--')
plt.title('Metoda Elbow')
plt.xlabel('Nr clustere')
plt.ylabel('Inertie')
plt.xticks(cluster_range)
plt.grid()
plt.savefig("outputs/elbow_plot.png")
plt.close()

# 5. Crearea partitie optima
optimal_k = 4
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42)
labels_optimal = kmeans_optimal.fit_predict(data_scaled)

# Salvarea etichetelor clusterelor
data['Cluster'] = labels_optimal
data.to_csv("outputs/clustered_data.csv", index=False)

# Caracteristici medii pentru fiecare cluster
# A. Pentru datele numerice
numeric_data_labels = data_numeric.copy()
numeric_data_labels['Cluster'] = labels_optimal
cluster_means_original = numeric_data_labels.groupby('Cluster').mean()

# B. Pentru datele non numerice (brand)
cd_label = data_categorical.copy()
cd_label['Cluster'] = labels_optimal

pref_brand = cd_label.groupby('Cluster')['brand'].apply(lambda p: p.value_counts().idxmax())
pref_mode = cd_label.groupby('Cluster')['state'].apply(lambda p: p.value_counts().idxmax())
prefs = pd.DataFrame({
    'p_brand': pref_brand,
    'p_state': pref_mode
})

cluster_summary = cluster_means_original.merge(prefs, left_index=True, right_index=True)
cluster_summary.to_csv("outputs/cluster_summary.csv", index=True)

# Transformare date in subset
subset_size = 500
data_sampled = data_scaled[:subset_size]
labels_sampled = kmeans_optimal.fit_predict(data_sampled)
data_categorical_sampled = data_categorical[:subset_size]

# 6. Afisare scor Silhouette
silhouette_avg_sampled = silhouette_score(data_sampled, labels_sampled)
silhouette_vals_sampled = silhouette_samples(data_sampled, labels_sampled)

# Salvare scoruri Silhouette
with open("outputs/silhouette_scores.txt", "w") as f:
    f.write(f"Scorul mediu Silhouette: {silhouette_avg_sampled}\n")

# Grafic Silhouette
plt.figure(figsize=(10, 7))
y_lower = 10
for i in range(optimal_k):
    ith_cluster_silhouette_values = silhouette_vals_sampled[labels_sampled == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / optimal_k)
    plt.fill_betweenx(
        np.arange(y_lower, y_upper),
        0,
        ith_cluster_silhouette_values,
        facecolor=color,
        edgecolor=color,
        alpha=0.7,
    )
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

plt.axvline(x=silhouette_avg_sampled, color="red", linestyle="--")
plt.title("Graficul Silhouette pentru partitia optima (subset)")
plt.xlabel("Coeficientii Silhouette")
plt.ylabel("Etichetele clusterelor")
plt.yticks([])
plt.savefig("outputs/silhouette_plot.png")
plt.close()

# 7. Dendrograma - metoda Ward
# Cu cat se aplica taietura mai sus pe axa verticala, cu atat nr de clusteri scade
linked_sampled = linkage(data_sampled, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked_sampled, truncate_mode="level", p=5)
plt.title("Dendrograma - Clustering Ierarhic (subset)")
plt.xlabel("Esantioane")
plt.ylabel("Distanta")
plt.savefig("outputs/dendrogram.png")
plt.close()

# 8. Reprezentarea pe primele 2 componente principale
pca = PCA(n_components=2)
data_pca_sampled = pca.fit_transform(data_sampled)

plt.figure(figsize=(10, 7))
for i in range(optimal_k):
    plt.scatter(
        data_pca_sampled[labels_sampled == i, 0],
        data_pca_sampled[labels_sampled == i, 1],
        label=f"Cluster {i}",
    )
plt.title("Reprezentarea clusterelor pe primele doua componente principale (subset)")
plt.xlabel("Componenta Principala 1")
plt.ylabel("Componenta Principala 2")
plt.legend()
plt.grid()
plt.savefig("outputs/pca_clusters.png")
plt.close()

# 9. Histograma pentru fiecare cluster
plt.figure(figsize=(12, 10))
for i in range(optimal_k):
    plt.subplot(2, 2, i + 1)
    cluster_data = data_sampled[labels_sampled == i]
    plt.hist(cluster_data, bins=20, alpha=0.7, label=f"Cluster {i}")
    plt.title(f"Histograma pentru Clusterul {i} (subset)")
    plt.xlabel("Caracteristici standardizate")
    plt.ylabel("Frecventa")
    plt.legend()
plt.tight_layout()
plt.savefig("outputs/histograms.png")
plt.close()