import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

persoane = pd.DataFrame({
    'varsta': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    'salariu': [3000, 4000, 5000, 5500, 6000, 7000, 8000, 8500, 9000, 9500],
    'nume': ['Dorel', 'Gigel', 'Floricel', 'Ion', 'Maria', 'Ana', 'Vasile', 'Elena', 'Dan', 'Cristina']
})

df_numeric = persoane[['varsta', 'salariu']]
print('\nDF Numeric:\n',df_numeric)

# 1. Standardizare date - transformare date (medie 0, ab standard 1)
scaler = StandardScaler()
date_standard = scaler.fit_transform(df_numeric)
# introducere in df
df_standard = pd.DataFrame(date_standard, index=df_numeric.index, columns=df_numeric.columns)
print('\nDF Standard:\n',df_standard)

# 2. Aplicare cluster prin creare dendograma
def create_dendogram(data):
    methods = ['complete', 'single', 'average', 'ward']
    matr_links= {} # astea sunt toate matricile de legatura

    for method in methods:
        plt.figure('Dendograma')
        link = linkage(data, method=method) # asta e 1 singura
        matr_links[method] = link
        dendrogram(link, truncate_mode='lastp', p=10)
        plt.title(f'Dendograma {method}')
        plt.xlabel('Sample')
        plt.ylabel('Distanta')
        # plt.show()
    return matr_links

matr_links = create_dendogram(df_standard)

# 3. Determinare nr optim de clusteri - Metoda Elbow
# gasesc punctul unde distantele dintre clustere cresc cel mai mult
matr_ward = matr_links['ward'] # doar schimb aici daca folosesc alta metoda
distanta = matr_ward[:,2]
diferente = np.diff(distanta,2)
puncte_elb = np.argmax(diferente) + 1
print('\nNr optim de clustere:', puncte_elb)

# 4. Partitionare = Atasare date de clustere
clusters = fcluster(matr_ward, puncte_elb, criterion='maxclust')
persoane['cluster'] = clusters
print('\nDF cu clusteri\n',persoane)

# 5. Scor Silhouette - pt calitate clustering
scorSH = silhouette_score(df_standard, clusters)
print('\nScor SH:\n',scorSH)
# + valori - in caz ca se cer
valoriSH = silhouette_samples(df_standard, clusters)
print('\nValori SH:\n',valoriSH)
# + plot in caz ca se cere
plt.figure('Scor SH')
plt.bar(range(1, len(valoriSH)+1), valoriSH)
plt.title('Scor SH')
# plt.show()

# 6. Dendograma cu partitia optima (daca se cere)
plt.figure('Dend. partitie optima')
dendrogram(matr_ward, truncate_mode='lastp', p=10, color_threshold=distanta[puncte_elb-1]) # color_thr pt taiere dendograma la punctul stabilit
plt.title(f'Dend. partitie optima')
plt.xlabel('Sample')
plt.ylabel('Distanta')
# plt.show()

# 7. Partitie oarecare (la fel ca partitia normala, doar ca am un nr fixat in plus)
nr_clusters = 10
clusters_oarecare = fcluster(matr_ward, nr_clusters, criterion='maxclust')
persoane['cluster oarecare'] = clusters_oarecare
print('\nDF si cu clusteri oarecare\n',persoane)

# 8. Pot sa aflu si valorile si scorul SH pentru clusterii optimi
# acelasi cod ca la normal, doar ca ma folosesc de clusterul oarecare
scorSH_oarecare = silhouette_score(df_standard, clusters_oarecare)
print('\nScor SH oarecare\n',scorSH_oarecare)

valoriSH_oarecare = silhouette_samples(df_standard, clusters_oarecare)
print('\nValori SH oarecare\n', valoriSH_oarecare)
plt.figure('Scor SH oarecare')
plt.bar(range(1, len(valoriSH_oarecare)+1), valoriSH_oarecare)
plt.title('Scor SH oarecare')
# plt.show()