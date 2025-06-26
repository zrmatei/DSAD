import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

df_location = pd.read_csv('input/LocationQ.csv')
df_numeric = df_location.drop('Judet', axis=1)

# 1. Standardizare date
scaler = StandardScaler()
date_standard = scaler.fit_transform(df_numeric)
#introduc in df
df_standard = pd.DataFrame(date_standard, index=df_numeric.index, columns=df_numeric.columns)
print('\nDF Standard:\n',df_standard)

# 2. Aplicare cluster prin creare dendograma (linkage = asta e matricea ierarhie)
# + punere in df matricea ierarhie
link = linkage(date_standard, method='ward')
dendrogram(link)
# plt.show()
df_link = pd.DataFrame(link, columns=['Cluster1', 'Cluster2', 'Distanta', 'Nr_observatii'])
print(df_link)

# 3. Nr optim de clusteri - metoda Elbow (aici gasesc distantele, dif, pct optime)
distanta = link[:, 2]
diferente = np.diff(distanta, 2)
pct_elbow = np.argmax(diferente)+1
print(pct_elbow)

# 4. Partitionare clusteri optimi - atasare date de cluster
clusters = fcluster(link, pct_elbow, criterion='maxclust')
df_location['cluster'] = clusters
print(df_location)

# 5. Scor SH - calitate clustering
scorSH = silhouette_score(df_standard, clusters)
print(scorSH)
# valori individuale SH
scor_individual = silhouette_samples(df_standard, clusters)
print(scor_individual)
# + plot in caz ca se cere
plt.figure('Scor SH')
plt.bar(range(1, len(scor_individual)+1), scorSH)
plt.title('Scor SH')
# plt.show()

# 6. Dendograma partitie optima
plt.figure()
dendrogram(link, color_threshold=distanta[pct_elbow-1])
plt.show()

# 7. Partitie oarecare (la fel ca la optima, doar am nr fixat de clusteri)
nr_clusteri = 10
clusters_oarecare = fcluster(link, nr_clusteri, criterion='maxclust')
df_location['optimi'] = clusters_oarecare
print(df_location)