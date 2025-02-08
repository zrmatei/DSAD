import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


persoane = pd.DataFrame({
    'nume': ['Gigel','Floricel','Dorel'],
    'varsta': [20,25,35],
    'salariul': [3500,5000,7500]
})

# puteam sa folosesc si iloc si prin [], folosesc [] in interior pentru ca preiau multiple coloane
df_numeric = persoane.iloc[:,1:3]
print(df_numeric)

# Transformare scalar
scaler = StandardScaler()
standard_data = scaler.fit_transform(df_numeric)
print('\nDate Standard:',standard_data)

# 1. Transformare in model pca
modelPCA = PCA()
C = modelPCA.fit_transform(standard_data)

# 2. Varianta
variance = modelPCA.explained_variance_ratio_
print('\nVariance:',variance)
# scriere in csv a listei de variante / putea sa ceara doar afisare la consola
with open('variante.csv', 'w') as file:
    for i in range(len(variance)):
        file.write(str(variance[i]) + ",")

# 3. Plot variante
cum_sum = np.cumsum(variance)
plt.figure("Plot variante")
plt.bar(range(1, len(variance)+1), variance)
plt.step(range(1,len(cum_sum)+1), cum_sum)
plt.title("Cumulative Variance")
# plt.show()

# 4. Scoruri prin df
eticheteDF = ["C" + str(i+1) for i in range(len(variance))]
df_scoruri = pd.DataFrame(C, columns=eticheteDF, index=df_numeric.index)
print(df_scoruri)
#plot cu primele 2 componente
plt.figure("Plot Cumulative Variance")
plt.scatter(df_scoruri['C1'], df_scoruri['C2'])
plt.xlabel("C1")
plt.ylabel("C2")
plt.title("Cumulative Variance")
# plt.show()

# 5. Corelatii factoriale - intre datele standardizate si matricea C (transpuse)
# practic intre variabilele originale si componentele principale
matrice_corelatii = np.corrcoef(standard_data.T, C.T)[:len(df_numeric.columns),len(df_numeric.columns):]
print('\nMatrice corelatii:',matrice_corelatii)
# punere in df
df_matrice_cor = pd.DataFrame(matrice_corelatii, index=df_numeric.columns, columns=eticheteDF)
df_matrice_cor.to_csv('Matrice_cor.csv', index=True)
# corelograma
plt.figure('Corelatii')
sb.heatmap(df_matrice_cor)
# plt.show()

# 6. Comunalitati
comunalitati = np.cumsum(matrice_corelatii**2, axis=1)
# punere in df
df_comunalitati = pd.DataFrame(comunalitati, index=df_numeric.columns, columns=eticheteDF)
print('\nComunalitati:',df_comunalitati.head())
# corelograma
plt.figure('Comunalitati')
plt.title('Comunalitati')
sb.heatmap(comunalitati)
# plt.show()

# 7. Contributii
contributii = matrice_corelatii ** 2 / np.sum(matrice_corelatii ** 2, axis=0)
# punere in df
df_contributii = pd.DataFrame(contributii, index=df_numeric.columns, columns=eticheteDF)
print('\nContributii:',df_contributii.head())