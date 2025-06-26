import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("input/E_NSAL_2008-2021.csv")


def clean_data(df):
    if df.isna().any().any():
        # verific pt numere
        numeric = df.select_dtypes(include=np.number).columns
        df[numeric] = df[numeric].fillna(df[numeric].mean())
        # verific pt objects
        objects = df.select_dtypes(exclude=np.number).columns
        for col in objects:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

df_clean = clean_data(df)
df_clean.drop('SIRUTA', axis=1, inplace=True)

# 1. Standardizare date
scaler = StandardScaler()
data_standard = scaler.fit_transform(df_clean)

# 2. Aplicare model PCA
modelPCA = PCA()
C = modelPCA.fit_transform(data_standard)

# 3. Variante
variante = modelPCA.explained_variance_ratio_
print(variante)

# 4. afisare df_scoruri cu etichete
etichetePCA = ['C' + str(i+1) for i in range(len(df_clean.columns))]
df_scoruri = pd.DataFrame(C, columns=etichetePCA, index=df_clean.index)
print(df_scoruri)
# afisare primele 2 sub forma de scatter
plt.figure()
plt.scatter(df_scoruri['C1'], df_scoruri['C2'])
plt.xlabel('c1')
plt.ylabel('c2')
# plt.show()

# 5. Matrice de corelatii prin valorile initiale si componentele
matrice_cor = np.corrcoef(data_standard.T, C.T)[:len(df_clean.columns), len(df_clean.columns):]
print(matrice_cor)
df_matrice = pd.DataFrame(matrice_cor, index=etichetePCA, columns=df_clean.columns)
# heatmap - corelograma
plt.figure()
sb.heatmap(df_matrice, annot=False)
plt.show()

# 6. Comunalitati
comunalitati = np.cumsum(matrice_cor ** 2, axis=1)
print(comunalitati)

# 7. Contributii
contributii = matrice_cor ** 2 / np.sum(matrice_cor ** 2, axis=0)
print(contributii)