import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from factor_analyzer import FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo
from sklearn.preprocessing import StandardScaler


persoane = pd.DataFrame({
    'varsta': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    'salariu': [3000, 4000, 5000, 5500, 6000, 7000, 8000, 8500, 9000, 9500],
    'nume': ['Dorel', 'Gigel', 'Floricel', 'Ion', 'Maria', 'Ana', 'Vasile', 'Elena', 'Dan', 'Cristina']
})

df_numeric = persoane[['varsta', 'salariu']]
print('\nDF Numeric:',df_numeric)

# 1. Standardizare date - transformare in date standard (medie 0, ab standard 1)
scaler = StandardScaler()
date_standard = scaler.fit_transform(df_numeric)
# transformare in df
df_standard = pd.DataFrame(date_standard, index=df_numeric.index, columns=df_numeric.columns)
print('\nDF Standardizat:\n',df_standard)

# 2. Aplicare FA pe datele standard
nr_variabile = len(df_standard.columns)
modelFA = FactorAnalyzer(n_factors=nr_variabile, rotation=None)
F = modelFA.fit(df_standard)

# 3. Scoruri prin transformare in DF + scatter la final
scoruri = modelFA.transform(df_standard)
eticheteFA = ['F' + str(i+1) for i in range(nr_variabile)]
# df
df_scoruri = pd.DataFrame(scoruri, index=df_standard.index, columns=eticheteFA)
print('\nDF Scoruri:\n',df_scoruri)
# scatter
plt.figure(figsize=(6,6))
plt.title('Scatter Scoruri')
plt.scatter(df_scoruri['F1'], df_scoruri['F2'])
plt.xlabel('F1')
plt.ylabel('F2')
# plt.show()

# 4. Testele Bartlett si KMO (pe datele standard)
b_test = calculate_bartlett_sphericity(df_standard)[1]
print('\nBartlett Test:\n',b_test)
# vezi documentatia functiei ca scrie acolo, la examen nu cred ca poti sa vezi documentatia daca nu ai internet
kmo = calculate_kmo(df_standard)[1] #am pus 1 pentru ca iau doar valoarea totala, puteam sa pun 0 si luam fiecare valoare, depinde ce mi se cere;
print('\nKMO Test Total:\n',kmo)

# 5. Variantele
variance = modelFA.get_factor_variance()[0]
print('\nVarianta:\n',variance)

# 6. Corelatii factoriale
matrice_corelatii = modelFA.loadings_
print('\nMatrice Corelatii:\n',matrice_corelatii)
# punere in df
df_corelatii = pd.DataFrame(matrice_corelatii, index=df_numeric.columns, columns=eticheteFA)
print('\nDF Corelatii:\n',df_corelatii)
# corelograma
plt.figure(figsize=(6,6))
sb.heatmap(df_corelatii, annot=True)
plt.title('Corelatii')
# plt.show()

# 7. Comunalitati
comunalitati = modelFA.get_communalities()
print('\nComunalitati:\n',comunalitati)
# df
# aici la columns pun eu o singura eticheta pentru ca la comunalitati am nevoie de 1 singura coloana, altfel primesc eroare cu dim invalida
df_comun = pd.DataFrame(comunalitati, index=df_numeric.columns, columns=['Comunalitati'])
print('\nDF Comunalitati:\n',df_comun)
# corelograma
plt.figure(figsize=(6,6))
sb.heatmap(df_comun, annot=True, vmin=0)
plt.title('Comunalitati')
plt.show()