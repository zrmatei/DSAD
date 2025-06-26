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

# 1. Standardizare date
scaler = StandardScaler()
data_standard = scaler.fit_transform(df_numeric)
df_data = pd.DataFrame(data_standard, index=df_numeric.index, columns=df_numeric.columns)

# 2. Aplicare FA
nr_variab = len(df_data.columns)
modelFA = FactorAnalyzer(rotation=None, n_factors=nr_variab)
F = modelFA.fit(df_data)

# 3. Scoruri (mai trebuia sa pun in df + scatter pt primele 2 componente)
eticheteFA = ['F' + str(i+1) for i in range(len(df_data.columns))]
df_scoruri = modelFA.transform(data_standard)
print(df_scoruri)

# 4. Variante
variante = modelFA.get_factor_variance()[0]
df_variante = pd.DataFrame(variante, columns = ['Variante'], index=df_data.columns)
print(df_variante)

# 5. Matrice
matrice_corelatii = modelFA.loadings_
df_matrice = pd.DataFrame(matrice_corelatii, columns=eticheteFA, index= df_data.columns)
plt.figure()
sb.heatmap(df_matrice)
plt.show()

# 6. Comunalitati
comunalitati = modelFA.get_communalities()
print(comunalitati)

# 7. Testul KMO si bartlett
kmo = calculate_kmo(df_data)[1]
print(kmo)
bartlett = calculate_bartlett_sphericity(df_data)[1]
print(bartlett)