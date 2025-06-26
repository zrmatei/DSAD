import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer, calculate_kmo
from sklearn.preprocessing import StandardScaler


def clean_data(df):
    if df.isna().any().any():
        # tratez coloanele numerice
        numeric = df.select_dtypes(include=np.number).columns
        df[numeric] = df[numeric].fillna(df[numeric].mean())
        # coloanele non-numerice
        objects = df.select_dtypes(exclude=np.number).columns
        for col in objects:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

df_caen = pd.read_csv('CAEN2_2021_NSAL.csv')
df_pop = pd.read_csv('PopulatieLocalitati.csv')
# cerinta 1
df_merged = df_caen.merge(df_pop, left_on='SIRUTA', right_on='Siruta')
df_sters = df_merged.drop(['Siruta', 'Localitate', 'Judet'], axis=1)
print(df_sters)
# df_sters.iloc[:, 1:-1] = df_sters.apply(lambda x: ((x.iloc[1:-1] / x['Populatie']) * 100).astype(np.float64), axis=1)


# cerinta 2
print(df_merged)
df_sters_2 = df_merged.drop(['Siruta', 'Localitate', 'SIRUTA'], axis=1)
df_grupat = df_sters_2.groupby('Judet').sum()
print(df_grupat)
df_ex2 = df_grupat.div(df_grupat['Populatie'], axis=0) * 10000
df_ex2.drop(columns='Populatie', inplace=True)
print(df_ex2)


df_numeric = clean_data(df_caen)
# Standardizare date
scaler = StandardScaler()
date_standard = scaler.fit_transform(df_numeric)
df_date = pd.DataFrame(date_standard, index=df_numeric.index, columns=df_numeric.columns)
# Aplicare model FA
nr_variabile = len(df_date.columns)
modelFA = FactorAnalyzer(rotation=None, n_factors=nr_variabile)
F = modelFA.fit(df_date)

#kmo
test_kmo = calculate_kmo(df_date)[0]
print(test_kmo)

# df_scoruri factoriale
eticheteFA = ['F' + str(i+1) for i in range(len(df_date.columns))]
scoruri = modelFA.transform(df_date)
df_scoruri = pd.DataFrame(scoruri, index=df_date.index, columns=eticheteFA)
df_scoruri.to_csv('df_scoruri.csv',index=False)

# plot pentru primele 2
plt.figure()
plt.scatter(df_scoruri['F1'], df_scoruri['F2'])
plt.xlabel('F1')
plt.ylabel('F2')
plt.show()