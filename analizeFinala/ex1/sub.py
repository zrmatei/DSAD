import numpy as np
import pandas as pd

df_siruta = pd.read_csv("siruta.csv")
df_ex2 = df_siruta.copy()
print(df_siruta)
# 1. creare index
index = df_siruta['Localitate']
df_siruta.index = index
df_siruta = df_siruta.sort_values(by='SIRUTA', ascending=False)
df_siruta.drop('Localitate', axis=1, inplace=True)
df_siruta['An'] = df_siruta.iloc[:, 1:].idxmax(axis=1)

# 2.
df_ex2['Medie'] = df_ex2.iloc[:, 2:].sum(axis='rows') / 1000
print('\n', df_ex2)
