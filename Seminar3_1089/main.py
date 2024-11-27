import numpy as np
import pandas as pd

from functii import inlocuire_nan, standardizare, salvare, corelatii_covariante, teste_concordante

pd.set_option("display.max_columns", None)
np.set_printoptions(5, threshold=5000, suppress=True)

set_date = pd.read_csv("FreeLancerT.csv", index_col=1,
                       na_values=[""], keep_default_na=False)
# print(set_date,type(set_date),sep="\n")
variabile = list(set_date.columns)
variabile_numerice = variabile[2:]

x = set_date[variabile_numerice].values
# print(x,type(x),sep="\n")

exista_valori_lipsa = set_date.isna().any().any()
if exista_valori_lipsa:
    inlocuire_nan(x)
# print(x)

x_c = standardizare(x, std=False)
x_std = standardizare(x)
salvare(x_c,set_date.index,variabile_numerice,"data_out/x_c.csv")
salvare(x_std,set_date.index,variabile_numerice,"data_out/x_std.csv")

# Covariante/corelatii
v = np.cov(x,rowvar=False)
r = np.corrcoef(x,rowvar=False)
salvare(v,variabile_numerice,variabile_numerice,"data_out/v.csv")
salvare(r,variabile_numerice,variabile_numerice,"data_out/r.csv")

covariante,corelatii,grupe = corelatii_covariante(x,set_date["Continent"].values)

print(corelatii)

for i in range(len(grupe)):
    salvare(covariante[i], variabile_numerice, variabile_numerice, "data_out/v_" + grupe[i] + ".csv")
    salvare(corelatii[i], variabile_numerice, variabile_numerice, "data_out/r_" + grupe[i] + ".csv")

# Aplicare teste
t = teste_concordante(x)
salvare(t,variabile_numerice, ["Kolmogorov-Smirnov", "Shapiro", "Anderson", "Chi"], "data_out/teste.csv")
print(t)