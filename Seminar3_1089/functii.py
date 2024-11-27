import numpy as np
import pandas
import pandas as pd
from scipy.stats import kstest,shapiro,anderson,norm,chi2
from pandas.api.types import is_numeric_dtype



def inlocuire_nan(x):
    is_nan = np.isnan(x)
    # print(is_nan)
    k = np.where(is_nan)
    # print(k)
    x[k] = np.nanmean(x[:, k[1]], axis=0)


def standardizare(x, std=True, ddof=0):
    x_ = x - np.mean(x, axis=0)
    if std:
        x_ = x_ / np.std(x, axis=0, ddof=ddof)
    return x_

def salvare(x,nume_linii=None,nume_coloane=None,nume_fisier="out.csv"):
    t = pd.DataFrame(x,nume_linii,nume_coloane)
    t.to_csv(nume_fisier)

def corelatii_covariante(x:np.ndarray,g):
    grupe = np.unique(g)
    q = len(grupe)
    m = x.shape[1]
    covariante = np.empty(shape=(q,m,m))
    corelatii = np.empty(shape=(q,m,m))
    for k in range(q):
        y = x[g == grupe[k],:]
        covariante[k] = np.cov(y,rowvar=False)
        corelatii[k] = np.corrcoef(y,rowvar=False)
    return covariante,corelatii,grupe

def teste_concordante(x:np.ndarray):
    m = x.shape[1] #nr linii, nr coloane - valori
    t = np.empty(shape=(m, 4), dtype=bool)
    for j in range(m):
        t_ks = kstest(x[:,j], "norm")
        print(t_ks)
        t[j, 0] = t_ks[1] > 0.01
        t_shapiro  = shapiro(x[:,j])
        print(t_shapiro)
        print("\n")
        t[j, 1] = t_shapiro[1] > 0.01

        t_anderson = anderson(x[:,j])
        t[j, 2] = any(t_anderson[0] < t_anderson[1])
        print(t_anderson)
        t_chi2 = test_chi2(x[:,j])
        #print(t_chi2)
        t[j, 3]
    return t

#formula suma(fj - fli)^2 / fej
def test_chi2(x): # Sturges formula - m = log2n + 1
    n = len(x)
    medie_x = np.mean(x)
    std_x = np.std(x)
    f,l = np.histogram(x, bins="sturges")
    m = len(f)
    p = norm.cdf(l, medie_x, std_x) # cdf = cumullative distribution function
    fe = n * (p[1:] - p[:m])
    fe[fe == 0] = 1 #produce un vector cu un nr egal de elemente
    stts = np.sum((f-fe)**2/fe)
    p_value = chi2.cdf(stts, m-1) #probabilitatea ca in distributia pi^2 sa am valori mai mici decat statistica
    return stts,p_value

def nan_replace_t(t:pd.DataFrame):
    for coloana in t.columns:
        if t[coloana].isna().any():
            if is_numeric_dtype(t[coloana]):
                t[coloana].fillna(t[coloana].mean(), inplace=True) #inplace intoarce seria modificata
            else:
                t[coloana].fillna(t[coloana].mode()[0], inplace=True)

def calcul_categorie_dominanta(t:pd.Series):
    #print(t.argmax())
    return t.index[t.argmax()]

def calcul_disparitate(t:pd.DataFrame):
    #print(t)
    x = t.values
    sx = np.sum(x, axis=0)
    sx[sx == 0] = 1
    p = x/sx
    p[p==0] = 1
    h = -np.sum(p * np.log2(p), axis=0)
    return pandas.Series(h, index=t.columns)

