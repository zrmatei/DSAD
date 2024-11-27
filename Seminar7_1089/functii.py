import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def nan_replace_t(t:pd.DataFrame):
    for coloana in t.columns:
        if t[coloana].isna().any():
            if is_numeric_dtype(t[coloana]):
                t.fillna({coloana:t[coloana].mean()},inplace=True)
            else:
                t.fillna({coloana:t[coloana].mode()[0]},inplace=True)

def acp(x:np.ndarray,std=True,nlib=0):
    n,m = x.shape
    # Standardizare/Centrare
    x_ = x - np.mean(x,axis=0)
    if std:
        x_ = x_/np.std(x,axis=0,ddof=nlib)
    r_v = (1/(n-nlib))*x_.T@x_
    valp,vecp = np.linalg.eig(r_v)
    # print(valp)
    k = np.flip( np.argsort(valp))
    # print(k)
    alpha = valp[k]
    a = vecp[:,k]
    c = x_@a
    if std:
        r = a*np.sqrt(alpha)
    else:
        r = np.corrcoef(x_,c,rowvar=False)[:m,m:]
    return alpha,a,c,r

def tabelare_varianta(alpha):
    procent_v = alpha*100/sum(alpha)
    t = pd.DataFrame(
        {
            "Varianta":alpha,
            "Varianta cumulata":np.cumsum(alpha),
            "Procent varianta":procent_v,
            "Procent cumulat":np.cumsum(procent_v)
        },["C"+str(i+1) for i in range(len(alpha))]
    )
    t.index.name = "Componenta"
    return t