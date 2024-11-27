import sys

import numpy as np
import pandas as pd

from functii import nan_replace_t, acp, tabelare_varianta
from grafice import plot_varianta

np.set_printoptions(5,
    threshold=sys.maxsize,
    suppress=True)

set_date = pd.read_csv("data_in/Mortalitate.csv",index_col=0)
valori_lipsa = set_date.isna().any().any()
if valori_lipsa:
    nan_replace_t(set_date)

x = set_date.values

alpha,a,c,r = acp(x)
# print(alpha)

# Analiza variabilitatii
t_varianta = tabelare_varianta(alpha)
t_varianta.to_csv("acp_out/varianta.csv")
plot_varianta(alpha)

