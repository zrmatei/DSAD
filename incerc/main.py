import numpy as np
import pandas as pd
from functii import *

#cerinta1
pers_analf = pd.read_csv("Educatie.csv",index_col=0,usecols=[0,6])
sum_pers = pers_analf.iloc[1:].sum()['PersoaneAnalfabete']
pers_analf['PondereAnalfabetism'] =  pers_analf['PersoaneAnalfabete'] / sum_pers
ordonare = pers_analf.sort_values(by='PondereAnalfabetism',ascending=False)
ordonare_select = ordonare[['PondereAnalfabetism']]
ordonare_select.to_csv("Cerinta1.csv")

#cerinta2
pers_analf = pd.read_csv("Educatie.csv",index_col=0,usecols=[0,5,6])
pers_analf['FS+A'] = (pers_analf['FaraScoala'] + pers_analf['PersoaneAnalfabete']) / 100000
ordonare = pers_analf.sort_values(by='FS+A',ascending=False)
#pun ordonar[['Judet', 'FS+A]] ca preiau doar cele 2 si sa le copiez in ordonare_final
ordonare_select = ordonare[['FS+A']]
ordonare_select.to_csv("Cerinta2.csv")

#cerinta3
educatie = pd.read_csv("Educatie.csv",index_col=0,usecols=[0,1,2,3,4,5,6])
ro_nuts = pd.read_csv("RO_NUTS.csv",index_col=0,usecols=[0,2])
tabela_join = educatie.join(ro_nuts,how='inner')
tabela_noua = tabela_join.groupby(['Regiune']).sum()
tabela_noua.to_csv("Cerinta3.csv")

# #cerinta4 (det pt regiune si nivel de educatie, judetul cu cea mai mare pondere)
# educatie = pd.read_csv("Educatie.csv",index_col=0,usecols=[0,1,2,3,4,5,6])
# ro_nuts = pd.read_csv("RO_NUTS.csv",index_col=0,usecols=[0,2])
# tabela_join = educatie.join(ro_nuts,how='inner')
# niveluri = pd.read_csv("Educatie.csv",index_col=0,usecols=[1,2,3,4,5,6])
# rezultat = tabela_join.groupby('Regiune')[niveluri].apply(lambda grup: grup.idxmax())
# rezultat = rezultat.rename(lambda x: tabela_join.loc[x, 'Judet'], axis=1)

#cerinta5(nr mediu pers pe niveluri de ed la nivel de regiune)
educatie = pd.read_csv("Educatie.csv",index_col=0,usecols=[0,1,2,3,4,5,6])
ro_nuts = pd.read_csv("RO_NUTS.csv",index_col=0,usecols=[0,2])
tabela_join = educatie.join(ro_nuts,how='inner')
niveluri = ['Superior', 'Liceal', 'Gimnazial', 'Primar', 'FaraScoala', 'PersoaneAnalfabete']
medie = tabela_join.groupby('Regiune')[niveluri].mean()
medie.to_csv("medie.csv")