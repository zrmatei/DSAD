import pandas as pd

from functii import *

prezenta_vot = pd.read_csv('tema3_in/prezenta_vot.csv', index_col = 0)
exista_nan = prezenta_vot.isna().any().any()
if exista_nan:
    nan_replace_t(prezenta_vot)

variabile = list(prezenta_vot)
variabile_vot = variabile[3:] #de la 3 in continuare

#cerinta1
procent_participare = prezenta_vot["LT"] * 100 / (prezenta_vot["Votanti_LP"]) + prezenta_vot["Votanti_LS"]
procent_participare.name = "Procent participare"
t_procent_participare = pd.DataFrame(procent_participare)
t_procent_participare.insert(0, "localitate", prezenta_vot["Localitate"])

#print(t_procent_participare)
cerinta1 = t_procent_participare[t_procent_participare["Procent participare"] > 50]
cerinta1.to_csv("tema3_out/Prezenta50.csv")

#cerinta2
cerinta2 = t_procent_participare.sort_values(by="Procent participare", ascending=False)
cerinta2.to_csv("tema3_out/PrezentaSort.csv")

#cerinta3
coduri_judete = pd.read_csv("tema3_in/Coduri_judete.csv", index_col =0)
prezenta_vot_ = prezenta_vot.merge(coduri_judete, left_on="Judet", right_index=True) #dreapta e considerat coduri_judete
#print(prezenta_vot_)
cerinta3 = (prezenta_vot_[variabile_vot + ["Regiune"]].groupby(by="Regiune").sum())
cerinta3.to_csv("tema3_out/Regiuni.csv")

#cerinta4
varabile_structura = variabile[variabile.index("Barbati_18-24"):]
print(varabile_structura)

#folosesc apply (cu 2 parametrii,
# in parametrul func specific functia, iar in axis pentru axa pe care o aplic: fie coloana, fie linie)
#daca am valoarea 1 - pe linie, altfel 0
categorie_dominanta = prezenta_vot[varabile_structura].apply(
    func=calcul_categorie_dominanta,
    axis=1)
categorie_dominanta.name = "Categorie dominanta"
#print(categorie_dominanta)

cerinta4 = pd.DataFrame(categorie_dominanta)
cerinta4.insert(0, "Localitate", prezenta_vot["Localitate"])
cerinta4.to_csv("tema3_out/Varsta.csv")

#cerinta5
categorie = "Femei_45-64"
cerinta5 = cerinta4[cerinta4["Categorie dominanta"] == categorie]
cerinta5.to_csv("tema3_out/" + categorie + ".csv")

#cerinta6
cerinta6 = prezenta_vot[varabile_structura + ["Judet"]].groupby(by="Judet").apply(func=calcul_disparitate, include_groups=False)
cerinta6.to_csv("tema3_out/Disparitate.csv")