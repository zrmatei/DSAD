import numpy as np
import pandas as pd

def f_cerinta5(t:pd.DataFrame):
    # print(t)
    v = np.average(t.values[:,:-1],weights=t.values[:,-1],axis=0)
    return pd.Series(v,t.columns[:-1])

agricultura = pd.read_csv("data_in/Agricultura.csv", index_col=0)
activitati = list(agricultura)[1:]

cerinta1 = agricultura.apply(func=lambda x: x["PlanteNepermanente"] + x["PlanteInmultire"], axis=1)
cerinta1.name = "PlanteNep_PlanteInm"
cerinta1_final = pd.DataFrame(cerinta1)
cerinta1_final.insert(0, "Localitate", agricultura["Localitate"])
cerinta1_final.sort_values(by="PlanteNep_PlanteInm", ascending=False, inplace=True)
cerinta1_final.to_csv("data_out/Cerinta1.csv")

cerinta1_ = agricultura.apply(
    func=lambda x: pd.Series(
        [x["Localitate"], x["PlanteNepermanente"] + x["PlanteInmultire"]],
        ["Localitate", "PlanteNep_PlanteInm"]
    ),axis=1)
cerinta1_.sort_values(by="PlanteNep_PlanteInm",
                      ascending=False, inplace=True)
cerinta1_.to_csv("data_out/Cerinta1_.csv")

cerinta2 = pd.DataFrame()
cerinta2["Localitate"] = agricultura["Localitate"]
cerinta2["CA_Totala"] = agricultura.iloc[:,1:].sum(axis=1)
cerinta2.to_csv("data_out/Cerinta2.csv")

populatie = pd.read_csv("data_in/PopulatieLocalitati.csv",index_col=0)
cerinta3_ = cerinta2.merge(populatie[["Judet","Populatie"]],left_index=True,
                           right_index=True)
cerinta3 = pd.DataFrame(cerinta3_["Localitate"])
cerinta3["CA_capita"] = cerinta3_["CA_Totala"]/cerinta3_["Populatie"]
cerinta3.to_csv("data_out/Cerinta3.csv")

agricultura_ = agricultura.merge(populatie[["Judet","Populatie"]],
                                 left_index=True,right_index=True)
cerinta4 = agricultura_[activitati+["Judet"]].groupby(by="Judet").sum()
cerinta4.to_csv("data_out/Cerinta4.csv")

# print(agricultura_)
ro_nuts = pd.read_csv("data_in/RO_NUTS.csv",index_col=0)
agricultura_reg = agricultura_.merge(ro_nuts["Regiune"],
                                     left_on="Judet",right_index=True)
# print(agricultura_reg)
cerinta5 = agricultura_reg[activitati+["Populatie","Regiune"]].groupby(by="Regiune").\
    apply(func=f_cerinta5,include_groups=False)
cerinta5.to_csv("data_out/Cerinta5.csv")
