import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_antrenare = pd.read_csv('input/Pacienti.csv')
df_apply = pd.read_csv('input/Pacienti_apply.csv')

# 1. Separ variabilele
X = df_antrenare.iloc[:,0:-1]
Y = df_antrenare['DECISION']

# 2. Standardizare date
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_apply = scaler.fit_transform(df_apply)

# 3. Separare date in date de antrenare si testare
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, stratify=Y, test_size=0.3, random_state=42)

# 4. Aplicare model LDA pe setul de date de antrenare
modelLDA = LinearDiscriminantAnalysis()
modelLDA.fit(X_train, Y_train)
# scoruri dicriminante pentru antrenare si testare
X_train_score = modelLDA.transform(X_train)
X_test_score = modelLDA.transform(X_test)
df_train = pd.DataFrame(X_train_score)
df_train['tip'] = 'train'
df_test = pd.DataFrame(X_test_score)
df_test['tip'] = 'test'
df_scoruri = pd.concat([df_train, df_test])
df_scoruri.to_csv('z.csv', index=False)

# 5. Evaluare model = Matrice confuzie prin datele de testare
Y_pred = modelLDA.predict(X_test)
matrice_confuzie = confusion_matrix(Y_test, Y_pred)
df_matrice = pd.DataFrame(matrice_confuzie)
df_matrice.to_csv('matr_conf.csv', index=False)
# acuratete globala
acuratete = accuracy_score(Y_test, Y_pred)
print('\nAcuratete globala: ', acuratete)
# acuratete medie (mai intai calculez acuratetea individuala)
acuratete_individuala = matrice_confuzie.diagonal() / np.sum(matrice_confuzie, axis=1)
acuratete_medie = np.mean(acuratete_individuala)

# Vizualizare axe discriminante
for label in np.unique(Y_train):
    sb.kdeplot(X_train_score[Y_train == label, 0], label=label)
plt.legend()
plt.show()

# 6. Predictie pe setul de date de aplicare
Y_apply_predict = modelLDA.predict(X_apply)
df_apply['Predict'] = Y_apply_predict
print(df_apply)

