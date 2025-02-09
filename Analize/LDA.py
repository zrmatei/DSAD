import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# asta e pt testare
persoane = pd.DataFrame({
    'varsta': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    'salariu': [3000, 4000, 5000, 5500, 6000, 7000, 8000, 8500, 9000, 9500],
    'nume': ['Dorel', 'Gigel', 'Floricel', 'Ion', 'Maria', 'Ana', 'Vasile', 'Elena', 'Dan', 'Cristina'],
    'DECISION': [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]  # Exemplu: 0 = Respins, 1 = Acceptat
})

# asta il folosesc pentru aplicare
persoane_apply = pd.DataFrame({
    'varsta': [22, 28, 34, 41, 47, 53, 59, 63, 67, 72],
    'salariu': [3200, 4200, 5100, 5800, 6400, 7200, 8100, 8700, 9200, 9800],
    'nume': ['Mihai', 'Andrei', 'Ioana', 'Raluca', 'Cristian', 'Simona', 'Gabriel', 'Oana', 'Radu', 'Carmen']
})


df_numeric = persoane[['varsta', 'salariu', 'DECISION']]

# 1. Separare variabile
# DECISION e variabila dependenta (am nevoie de asa ceva)
# varsta si salariul sunt variabile independente
X = df_numeric[['varsta', 'salariu']]
Y = df_numeric['DECISION'] #trebuie sa fie vector, nu DF
print('\n',X)
print('\n',Y)

# 2. Standardizare date - transformare date (medie 0, ab standard 1)
standard = StandardScaler()
X = standard.fit_transform(X)
X_apply = standard.fit_transform(persoane_apply[['varsta', 'salariu']])

# 3. Impartire date in set de antrenare/test - 70% antrenare, 30% testare
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
print(Y_train.value_counts(), Y_test.value_counts())

# 4. Aplicare Analiza Discriminanta (LDA) - se face pe modelul de antrenare
modelLDA = LinearDiscriminantAnalysis()
modelLDA.fit(X_train, Y_train)
# Scoruri discrminante
X_train_score = modelLDA.transform(X_train)
X_test_score = modelLDA.transform(X_test)

# 5. Evaluare model
# - Matricea de confuzie + predictie pe setul de testare
Y_predict = modelLDA.predict(X_test)
matr_confuzie = confusion_matrix(Y_test, Y_predict)
print('\nMatricea de confuzie: \n', matr_confuzie)

# - Acuratetea globala
acuratete = accuracy_score(Y_test, Y_predict)
print('\nAcuratete: ', acuratete)
# - Medie acuratete prin acuratete individuala
acuratete_individuala = matr_confuzie.diagonal() / matr_confuzie.sum(axis=1)
medie_acuratete = np.mean(acuratete_individuala)
print('Media de acuratete: ', medie_acuratete)

# Vizualizare distributii pe axe discriminante
for label in np.unique(Y_train):
    sb.kdeplot(X_train_score[Y_train == label, 0], label=label)
plt.title('Distributie pe axa')
plt.xlabel('Axa discriminanta')
plt.legend()
plt.show()

# 6. Predictie pe setul de date de aplicare
Y_predict_apply = modelLDA.predict(X_apply)
# punere predictie in df
persoane_apply['Predict:'] = Y_predict_apply
print('\n',persoane_apply)