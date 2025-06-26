import numpy as np
import pandas as pd




# cerinta 1
df_sal = pd.read_csv("E_NSAL_2008-2021.csv")
coloane_drop = df_sal.iloc[:,1:]
df_cerinta1 = df_sal.copy().drop(columns=coloane_drop)
df_cerinta1['Anul'] = df_sal.iloc[:,1:].idxmax(axis=1)
print(df_cerinta1)

# cerinta 2
df_pop = pd.read_csv('PopulatieLocalitati.csv', usecols=['Siruta', 'Judet', 'Populatie'])
print(df_pop)
df_cerinta2 = df_sal.merge(df_pop, left_on='SIRUTA', right_on='Siruta')
df_cerinta2.drop('Siruta', inplace=True, axis=1)
# 3. Lista anilor
ani = [str(an) for an in range(2008, 2022)]

# 4. Calculează rata ocupării pentru fiecare an
for an in ani:
    df_cerinta2[an] = df_cerinta2[an] / df_cerinta2['Populatie']

# 5. Grupare pe județ și calcul media
df_rata = df_cerinta2.groupby('Judet')[ani].mean()
df_rata['Media'] = df_rata.mean(axis=1)

# 6. Sortare descrescătoare după rata medie
df_rata_sorted = df_rata.sort_values(by='Media', ascending=False)
print(df_rata_sorted)