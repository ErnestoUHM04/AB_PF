# Cornejo Morales Paola
# Hernández Martínez Ernesto Ulises

import pandas as pd

df = pd.read_csv('data/datos.csv')

for i in range(len(df)):
    momio_L = df.loc[i, 'Momio_L']
    momio_V = df.loc[i, 'Momio_V']
    momio_E = df.loc[i, 'Momio_E']

    if momio_L > 0:
        momio_L_norm = (momio_L / 100) +1
    else:
        momio_L_norm = (100 / momio_L) +1

    if momio_V > 0:
        momio_V_norm = (momio_V / 100) +1
    else:
        momio_V_norm = (100 / momio_V) +1

    if momio_E > 0:
        momio_E_norm = (momio_E / 100) +1
    else:
        momio_E_norm = (100 / momio_E) +1

    df.loc[i, 'Momio_L_norm'] = momio_L_norm
    df.loc[i, 'Momio_V_norm'] = momio_V_norm
    df.loc[i, 'Momio_E_norm'] = momio_E_norm

df.to_csv('data/datos_normalizados.csv', index=False)