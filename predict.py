import joblib
import os
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# Load processed data and models
if not os.path.exists('models/df_processed.joblib'):
    print("Error: Datos procesados no encontrados. Ejecuta modelo_liga_mx.py primero.")
    exit()

if not os.path.exists('models/modelo_bin.joblib') or not os.path.exists('models/modelo_multi.joblib'):
    print("Error: Modelos no encontrados. Ejecuta modelo_liga_mx.py primero.")
    exit()

print("Cargando datos y modelos...")
df = joblib.load('models/df_processed.joblib')
modelo_bin = joblib.load('models/modelo_bin.joblib')
modelo_multi = joblib.load('models/modelo_multi.joblib')
print("Cargado exitosamente.")

# Define necessary functions
def puntos(res):
    if res == 'H': return 3,0
    if res == 'A': return 0,3
    return 1,1

def metricas_previas(equipo, fecha, data):
    prev = data[
        ((data.local==equipo)|(data.visitante==equipo)) &
        (data.fecha < fecha)
    ].tail(5)

    if prev.empty:
        return 0,0,0

    pts, gf, gc = 0,0,0
    for _,p in prev.iterrows():
        if p.local == equipo:
            pts += p.pts_local
            gf += p.goles_local
            gc += p.goles_visita
        else:
            pts += p.pts_visita
            gf += p.goles_visita
            gc += p.goles_local

    return pts/15, gf/5, gc/5  # normalizado

def fuerza(equipo):
    partidos = df[(df.local==equipo)|(df.visitante==equipo)]
    if len(partidos)==0:
        return 1
    pts = 0
    for _,p in partidos.iterrows():
        pts += p.pts_local if p.local==equipo else p.pts_visita
    return pts / len(partidos)

features = [
    'diff_racha','diff_gf','diff_gc','diff_fuerza'
]

def predecir(local, visita):
    hoy = pd.Timestamp.today() + pd.Timedelta(days=1)

    rl,gfl,gcl = metricas_previas(local, hoy, df)
    rv,gfv,gcv = metricas_previas(visita, hoy, df)

    fl = fuerza(local)
    fv = fuerza(visita)

    Xp = pd.DataFrame([[
        rl-rv,
        gfl-gfv,
        gcl-gcv,
        fl-fv
    ]], columns=features)

    p_local = modelo_bin.predict_proba(Xp)[0,1]

    p_ev = modelo_multi.predict_proba(Xp)[0]

    p_emp = (1 - p_local) * p_ev[0]
    p_vis = (1 - p_local) * p_ev[1]

    print(f"\n {local} vs {visita}")
    print(f"Local:   {p_local*100:.1f}%")
    print(f"Empate: {p_emp*100:.1f}%")
    print(f"Visita: {p_vis*100:.1f}%")

# Importamos el modelo ya guardado y lo usamos para predecir un partido
# modelo_bin = joblib.load('modelo_bin.joblib')
# modelo_multi = joblib.load('modelo_multi.joblib')

# =============================== Lista de equipos de la Liga MX y sus nombres en el dataset + nombres comunes ===============================
team_patterns = {
    "América": r"américa|america|club america|aguilas|las aguilas|águilas|club águilas|aguilas del america|aguilas del américa",
    "Atlas": r"atlas|atlas fc|rojinegros|los rojinegros|club atlas|rojinegros de atlas|los rojinegros de atlas",
    "Atlético San Luis": r"atlético san luis|atletico san luis|san luis|atl. san luis|san luis potosí|san luis potosi",
    "Atlante": r"atlante|club atlante|los potros|potros de hierba|los potros de hierba",
    "Chiapas": r"chiapas|jaguares|jaguares de chiapas",
    "Cruz Azul": r"cruz azul|la maquina|maquina|la máquina|club cruz azul|la maquina celeste|los cementeros",
    "Dorados de Sinaloa": r"dorados de sinaloa|dorados|sinaloa",
    "Guadalajara": r"guadalajara|chivas|chivas guadalajara|club guadalajara|las chivas|chivas rayadas|chivas rayadas del guadalajara",
    "Juárez": r"juárez|juarez|fc juarez",
    "León": r"león|leon|club leon|panzas verdes|las panzas verdes",
    "Leones Negros": r"leones negros|udeg|universidad de guadalajara",
    "Lobos BUAP": r"lobos buap|lobos|buap",
    "Mazatlán": r"mazatlán|mazatlan|mazatlan fc",
    "Monarcas": r"monarcas|morelia|monarcas morelia",
    "Monterrey": r"monterrey|rayados|club monterrey|los rayados",
    "Necaxa": r"necaxa|rayos|club necaxa|los rayos",
    "Pachuca": r"pachuca|tuzos|club pachuca|los tuzos",
    "Puebla": r"puebla|la franja|club puebla|los camoteros|camoteros",
    "Querétaro": r"querétaro|queretaro|gallos blancos|club queretaro|gallos blancos de queretaro",
    "Santos Laguna": r"santos laguna|guerreros|club santos|los guerreros",
    "Tijuana": r"tijuana|xolos|club tijuana|los xolos|xolos de tijuana",
    "Tigres UANL": r"tigres uanl|tigres|uanl|club tigres|los tigres|tigres de la uanl",
    "Toluca": r"toluca|diablos rojos|deportivo toluca|club toluca|los diablos rojos",
    "UNAM": r"unam|pumas|unam pumas|club unam|los pumas|pumas unam",
    "Veracruz": r"veracruz|los tiburones|tiburones rojos|club veracruz|tiburones",
}

def normalize_team(name):
    name = name.lower().strip()
    for canonical, pattern in team_patterns.items():
        if re.search(pattern, name, re.IGNORECASE):
            return canonical
    return None

# PRUEBA
# predecir("Toluca","Tigres UANL")

# Para la predicción, se le pide al usuario que ingrese los nombres de los dos equipos
while True:
    local_team = input("Ingrese el nombre del equipo local: ")
    local_normalized = normalize_team(local_team) # Normalize the team names
    if local_normalized is not None:
        break

while True:
    away_team = input("Ingrese el nombre del equipo visitante: ")
    away_normalized = normalize_team(away_team)
    if away_normalized is not None:
        break

predecir(local_normalized, away_normalized)