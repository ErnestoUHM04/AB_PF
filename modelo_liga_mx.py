import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
import joblib
import os

# =====================================================
# 1. CARGA Y LIMPIEZA DE DATOS
# =====================================================
print("Cargando datos Liga MX...")
url = "https://www.football-data.co.uk/new/MEX.csv"
df = pd.read_csv(url)

df = df[['Date','Home','Away','HG','AG','Res']]
df.columns = ['fecha','local','visitante','goles_local','goles_visita','res']
df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)
df = df.sort_values('fecha')

# =====================================================
# 2. PUNTOS
# =====================================================
def puntos(res):
    if res == 'H': return 3,0
    if res == 'A': return 0,3
    return 1,1

df[['pts_local','pts_visita']] = df.apply(
    lambda x: pd.Series(puntos(x['res'])), axis=1
)

# =====================================================
# 3. MÉTRICAS HISTÓRICAS (últimos 5 partidos)
# =====================================================
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

print("Calculando historial...")
datos_l = df.apply(lambda r: metricas_previas(r.local, r.fecha, df), axis=1)
datos_v = df.apply(lambda r: metricas_previas(r.visitante, r.fecha, df), axis=1)

df[['racha_l','gf_l','gc_l']] = pd.DataFrame(datos_l.tolist(), index=df.index)
df[['racha_v','gf_v','gc_v']] = pd.DataFrame(datos_v.tolist(), index=df.index)

# =====================================================
# 4. FUERZA HISTÓRICA (promedio de puntos)
# =====================================================
def fuerza(equipo):
    partidos = df[(df.local==equipo)|(df.visitante==equipo)]
    if len(partidos)==0:
        return 1
    pts = 0
    for _,p in partidos.iterrows():
        pts += p.pts_local if p.local==equipo else p.pts_visita
    return pts / len(partidos)

df['fuerza_l'] = df.local.apply(fuerza)
df['fuerza_v'] = df.visitante.apply(fuerza)

# =====================================================
# SAVE PROCESSED DATA
# =====================================================
if not os.path.exists('models/df_processed.joblib'):
    print("Guardando datos procesados...")
    joblib.dump(df, 'models/df_processed.joblib')
    print("Datos procesados guardados.")

# =====================================================
# 5. FEATURES DIFERENCIALES
# =====================================================
df['diff_racha'] = df.racha_l - df.racha_v
df['diff_gf']    = df.gf_l - df.gf_v
df['diff_gc']    = df.gc_l - df.gc_v
df['diff_fuerza']= df.fuerza_l - df.fuerza_v

features = [
    'diff_racha','diff_gf','diff_gc','diff_fuerza'
]

X = df[features]

# =====================================================
# 6. TARGETS
# =====================================================
# Binario: ¿Gana el local?
df['y_bin'] = (df.res == 'H').astype(int)

# Multiclase secundaria: Empate vs Visita
df['y_multi'] = df.res.map({'D':0,'A':1})

# =====================================================
# 7. SPLIT TEMPORAL
# =====================================================
corte = int(len(df)*0.85)

X_train, X_test = X.iloc[:corte], X.iloc[corte:]
y_bin_train, y_bin_test = df.y_bin[:corte], df.y_bin[corte:]
y_multi_train, y_multi_test = df.y_multi[:corte], df.y_multi[corte:]

# =====================================================
# 8. MODELOS
# =====================================================
if os.path.exists('models/modelo_bin.joblib') and os.path.exists('models/modelo_multi.joblib'):
    print("Cargando modelos desde archivos...")
    modelo_bin = joblib.load('models/modelo_bin.joblib')
    modelo_multi = joblib.load('models/modelo_multi.joblib')
    print("Modelos cargados exitosamente.")
else:
    print("Entrenando modelos...")
    
    modelo_bin = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42
    )
    modelo_bin.fit(X_train, y_bin_train)
    
    modelo_multi = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    )
    modelo_multi.fit(
        X_train[y_bin_train==0],
        y_multi_train[y_bin_train==0]
    )
    
    # =====================================================
    # SAVE MODELS
    # =====================================================
    print("Guardando modelos...")
    joblib.dump(modelo_bin, 'models/modelo_bin.joblib')
    joblib.dump(modelo_multi, 'models/modelo_multi.joblib')
    print("Modelos guardados exitosamente.")

# =====================================================
# 9. EVALUACIÓN
# =====================================================
pred_bin = modelo_bin.predict(X_test)
acc = accuracy_score(y_bin_test, pred_bin)
print(f"\nACCURACY BINARIO (Local vs No Local): {acc*100:.2f}%")

# Matriz de Confusión
cm = confusion_matrix(y_bin_test, pred_bin)
ConfusionMatrixDisplay(cm, display_labels=['No Local','Local']).plot()
plt.title("Matriz de Confusión - Binario")
# save figure
plt.savefig('fig/confusion_matrix_binario.png')
plt.show()


# Curva ROC
probs = modelo_bin.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_bin_test, probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("Curva ROC - Gana Local")
plt.legend()
# save figure
plt.savefig('fig/roc_curve_binario.png')
plt.show()


# =====================================================
# 10. PREDICTOR FINAL (este es de prueba) -> Se debería hacer un script propio
# =====================================================
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

# PRUEBA
# predecir("Toluca","Tigres UANL")
