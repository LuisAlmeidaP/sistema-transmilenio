"""
entrenar.py
===========
Entrena y evalúa los modelos de aprendizaje supervisado que predicen el nivel
de congestión de una estación (clasificación en 3 clases: baja, media, alta).

Modelos comparados:
  1. Línea base (DummyClassifier): siempre predice la clase más frecuente.
     Cualquier modelo útil tiene que superarla.
  2. Árbol de decisión con entropía (estilo ID3/C4.5, Palma 2008 cap. 17).
     Sus hiperparámetros (profundidad y tamaño mínimo de hoja, que actúan
     como PODA) se eligen con validación cruzada agrupada por día.
  3. Random Forest: muchos árboles sobre muestras distintas, votan la clase.
  4. Regresión logística: modelo lineal, como contraste con los árboles.

Salidas en aprendizaje/resultados/ y el modelo final en aprendizaje/modelos/.

Uso:  python -m aprendizaje.entrenar

H( s ) = - Σ p(x) log2 p(x)  (entropía de Shannon, medida de incertidumbre)
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402
from sklearn.dummy import DummyClassifier                    # noqa: E402
from sklearn.ensemble import RandomForestClassifier          # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import GridSearchCV, GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline                        # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree   # noqa: E402

from aprendizaje.preparar_datos import (CLASES, PREDICTORAS, cargar,  # noqa: E402
                                        dividir_temporal, preprocesador)

DIR = os.path.dirname(os.path.abspath(__file__))
DIR_RES = os.path.join(DIR, "resultados")
DIR_MOD = os.path.join(DIR, "modelos")
RUTA_MODELO = os.path.join(DIR_MOD, "arbol_congestion.joblib")
SEMILLA = 42
COLORES = {"baja": "#2a9d8f", "media": "#e9c46a", "alta": "#d7263d"}


# ------------------------------------------------------------------ modelos
def modelos():
    arbol = Pipeline([("prep", preprocesador()),
                      ("modelo", DecisionTreeClassifier(criterion="entropy", random_state=SEMILLA))])
    return {
        "Línea base (clase más frecuente)": Pipeline([("prep", preprocesador()),
                                                      ("modelo", DummyClassifier(strategy="most_frequent"))]),
        "Árbol de decisión (entropía)": arbol,
        "Random Forest (200 árboles)": Pipeline([("prep", preprocesador()),
                                                 ("modelo", RandomForestClassifier(
                                                     n_estimators=200, min_samples_leaf=5,
                                                     n_jobs=-1, random_state=SEMILLA))]),
        "Regresión logística": Pipeline([("prep", preprocesador(escalar=True)),
                                         ("modelo", LogisticRegression(max_iter=3000))]),
    }


REJILLA_ARBOL = {"modelo__max_depth": [3, 4, 6, 8, 10, 12, 16, None],
                 "modelo__min_samples_leaf": [1, 10, 30, 60]}


TOLERANCIA = 0.01   # se acepta perder hasta 0,01 de F1 a cambio de un árbol más simple


def elegir_arbol_simple(cv_results):
    """Poda por complejidad: entre las combinaciones cuyo F1 de validación está a
    menos de TOLERANCIA del mejor, elige el árbol MÁS SIMPLE (menor profundidad y,
    a igual profundidad, hojas más grandes). Un árbol pequeño generaliza igual
    y sus reglas se pueden leer."""
    r = pd.DataFrame(cv_results)
    r["prof"] = r["param_modelo__max_depth"].fillna(1000).astype(float)
    r["hoja"] = r["param_modelo__min_samples_leaf"].astype(float)
    candidatos = r[r["mean_test_score"] >= r["mean_test_score"].max() - TOLERANCIA]
    return int(candidatos.sort_values(["prof", "hoja"], ascending=[True, False]).index[0])


def ajustar_arbol(pipe, X, y, grupos):
    """Validación cruzada de 4 particiones agrupadas por FECHA: un mismo día
    nunca queda a la vez en entrenamiento y validación."""
    busq = GridSearchCV(pipe, REJILLA_ARBOL, scoring="f1_macro", cv=GroupKFold(n_splits=4),
                        n_jobs=-1, return_train_score=True, refit=elegir_arbol_simple)
    busq.fit(X, y, groups=grupos)
    return busq


def evaluar(y_real, y_pred):
    return {
        "exactitud": accuracy_score(y_real, y_pred),
        "f1_macro": f1_score(y_real, y_pred, average="macro"),
        "recall_alta": recall_score(y_real, y_pred, labels=["alta"], average="macro", zero_division=0),
    }


# ------------------------------------------------------- reglas del árbol
def _condicion(nombre, umbral, izquierda):
    """Traduce un corte del árbol a lenguaje natural."""
    for cat in ("tipo_estacion_", "troncal_"):
        if nombre.startswith(cat):
            var, val = cat[:-1], nombre[len(cat):]
            return f"{var} {'≠' if izquierda else '='} {val}"
    if nombre in ("es_fin_semana", "es_festivo", "evento_masivo"):
        return f"{nombre} = {'no' if izquierda else 'sí'}"
    if nombre == "hora":
        return f"hora {'<' if izquierda else '≥'} {int(np.floor(umbral)) + 1}:00"
    return f"{nombre} {'≤' if izquierda else '>'} {umbral:.2f}"


def extraer_reglas(pipe):
    """Recorre el árbol y devuelve una regla SI–ENTONCES por hoja, con su
    soporte (ejemplos de entrenamiento que cubre) y confianza (% de acierto)."""
    arbol = pipe.named_steps["modelo"].tree_
    nombres = pipe.named_steps["prep"].get_feature_names_out()
    clases = list(pipe.named_steps["modelo"].classes_)
    reglas = []

    def recorrer(nodo, condiciones):
        if arbol.children_left[nodo] == -1:            # hoja
            dist = arbol.value[nodo][0]
            dist = dist / dist.sum()
            k = int(np.argmax(dist))
            reglas.append({"condiciones": _simplificar(condiciones), "clase": clases[k],
                           "confianza": float(dist[k]), "soporte": int(arbol.n_node_samples[nodo])})
            return
        f, u = nombres[arbol.feature[nodo]], arbol.threshold[nodo]
        recorrer(arbol.children_left[nodo], condiciones + [(f, u, True)])
        recorrer(arbol.children_right[nodo], condiciones + [(f, u, False)])

    recorrer(0, [])
    return reglas


def _simplificar(condiciones):
    """Une cortes repetidos sobre la misma variable numérica (p. ej. hora ≥ 6 y hora ≥ 7 → hora ≥ 7)."""
    limites, otros = {}, []
    for f, u, izq in condiciones:
        if f.startswith(("tipo_estacion_", "troncal_")) or f in ("es_fin_semana", "es_festivo", "evento_masivo"):
            texto = _condicion(f, u, izq)
            if texto not in otros:
                otros.append(texto)
            continue
        lo, hi = limites.get(f, (-np.inf, np.inf))
        limites[f] = (lo, min(hi, u)) if izq else (max(lo, u), hi)
    salida = list(otros)
    for f, (lo, hi) in limites.items():
        if f == "hora":
            a = int(np.floor(lo)) + 1 if lo > -np.inf else None
            b = int(np.floor(hi)) if hi < np.inf else None
            if a is not None and b is not None:
                salida.append(f"hora entre {a}:00 y {b}:00" if a != b else f"hora = {a}:00")
            elif a is not None:
                salida.append(f"hora ≥ {a}:00")
            else:
                salida.append(f"hora ≤ {b}:00")
        elif f == "dia_semana":
            dias = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
            a = int(np.floor(lo)) + 1 if lo > -np.inf else 0
            b = int(np.floor(hi)) if hi < np.inf else 6
            salida.append(f"día entre {dias[a]} y {dias[b]}" if a != b else f"día = {dias[a]}")
        else:
            if lo > -np.inf:
                salida.append(f"{f} > {lo:.2f}")
            if hi < np.inf:
                salida.append(f"{f} ≤ {hi:.2f}")
    return salida


def formatear_regla(r, i=None):
    pre = f"Regla {i}: " if i is not None else ""
    return (f"{pre}SI " + " ∧ ".join(r["condiciones"]) +
            f"  ENTONCES congestión = {r['clase'].upper()}"
            f"   [soporte = {r['soporte']}, confianza = {r['confianza']:.0%}]")


# ----------------------------------------------------------------- figuras
def graficar_matriz(cm, archivo, titulo):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASES)
    ax.set_yticks(range(3), CLASES)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_title(titulo, fontsize=11)
    fig.tight_layout()
    fig.savefig(archivo, dpi=140)
    plt.close(fig)


def graficar_importancia(pipe, archivo, top=12):
    imp = pd.Series(pipe.named_steps["modelo"].feature_importances_,
                    index=pipe.named_steps["prep"].get_feature_names_out()).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(imp.index, imp.values, color="#1d70b8")
    ax.set_xlabel("Importancia (reducción total de entropía)")
    ax.set_title("Variables más importantes del árbol de decisión", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(archivo, dpi=140)
    plt.close(fig)


def graficar_arbol(pipe, archivo, niveles=3):
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(pipe.named_steps["modelo"], max_depth=niveles, ax=ax, filled=True, rounded=True,
              feature_names=list(pipe.named_steps["prep"].get_feature_names_out()),
              class_names=list(pipe.named_steps["modelo"].classes_), fontsize=8, impurity=True,
              proportion=True)
    ax.set_title(f"Primeros {niveles} niveles del árbol de decisión (criterio: entropía)", fontsize=13)
    fig.tight_layout()
    fig.savefig(archivo, dpi=120)
    plt.close(fig)


def graficar_curva_profundidad(X_ent, y_ent, X_pru, y_pru, archivo):
    """Sobreajuste: el árbol sin poda memoriza el entrenamiento y empeora en prueba."""
    profundidades = list(range(1, 21))
    ent, pru = [], []
    for p in profundidades:
        m = Pipeline([("prep", preprocesador()),
                      ("modelo", DecisionTreeClassifier(criterion="entropy", max_depth=p,
                                                        random_state=SEMILLA))]).fit(X_ent, y_ent)
        ent.append(f1_score(y_ent, m.predict(X_ent), average="macro"))
        pru.append(f1_score(y_pru, m.predict(X_pru), average="macro"))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(profundidades, ent, "o-", label="Entrenamiento", color="#6a4c93")
    ax.plot(profundidades, pru, "s-", label="Prueba (semana 4)", color="#d7263d")
    ax.set_xlabel("Profundidad máxima del árbol")
    ax.set_ylabel("F1 macro")
    ax.set_title("Profundidad del árbol vs. desempeño: aparece el sobreajuste", fontsize=11)
    ax.set_xticks(profundidades)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(archivo, dpi=140)
    plt.close(fig)
    return dict(zip(profundidades, zip(ent, pru)))


# --------------------------------------------------------------- principal
def main():
    os.makedirs(DIR_RES, exist_ok=True)
    os.makedirs(DIR_MOD, exist_ok=True)
    t0 = time.time()
    df = cargar()
    X_ent, X_pru, y_ent, y_pru, grupos = dividir_temporal(df)
    print(f"Datos: {len(df):,} registros | entrenamiento {len(X_ent):,} (semanas 1–3) | "
          f"prueba {len(X_pru):,} (semana 4)")
    print(f"Variables predictoras ({len(PREDICTORAS)}): {', '.join(PREDICTORAS)}\n")

    resultados, entrenados = [], {}
    for nombre, pipe in modelos().items():
        t = time.time()
        if nombre.startswith("Árbol"):
            busq = ajustar_arbol(pipe, X_ent, y_ent, grupos)
            pipe = busq.best_estimator_
            mejores = {k.replace("modelo__", ""): v for k, v in busq.best_params_.items()}
            cv = busq.cv_results_["mean_test_score"]
            print(f"  Validación cruzada del árbol (32 combinaciones × 4 particiones por día):\n"
                  f"    mejor F1 macro posible = {cv.max():.3f}; árbol más simple dentro de "
                  f"{TOLERANCIA} → {mejores} con F1 = {cv[busq.best_index_]:.3f}")
        else:
            pipe.fit(X_ent, y_ent)
        y_hat = pipe.predict(X_pru)
        m = evaluar(y_pru, y_hat)
        m["f1_entrenamiento"] = f1_score(y_ent, pipe.predict(X_ent), average="macro")
        m["segundos"] = time.time() - t
        resultados.append({"modelo": nombre, **m})
        entrenados[nombre] = pipe

    tabla = pd.DataFrame(resultados).set_index("modelo")
    tabla.round(4).to_csv(os.path.join(DIR_RES, "comparacion_modelos.csv"), encoding="utf-8")
    print("\nComparación en el conjunto de PRUEBA (semana 4):")
    print(f"{'Modelo':<36}{'Exactitud':>10}{'F1 macro':>10}{'Recall alta':>13}{'F1 entren.':>12}")
    print("-" * 81)
    for nombre, r in tabla.iterrows():
        print(f"{nombre:<36}{r.exactitud:>10.3f}{r.f1_macro:>10.3f}{r.recall_alta:>13.3f}"
              f"{r.f1_entrenamiento:>12.3f}")

    # ---- análisis del árbol (modelo principal de la actividad)
    arbol = entrenados["Árbol de decisión (entropía)"]
    y_hat = arbol.predict(X_pru)
    cm = confusion_matrix(y_pru, y_hat, labels=CLASES)
    reporte = classification_report(y_pru, y_hat, labels=CLASES, digits=3)
    print("\nÁrbol de decisión — reporte por clase (prueba):")
    print(reporte)
    print("Matriz de confusión (filas = real, columnas = predicho):", CLASES)
    print(cm)

    est = arbol.named_steps["modelo"]
    reglas = extraer_reglas(arbol)
    reglas_alta = sorted([r for r in reglas if r["clase"] == "alta"], key=lambda r: -r["soporte"])
    print(f"\nEl árbol tiene profundidad {est.get_depth()} y {est.get_n_leaves()} hojas "
          f"(= {len(reglas)} reglas SI–ENTONCES).")
    print("Las 5 reglas con más soporte que predicen congestión ALTA:")
    for i, r in enumerate(reglas_alta[:5], 1):
        print("  " + formatear_regla(r, i))

    with open(os.path.join(DIR_RES, "reglas_arbol.txt"), "w", encoding="utf-8") as f:
        f.write(f"Reglas extraídas del árbol de decisión ({len(reglas)} hojas)\n")
        f.write("Soporte = ejemplos de entrenamiento que cumplen la regla; "
                "confianza = % de esos ejemplos que tienen la clase predicha.\n\n")
        for clase in ["alta", "media", "baja"]:
            sub = sorted([r for r in reglas if r["clase"] == clase], key=lambda r: -r["soporte"])
            f.write(f"===== CONGESTIÓN {clase.upper()} ({len(sub)} reglas) =====\n")
            for i, r in enumerate(sub, 1):
                f.write(formatear_regla(r, i) + "\n")
            f.write("\n")

    graficar_matriz(cm, os.path.join(DIR_RES, "matriz_confusion_arbol.png"),
                    "Árbol de decisión — matriz de confusión (semana 4)")
    graficar_matriz(confusion_matrix(y_pru, entrenados["Random Forest (200 árboles)"].predict(X_pru),
                                     labels=CLASES),
                    os.path.join(DIR_RES, "matriz_confusion_random_forest.png"),
                    "Random Forest — matriz de confusión (semana 4)")
    graficar_importancia(arbol, os.path.join(DIR_RES, "importancia_variables.png"))
    graficar_arbol(arbol, os.path.join(DIR_RES, "arbol_decision.png"))
    curva = graficar_curva_profundidad(X_ent, y_ent, X_pru, y_pru,
                                       os.path.join(DIR_RES, "curva_profundidad.png"))

    joblib.dump(arbol, RUTA_MODELO, compress=3)
    metricas = {
        "registros": len(df), "entrenamiento": len(X_ent), "prueba": len(X_pru),
        "predictoras": PREDICTORAS,
        "hiperparametros_arbol": {k: v for k, v in est.get_params().items()
                                  if k in ("criterion", "max_depth", "min_samples_leaf")},
        "profundidad_arbol": est.get_depth(), "hojas_arbol": int(est.get_n_leaves()),
        "modelos": tabla.round(4).reset_index().to_dict(orient="records"),
        "matriz_confusion_arbol": cm.tolist(),
        "curva_profundidad": {str(k): [round(a, 4), round(b, 4)] for k, (a, b) in curva.items()},
    }
    with open(os.path.join(DIR_RES, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    print(f"\nModelo guardado en {os.path.relpath(RUTA_MODELO)}")
    print(f"Resultados (tablas, reglas y gráficas) en {os.path.relpath(DIR_RES)}/")
    print(f"Tiempo total: {time.time() - t0:.1f} s")
    return metricas


if __name__ == "__main__":
    main()
