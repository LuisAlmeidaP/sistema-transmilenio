"""
preparar_datos.py
=================
Carga el dataset, selecciona las variables predictoras y separa los datos en
entrenamiento y prueba.

Decisiones importantes:
  • FUGA DE INFORMACIÓN: 'validaciones' e 'indice_ocupacion' definen la clase
    (nivel_congestion = f(indice_ocupacion)), así que NO pueden ser predictoras.
    El modelo debe anticipar la congestión con lo que se sabe ANTES de que
    ocurra: estación, día, hora, lluvia y eventos.
  • DIVISIÓN TEMPORAL: se entrena con las 3 primeras semanas y se prueba con la
    4.ª. Mezclar al azar filas del mismo día en ambos conjuntos daría un
    resultado optimista que no se vería en la operación real.
"""
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_DATASET = os.path.join(RAIZ, "datos", "afluencia_transmilenio_muestra.csv")
RUTA_ESTACIONES = os.path.join(RAIZ, "datos", "estaciones.csv")

CLASE = "nivel_congestion"
CLASES = ["baja", "media", "alta"]
CATEGORICAS = ["tipo_estacion", "troncal"]
NUMERICAS = ["hora", "dia_semana", "es_fin_semana", "es_festivo", "num_troncales",
             "latitud", "longitud", "distancia_centro_km", "lluvia_mm", "evento_masivo"]
PREDICTORAS = NUMERICAS + CATEGORICAS
EXCLUIDAS_POR_FUGA = ["validaciones", "indice_ocupacion"]
FECHA_CORTE = "2026-08-24"   # desde esta fecha (4.ª semana) los datos son de prueba


def cargar(ruta=RUTA_DATASET):
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe {ruta}. Genérelo con:  python -m aprendizaje.generar_dataset")
    df = pd.read_csv(ruta, encoding="utf-8")
    faltantes = set(PREDICTORAS + [CLASE, "fecha"]) - set(df.columns)
    if faltantes:
        raise ValueError(f"Al dataset le faltan columnas: {sorted(faltantes)}")
    if df[PREDICTORAS + [CLASE]].isna().any().any():
        raise ValueError("El dataset tiene valores vacíos en las columnas del modelo.")
    desconocidas = set(df[CLASE]) - set(CLASES)
    if desconocidas:
        raise ValueError(f"Clases no reconocidas: {desconocidas}")
    return df


def dividir_temporal(df, fecha_corte=FECHA_CORTE):
    """Devuelve X_ent, X_prueba, y_ent, y_prueba, grupos_ent (fecha de cada fila)."""
    ent = df[df["fecha"] < fecha_corte]
    pru = df[df["fecha"] >= fecha_corte]
    if ent.empty or pru.empty:
        raise ValueError("La fecha de corte deja vacío el conjunto de entrenamiento o el de prueba.")
    return ent[PREDICTORAS], pru[PREDICTORAS], ent[CLASE], pru[CLASE], ent["fecha"]


def preprocesador(escalar=False):
    """One-hot para las categóricas; escalado opcional (solo lo necesita la regresión logística)."""
    num = StandardScaler() if escalar else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
        ("num", num, NUMERICAS),
    ], verbose_feature_names_out=False)
