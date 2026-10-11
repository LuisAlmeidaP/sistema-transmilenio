"""
generar_dataset.py
==================
Construye el dataset de MUESTRA  datos/afluencia_transmilenio_muestra.csv

¿Por qué una muestra?  Las fuentes abiertas reales (ver docs/Descripcion_de_los_datos.pdf)
publican salidas agregadas por franja horaria, la geografía de las estaciones y
la programación GTFS, pero no una tabla lista de afluencia POR ESTACIÓN Y POR
HORA con su nivel de congestión. El enunciado permite, en ese caso, desarrollar
un dataset con una muestra de los datos. Este script lo hace de forma
reproducible (semilla fija) y con la MISMA estructura que tendría la tabla real,
de modo que el CSV puede reemplazarse por datos reales sin cambiar el modelo.

Supuestos del generador (modelo académico, no la operación real):
  • Pico de la mañana (≈ 6:30) fuerte en estaciones de la periferia y portales:
    la gente sale de los barrios hacia el centro.
  • Pico de la tarde (≈ 17:30) fuerte en estaciones centrales: la gente regresa.
  • Sábado ≈ 60 % y domingo/festivo ≈ 40 % de la demanda de un día hábil.
  • La lluvia de la tarde aumenta la ocupación (menos gente camina o va en bici).
  • Los eventos masivos (Movistar Arena, El Campín) suman asistentes en la estación
    sede y en sus vecinas, sin importar el día.
  • Ruido aleatorio multiplicativo, para que el problema no sea trivial.

Uso:  python -m aprendizaje.generar_dataset
"""
import datetime as dt
import math
import os
import sys

import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import base_conocimiento as bc            # noqa: E402
from busqueda import haversine_km         # noqa: E402

RUTA_DATASET = os.path.join(RAIZ, "datos", "afluencia_transmilenio_muestra.csv")
RUTA_ESTACIONES = os.path.join(RAIZ, "datos", "estaciones.csv")

SEMILLA = 2026
FECHA_INICIO = dt.date(2026, 8, 3)       # lunes
DIAS = 28                                # 4 semanas: 3 para entrenar, 1 para probar
HORAS = range(5, 23)                     # 5:00 a 22:00 (franjas de una hora)
FESTIVOS = {dt.date(2026, 8, 7): "Batalla de Boyacá",
            dt.date(2026, 8, 17): "Asunción de la Virgen (trasladado)"}
CENTRO = (4.6020, -74.0775)              # Av. Jiménez, punto de referencia del centro
CAPACIDAD_HORA = {"portal": 12000, "transbordo": 9000, "sencilla": 4000}   # pasajeros/hora
UMBRAL_MEDIA, UMBRAL_ALTA = 0.45, 0.75   # índice de ocupación que separa las clases

# Eventos masivos: (fecha, estación sede, hora inicio, hora fin)
EVENTOS = [
    # Movistar Arena: conciertos frecuentes, entre semana y sábados
    *[(dt.date(2026, 8, d), "Movistar Arena", 17, 22) for d in (5, 8, 12, 15, 20, 22, 26, 29)],
    # Estadio El Campín: partidos de fútbol
    *[(dt.date(2026, 8, d), "El Campín", 17, 22) for d in (9, 14, 19, 23, 27)],
]
DIAS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]


def franja(hora):
    if hora < 6:
        return "madrugada"
    if hora < 9:
        return "pico_am"
    if hora < 16:
        return "valle"
    if hora < 20:
        return "pico_pm"
    return "noche"


def tabla_estaciones(rng):
    """Una fila por estación, con sus atributos fijos."""
    filas, vistas = [], {}
    for linea, estaciones in bc.LINEAS.items():
        for nombre, lat, lon in estaciones:
            vistas.setdefault(nombre, {"lat": lat, "lon": lon, "lineas": []})["lineas"].append(linea)
    for nombre, d in vistas.items():
        if nombre.startswith("Portal"):
            tipo = "portal"
        elif len(d["lineas"]) > 1:
            tipo = "transbordo"
        else:
            tipo = "sencilla"
        filas.append({
            "estacion": nombre,
            "troncal": d["lineas"][0],
            "tipo_estacion": tipo,
            "num_troncales": len(d["lineas"]),
            "latitud": d["lat"],
            "longitud": d["lon"],
            "distancia_centro_km": round(haversine_km(d["lat"], d["lon"], *CENTRO), 2),
            "capacidad_hora": CAPACIDAD_HORA[tipo],
            # factor propio de cada estación (barrios más o menos densos); no se publica en el CSV
            "_factor": float(rng.lognormal(0, 0.18)),
        })
    return pd.DataFrame(filas).sort_values("estacion").reset_index(drop=True)


def _gauss(x, mu, sigma):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def perfil_horario(hora, periferia, tipo_dia):
    """Fracción de la demanda pico esperada a esa hora (0..~1.1)."""
    if tipo_dia == "habil":
        return (0.22
                + (0.30 + 0.70 * periferia) * _gauss(hora, 6.5, 1.0)
                + (0.30 + 0.70 * (1 - periferia)) * _gauss(hora, 17.5, 1.2)
                + 0.12 * _gauss(hora, 12.5, 1.5))
    # fin de semana / festivo: sin picos marcados, máximo a media mañana
    return 0.18 + 0.55 * _gauss(hora, 11.5, 3.0) + 0.15 * _gauss(hora, 17, 2.0)


def _estaciones_vecinas(sede):
    vecinas = set()
    for est in bc.LINEAS.values():
        nombres = [n for n, _, _ in est]
        if sede in nombres:
            i = nombres.index(sede)
            vecinas.update(nombres[max(0, i - 1): i + 2])
    vecinas.discard(sede)
    return vecinas


def generar(semilla=SEMILLA, dias=DIAS):
    rng = np.random.default_rng(semilla)
    est = tabla_estaciones(rng)
    periferia = (est["distancia_centro_km"] / est["distancia_centro_km"].max()).clip(0, 1)
    base_tipo = {"portal": 1.50, "transbordo": 1.40, "sencilla": 1.15}

    # Lluvia: ~45 % de los días llueve en la tarde (14:00–19:00), régimen bimodal de Bogotá
    lluvia = {}
    for d in range(dias):
        if rng.random() < 0.45:
            inicio = int(rng.integers(13, 17))
            for h in range(inicio, min(inicio + int(rng.integers(2, 5)), 23)):
                lluvia[(d, h)] = round(float(rng.gamma(1.6, 2.2)), 1)

    efecto_evento = {}
    for fecha, sede, h0, h1 in EVENTOS:
        for h in range(h0, h1 + 1):
            efecto_evento[(fecha, sede, h)] = 0.90
            for v in _estaciones_vecinas(sede):
                efecto_evento[(fecha, v, h)] = 0.45

    filas = []
    for d in range(dias):
        fecha = FECHA_INICIO + dt.timedelta(days=d)
        dsem = fecha.weekday()
        festivo = fecha in FESTIVOS
        if festivo or dsem == 6:
            tipo_dia, factor_dia = "festivo_domingo", 0.42
        elif dsem == 5:
            tipo_dia, factor_dia = "sabado", 0.62
        else:
            tipo_dia, factor_dia = "habil", 1.0
        for i, e in est.iterrows():
            for h in HORAS:
                mm = lluvia.get((d, h), 0.0)
                ev = efecto_evento.get((fecha, e["estacion"], h), 0.0)
                ocup = (base_tipo[e["tipo_estacion"]] * e["_factor"]
                        * perfil_horario(h, periferia[i], tipo_dia) * factor_dia
                        * (1 + 0.035 * min(mm, 8))
                        * rng.lognormal(0, 0.16)
                        + ev * rng.uniform(0.8, 1.2))   # asistentes al evento (suma fija)
                validaciones = int(round(ocup * e["capacidad_hora"]))
                indice = validaciones / e["capacidad_hora"]
                nivel = "alta" if indice >= UMBRAL_ALTA else "media" if indice >= UMBRAL_MEDIA else "baja"
                filas.append({
                    "fecha": fecha.isoformat(),
                    "dia_semana": dsem,
                    "nombre_dia": DIAS_ES[dsem],
                    "es_fin_semana": int(dsem >= 5),
                    "es_festivo": int(festivo),
                    "hora": h,
                    "franja": franja(h),
                    "estacion": e["estacion"],
                    "troncal": e["troncal"],
                    "tipo_estacion": e["tipo_estacion"],
                    "num_troncales": e["num_troncales"],
                    "latitud": e["latitud"],
                    "longitud": e["longitud"],
                    "distancia_centro_km": e["distancia_centro_km"],
                    "lluvia_mm": mm,
                    "evento_masivo": int(ev > 0),
                    "capacidad_hora": e["capacidad_hora"],
                    "validaciones": validaciones,
                    "indice_ocupacion": round(indice, 3),
                    "nivel_congestion": nivel,
                })
    return pd.DataFrame(filas), est.drop(columns="_factor")


def main():
    df, est = generar()
    os.makedirs(os.path.dirname(RUTA_DATASET), exist_ok=True)
    df.to_csv(RUTA_DATASET, index=False, encoding="utf-8")
    est.to_csv(RUTA_ESTACIONES, index=False, encoding="utf-8")
    print(f"Dataset: {RUTA_DATASET}")
    print(f"  {len(df):,} registros  ·  {df['estacion'].nunique()} estaciones  ·  "
          f"{df['fecha'].nunique()} días  ·  {len(HORAS)} franjas horarias por día")
    print("  Distribución de la clase nivel_congestion:")
    for k, v in df["nivel_congestion"].value_counts(normalize=True).items():
        print(f"    {k:<6} {v:6.1%}")
    print(f"Estaciones: {RUTA_ESTACIONES} ({len(est)} filas)")


if __name__ == "__main__":
    main()
