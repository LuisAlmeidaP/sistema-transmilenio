"""
id3.py
======
Cálculo "a mano" de la entropía y la ganancia de información, el criterio con
el que el algoritmo ID3 (Quinlan) elige qué atributo va en cada nodo del árbol
(Palma Méndez, 2008, cap. 17).

    Entropía:   H(S) = − Σ p_c · log2(p_c)
    Ganancia:   G(S, A) = H(S) − Σ_v (|S_v| / |S|) · H(S_v)

ID3 coloca en la raíz el atributo con MAYOR ganancia y repite el proceso en
cada rama. Aquí se calcula sobre atributos discretos del dataset para
compararlo con la raíz que elige scikit-learn (criterion="entropy").

Uso:  python -m aprendizaje.id3
"""
import math

import pandas as pd

from aprendizaje.preparar_datos import CLASE, cargar, dividir_temporal

# Atributos discretos (ID3 trabaja con valores nominales)
ATRIBUTOS = ["franja", "tipo_dia", "tipo_estacion", "troncal", "llueve", "evento_masivo", "zona"]


def entropia(clases):
    """H(S) en bits. Recibe una lista o Serie con la clase de cada ejemplo."""
    total = len(clases)
    if total == 0:
        return 0.0
    conteo = pd.Series(clases).value_counts()
    return -sum((n / total) * math.log2(n / total) for n in conteo if n > 0)


def ganancia_informacion(df, atributo, clase=CLASE):
    """G(S, A): cuánto se reduce la entropía al partir S por los valores de A."""
    h_s = entropia(df[clase])
    resto = sum(len(sub) / len(df) * entropia(sub[clase]) for _, sub in df.groupby(atributo))
    return h_s - resto


def discretizar(df):
    """Convierte las variables numéricas en atributos nominales para ID3."""
    d = df.copy()
    if "franja" not in d:
        from aprendizaje.generar_dataset import franja
        d["franja"] = d["hora"].map(franja)
    d["tipo_dia"] = "habil"
    d.loc[d["dia_semana"] == 5, "tipo_dia"] = "sabado"
    d.loc[(d["dia_semana"] == 6) | (d["es_festivo"] == 1), "tipo_dia"] = "domingo_festivo"
    d["llueve"] = (d["lluvia_mm"] > 0).map({True: "si", False: "no"})
    d["zona"] = pd.cut(d["distancia_centro_km"], [-1, 3, 7, 100], labels=["centro", "intermedia", "periferia"])
    return d


def tabla_ganancias(df):
    d = discretizar(df)
    filas = [(a, ganancia_informacion(d, a)) for a in ATRIBUTOS]
    return sorted(filas, key=lambda x: x[1], reverse=True), entropia(d[CLASE])


def main():
    df = cargar()
    X, _, y, _, _ = dividir_temporal(df)
    ent = df.loc[X.index]
    ganancias, h = tabla_ganancias(ent)
    print(f"Entropía del conjunto de entrenamiento H(S) = {h:.4f} bits "
          f"({len(ent):,} ejemplos, 3 clases → máximo log2(3) = {math.log2(3):.4f})\n")
    print(f"{'Atributo':<16}{'Ganancia G(S,A)':>18}")
    print("-" * 34)
    for a, g in ganancias:
        print(f"{a:<16}{g:>18.4f}")
    print(f"\nID3 elegiría '{ganancias[0][0]}' como atributo raíz del árbol.")


if __name__ == "__main__":
    main()
