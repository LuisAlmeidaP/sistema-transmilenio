"""
predecir.py
===========
Usa el árbol de decisión entrenado para predecir la congestión de una estación
y EXPLICA la predicción mostrando la regla (el camino del árbol) que se aplicó.

Ejemplos:
  python -m aprendizaje.predecir --estacion "Portal del Norte" --dia lunes --hora 6
  python -m aprendizaje.predecir --estacion "Movistar Arena" --dia jueves --hora 19 --evento
  python -m aprendizaje.predecir --estacion "Calle 26" --dia martes          (perfil de todo el día)
"""
import argparse
import os
import sys

import joblib
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from aprendizaje.entrenar import RUTA_MODELO, _simplificar          # noqa: E402
from aprendizaje.generar_dataset import DIAS_ES, HORAS              # noqa: E402
from aprendizaje.preparar_datos import PREDICTORAS, RUTA_ESTACIONES  # noqa: E402
from busqueda import normalizar                                     # noqa: E402

BARRA = {"baja": "▁▁", "media": "▄▄", "alta": "██"}


def cargar_modelo(ruta=RUTA_MODELO):
    if not os.path.exists(ruta):
        raise FileNotFoundError("No hay modelo entrenado. Ejecute primero:  python -m aprendizaje.entrenar")
    try:
        return joblib.load(ruta)
    except Exception as e:   # p. ej. el modelo se guardó con otra versión de scikit-learn
        raise FileNotFoundError(f"No se pudo cargar el modelo ({type(e).__name__}). "
                                "Vuelva a entrenarlo con:  python -m aprendizaje.entrenar") from e


def buscar_estacion(nombre, ruta=RUTA_ESTACIONES):
    est = pd.read_csv(ruta, encoding="utf-8")
    coincide = est[est["estacion"].map(normalizar) == normalizar(nombre)]
    if coincide.empty:
        parecidas = est[est["estacion"].map(normalizar).str.contains(normalizar(nombre), regex=False)]
        sugerencia = f" ¿Quiso decir: {', '.join(parecidas['estacion'])}?" if not parecidas.empty else ""
        raise ValueError(f"La estación '{nombre}' no existe.{sugerencia}")
    return coincide.iloc[0]


def dia_a_numero(dia):
    """Acepta el nombre completo o la abreviatura de 3 letras, con o sin tildes."""
    d = normalizar(dia)
    for i, nombre in enumerate(DIAS_ES):
        n = normalizar(nombre)
        if d == n or d == n[:3]:
            return i
    raise ValueError(f"Día no reconocido: '{dia}'. Use lunes, martes, ..., domingo.")


def construir_fila(estacion, dia_semana, hora, lluvia_mm=0.0, evento=False, festivo=False):
    e = buscar_estacion(estacion) if isinstance(estacion, str) else estacion
    fila = {
        "hora": int(hora), "dia_semana": int(dia_semana), "es_fin_semana": int(dia_semana >= 5),
        "es_festivo": int(festivo), "num_troncales": int(e["num_troncales"]),
        "latitud": float(e["latitud"]), "longitud": float(e["longitud"]),
        "distancia_centro_km": float(e["distancia_centro_km"]), "lluvia_mm": float(lluvia_mm),
        "evento_masivo": int(evento), "tipo_estacion": e["tipo_estacion"], "troncal": e["troncal"],
    }
    return pd.DataFrame([fila])[PREDICTORAS]


def predecir(modelo, fila):
    """Devuelve (clase, {clase: probabilidad})."""
    proba = modelo.predict_proba(fila)[0]
    clases = list(modelo.named_steps["modelo"].classes_)
    return clases[int(proba.argmax())], dict(zip(clases, proba))


def explicar(modelo, fila):
    """Condiciones del camino raíz→hoja que siguió esta consulta (la regla aplicada)."""
    prep, arbol = modelo.named_steps["prep"], modelo.named_steps["modelo"]
    Xt = prep.transform(fila)
    nombres = prep.get_feature_names_out()
    camino = arbol.decision_path(Xt).indices
    t = arbol.tree_
    condiciones = []
    for nodo in camino[:-1]:
        f, u = nombres[t.feature[nodo]], t.threshold[nodo]
        condiciones.append((f, u, Xt[0, t.feature[nodo]] <= u))
    hoja = camino[-1]
    return _simplificar(condiciones), int(t.n_node_samples[hoja])


def main(argv=None):
    p = argparse.ArgumentParser(description="Predice la congestión de una estación (árbol de decisión)")
    p.add_argument("--estacion", required=True)
    p.add_argument("--dia", required=True, help="lunes, martes, ..., domingo")
    p.add_argument("--hora", type=int, help="5 a 22. Si se omite, muestra el perfil de todo el día")
    p.add_argument("--lluvia", type=float, default=0.0, help="milímetros de lluvia en esa hora")
    p.add_argument("--evento", action="store_true", help="hay un evento masivo cerca de la estación")
    p.add_argument("--festivo", action="store_true")
    a = p.parse_args(argv)

    try:
        modelo = cargar_modelo()
        e = buscar_estacion(a.estacion)
        d = dia_a_numero(a.dia)
    except (ValueError, FileNotFoundError) as err:
        print(f"Error: {err}")
        return 1
    contexto = f"{e['estacion']} ({e['tipo_estacion']}, troncal {e['troncal']}) · {DIAS_ES[d]}"
    if a.festivo:
        contexto += " festivo"
    if a.lluvia:
        contexto += f" · lluvia {a.lluvia} mm"
    if a.evento:
        contexto += " · evento masivo"

    if a.hora is None:
        print(f"Congestión estimada por hora — {contexto}\n")
        for h in HORAS:
            clase, proba = predecir(modelo, construir_fila(e, d, h, a.lluvia, a.evento, a.festivo))
            print(f"  {h:02d}:00  {BARRA[clase]}  {clase:<6} (P = {proba[clase]:.0%})")
        return 0

    if not 5 <= a.hora <= 22:
        print("Error: la hora debe estar entre 5 y 22 (horario modelado).")
        return 1
    fila = construir_fila(e, d, a.hora, a.lluvia, a.evento, a.festivo)
    clase, proba = predecir(modelo, fila)
    condiciones, soporte = explicar(modelo, fila)
    print(f"Consulta : {contexto} · {a.hora:02d}:00")
    print(f"Predicción: congestión {clase.upper()}")
    print("Probabilidades: " + "  ".join(f"{c} {proba.get(c, 0):.0%}" for c in ["baja", "media", "alta"]))
    print("\n¿Por qué? Regla del árbol que se aplicó:")
    print("  SI " + "\n   ∧ ".join(condiciones))
    print(f"  ENTONCES congestión = {clase.upper()}   (hoja con {soporte} ejemplos de entrenamiento)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
