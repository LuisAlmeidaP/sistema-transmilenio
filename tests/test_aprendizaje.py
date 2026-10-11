"""
Pruebas del componente de aprendizaje supervisado (Actividad 3).
Ejecutar desde la raíz del proyecto:   python -m unittest -v
"""
import contextlib
import io
import math
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd                                                            # noqa: E402
from sklearn.metrics import f1_score, recall_score                             # noqa: E402
from sklearn.pipeline import Pipeline                                          # noqa: E402
from sklearn.tree import DecisionTreeClassifier                                # noqa: E402

from aprendizaje import generar_dataset as gen                                 # noqa: E402
from aprendizaje.entrenar import extraer_reglas                                # noqa: E402
from aprendizaje.id3 import entropia, ganancia_informacion                     # noqa: E402
from aprendizaje.predecir import (buscar_estacion, construir_fila, dia_a_numero,  # noqa: E402
                                  explicar, predecir)
from aprendizaje.preparar_datos import (CLASE, EXCLUIDAS_POR_FUGA, PREDICTORAS,  # noqa: E402
                                        cargar, dividir_temporal, preprocesador)


class TestDatos(unittest.TestCase):
    """Calidad e integridad del dataset."""

    @classmethod
    def setUpClass(cls):
        cls.df = cargar()

    def test_01_tamano_y_estructura(self):
        """Una fila por estación, día y hora: 95 × 28 × 18 = 47 880."""
        self.assertEqual(len(self.df), 95 * 28 * 18)
        self.assertEqual(self.df["estacion"].nunique(), 95)
        self.assertFalse(self.df.duplicated(["fecha", "hora", "estacion"]).any())

    def test_02_sin_vacios_y_clases_validas(self):
        self.assertFalse(self.df.isna().any().any())
        self.assertEqual(set(self.df[CLASE]), {"baja", "media", "alta"})

    def test_03_etiqueta_coherente_con_ocupacion(self):
        """La clase se deriva del índice de ocupación con los umbrales 0,45 y 0,75."""
        esperado = pd.cut(self.df["validaciones"] / self.df["capacidad_hora"],
                          [-1, gen.UMBRAL_MEDIA, gen.UMBRAL_ALTA, 100], right=False,
                          labels=["baja", "media", "alta"]).astype(str)
        self.assertTrue((esperado == self.df[CLASE]).all())

    def test_04_generador_reproducible(self):
        """Con la misma semilla el generador produce exactamente los mismos datos."""
        a, _ = gen.generar(semilla=7, dias=2)
        b, _ = gen.generar(semilla=7, dias=2)
        pd.testing.assert_frame_equal(a, b)

    def test_05_estaciones_coinciden_con_base_conocimiento(self):
        """Las estaciones del dataset son las mismas de la base de conocimiento (Actividad 2)."""
        import base_conocimiento as bc
        nombres = {n for est in bc.LINEAS.values() for n, _, _ in est}
        self.assertEqual(set(self.df["estacion"]), nombres)

    def test_06_sin_fuga_de_informacion(self):
        """Las columnas que definen la clase no pueden usarse para predecirla."""
        for col in EXCLUIDAS_POR_FUGA + [CLASE]:
            self.assertNotIn(col, PREDICTORAS)

    def test_07_division_temporal(self):
        """Todo el entrenamiento es anterior a la prueba (semanas 1–3 vs. semana 4)."""
        X_e, X_p, _, _, fechas = dividir_temporal(self.df)
        fechas_p = self.df.loc[X_p.index, "fecha"]
        self.assertLess(fechas.max(), fechas_p.min())
        self.assertEqual(len(X_e) + len(X_p), len(self.df))
        self.assertEqual(len(X_p), 95 * 7 * 18)


class TestID3(unittest.TestCase):
    """Entropía y ganancia de información (Palma, cap. 17)."""

    def test_08_entropia_casos_conocidos(self):
        self.assertAlmostEqual(entropia(["alta"] * 10), 0.0)
        self.assertAlmostEqual(entropia(["a", "b"] * 5), 1.0)
        self.assertAlmostEqual(entropia(["a", "b", "c"] * 4), math.log2(3))

    def test_09_ganancia_atributo_perfecto_e_irrelevante(self):
        df = pd.DataFrame({"perfecto": ["x", "x", "y", "y"] * 5,
                           "irrelevante": ["p", "q", "p", "q"] * 5,
                           CLASE: ["alta", "alta", "baja", "baja"] * 5})
        self.assertAlmostEqual(ganancia_informacion(df, "perfecto"), entropia(df[CLASE]))
        self.assertAlmostEqual(ganancia_informacion(df, "irrelevante"), 0.0)


class TestModelo(unittest.TestCase):
    """Desempeño y comportamiento del árbol de decisión."""

    @classmethod
    def setUpClass(cls):
        df = cargar()
        cls.X_e, cls.X_p, cls.y_e, cls.y_p, _ = dividir_temporal(df)
        cls.modelo = Pipeline([("prep", preprocesador()),
                               ("modelo", DecisionTreeClassifier(criterion="entropy", max_depth=16,
                                                                 min_samples_leaf=10, random_state=42))])
        cls.modelo.fit(cls.X_e, cls.y_e)
        cls.y_hat = cls.modelo.predict(cls.X_p)

    def test_10_supera_linea_base(self):
        """F1 macro en la semana de prueba ≥ 0,75 (la línea base obtiene ≈ 0,25)."""
        self.assertGreaterEqual(f1_score(self.y_p, self.y_hat, average="macro"), 0.75)

    def test_11_detecta_congestion_alta(self):
        """Recall de la clase ALTA ≥ 0,75: el modelo no debe pasar por alto las horas críticas."""
        self.assertGreaterEqual(recall_score(self.y_p, self.y_hat, labels=["alta"], average="macro"), 0.75)

    def test_12_nunca_confunde_extremos(self):
        """Casi nunca predice BAJA cuando es ALTA (el error más costoso): menos del 1 %."""
        reales_alta = self.y_p == "alta"
        self.assertLess((self.y_hat[reales_alta.values] == "baja").mean(), 0.01)

    def test_13_patron_hora_pico(self):
        """Portal del Norte: ALTA un lunes a las 6:00 y BAJA un domingo a la misma hora."""
        e = buscar_estacion("Portal del Norte")
        self.assertEqual(predecir(self.modelo, construir_fila(e, 0, 6))[0], "alta")
        self.assertEqual(predecir(self.modelo, construir_fila(e, 6, 6))[0], "baja")

    def test_14_evento_aumenta_congestion(self):
        """Un evento masivo un sábado a las 20:00 eleva la congestión en Movistar Arena."""
        e = buscar_estacion("Movistar Arena")
        sin = predecir(self.modelo, construir_fila(e, 5, 20, evento=False))[1]
        con = predecir(self.modelo, construir_fila(e, 5, 20, evento=True))[1]
        self.assertGreater(con.get("alta", 0), sin.get("alta", 0))

    def test_15_reglas_extraidas(self):
        """Una regla por hoja; los soportes suman el total de entrenamiento; confianza en [0, 1]."""
        reglas = extraer_reglas(self.modelo)
        self.assertEqual(len(reglas), self.modelo.named_steps["modelo"].get_n_leaves())
        self.assertEqual(sum(r["soporte"] for r in reglas), len(self.X_e))
        self.assertTrue(all(0 <= r["confianza"] <= 1 for r in reglas))

    def test_16_prediccion_explicada(self):
        """La explicación devuelve las condiciones del camino raíz→hoja de la consulta."""
        condiciones, soporte = explicar(self.modelo, construir_fila("Portal del Norte", 0, 6))
        self.assertTrue(any("hora" in c for c in condiciones))
        self.assertGreater(soporte, 0)

    def test_17_entradas_invalidas(self):
        with self.assertRaises(ValueError):
            buscar_estacion("Estación Inventada")
        with self.assertRaises(ValueError):
            dia_a_numero("juernes")
        self.assertEqual(dia_a_numero("Sábado"), 5)
        self.assertEqual(dia_a_numero("miercoles"), 2)
        self.assertEqual(dia_a_numero("vie"), 4)

    def test_18_integracion_con_rutas(self):
        """main.py --dia --hora: estima la congestión donde el usuario sube, transborda y baja."""
        import main
        from sistema import SistemaRutas
        s = SistemaRutas()
        res, _ = s.ruta("Portal El Dorado", "Portal Américas")
        with mock.patch("aprendizaje.predecir.cargar_modelo", return_value=self.modelo), \
                contextlib.redirect_stdout(io.StringIO()):
            puntos = main.mostrar_congestion(s, res, "lunes", 7)
        self.assertEqual([p[0] for p in puntos], ["Sube", "Transbordo", "Transbordo", "Baja"])
        self.assertEqual(puntos[0][1], "Portal El Dorado")
        self.assertTrue(all(p[3] in ("baja", "media", "alta") for p in puntos))


if __name__ == "__main__":
    unittest.main()
