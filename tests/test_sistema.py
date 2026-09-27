"""
Pruebas unitarias del sistema inteligente de rutas.
Ejecutar desde la raíz del proyecto:   python -m unittest -v
"""
import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base_conocimiento as bc                       # noqa: E402
from busqueda import a_estrella, dijkstra            # noqa: E402
from motor_inferencia import MotorInferencia, unificar  # noqa: E402
from sistema import SistemaRutas                     # noqa: E402


class TestMotorInferencia(unittest.TestCase):
    """Pruebas del razonamiento lógico (caps. 2 y 3)."""

    @classmethod
    def setUpClass(cls):
        cls.s = SistemaRutas()
        cls.m = cls.s.motor

    def test_01_unificacion(self):
        self.assertEqual(unificar(("p", "?x", "b"), ("p", "a", "b"), {}), {"?x": "a"})
        self.assertIsNone(unificar(("p", "?x", "?x"), ("p", "a", "b"), {}))

    def test_02_regla_bidireccional(self):
        """R1 y R2: si siguiente(A,B,L) entonces conecta(A,B,L) y conecta(B,A,L)."""
        self.assertTrue(self.m.es_verdad(("conecta", "Portal del Norte", "Toberín", "Caracas-Autonorte")))
        self.assertTrue(self.m.es_verdad(("conecta", "Toberín", "Portal del Norte", "Caracas-Autonorte")))

    def test_03_estaciones_transbordo(self):
        """R3 y R4 deducen exactamente las estaciones compartidas por dos troncales."""
        esperadas = {"Av. Jiménez", "Calle 26", "Héroes", "Polo", "Ricaurte"}
        self.assertEqual(set(self.s.estaciones_transbordo()), esperadas)

    def test_04_clausura_transitiva(self):
        """R7: toda estación alcanza a todas las demás (red conexa)."""
        n = len(self.s.estaciones())
        self.assertEqual(len(self.m.consultar(("alcanzable", "?a", "?b"))), n * (n - 1))

    def test_05_explicacion(self):
        traza = self.m.explicar(("transbordo", "Ricaurte", "NQS", "Américas"))
        self.assertIn("R3_transbordo", traza[0])

    def test_06_estacion_cerrada_rompe_conexion(self):
        """R5 + negación por fallo: cerrar Ricaurte y Av. Jiménez aísla la troncal Américas."""
        s = SistemaRutas(cerradas=["Ricaurte", "Av. Jiménez"])
        self.assertFalse(s.motor.es_verdad(("alcanzable", "Calle 100", "Portal Américas")))
        r, deducible = s.ruta("Calle 100", "Portal Américas")
        self.assertFalse(deducible)
        self.assertFalse(r.encontrada)

    def test_07_motor_generico(self):
        """El motor sirve para cualquier BC (independencia conocimiento/control)."""
        hechos = {("padre", "ana", "luis"), ("padre", "luis", "eva")}
        reglas = [("abuelo", [("padre", "?x", "?y"), ("padre", "?y", "?z")], ("abuelo", "?x", "?z"))]
        m = MotorInferencia(hechos, reglas)
        m.inferir()
        self.assertTrue(m.es_verdad(("abuelo", "ana", "eva")))


class TestBusqueda(unittest.TestCase):
    """Pruebas de la búsqueda heurística (cap. 9)."""

    @classmethod
    def setUpClass(cls):
        cls.s = SistemaRutas()

    def test_08_ruta_directa_sin_transbordo(self):
        r, _ = self.s.ruta("Portal del Norte", "Calle 100")
        self.assertEqual(r.transbordos, 0)
        self.assertEqual(r.estaciones[0], "Portal del Norte")
        self.assertEqual(r.estaciones[-1], "Calle 100")

    def test_09_ruta_con_transbordo(self):
        r, _ = self.s.ruta("Portal 80", "Portal Sur")
        self.assertEqual(r.transbordos, 1)
        self.assertIn("Polo", r.estaciones)

    def test_10_origen_igual_destino(self):
        r, _ = self.s.ruta("Héroes", "Héroes")
        self.assertTrue(r.encontrada)
        self.assertAlmostEqual(r.costo_min, 0.0)

    def test_11_nombre_sin_tildes(self):
        r, _ = self.s.ruta("portal americas", "toberin")
        self.assertTrue(r.encontrada)

    def test_12_estacion_inexistente(self):
        with self.assertRaises(ValueError):
            self.s.ruta("Portal Narnia", "Polo")

    def test_13_heuristica_admisible(self):
        """h(n) ≤ h*(n) para todos los pares: comparar con el costo óptimo real."""
        est = self.s.estaciones()
        for o, d in itertools.product(est[::4], est[::3]):
            if o == d:
                continue
            opt = dijkstra(self.s.red, o, d).costo_min
            self.assertLessEqual(self.s.red.heuristica(o, d), opt + 1e-9, (o, d))

    def test_14_astar_optimo_y_eficiente(self):
        """A* halla el mismo costo que Dijkstra y nunca expande más nodos."""
        est = self.s.estaciones()
        total_a = total_d = 0
        for o, d in itertools.product(est[::5], est[::7]):
            if o == d:
                continue
            ra, rd = a_estrella(self.s.red, o, d), dijkstra(self.s.red, o, d)
            self.assertAlmostEqual(ra.costo_min, rd.costo_min, places=6)
            self.assertLessEqual(ra.nodos_expandidos, rd.nodos_expandidos)
            total_a += ra.nodos_expandidos
            total_d += rd.nodos_expandidos
        self.assertLess(total_a, total_d)

    def test_15_ruta_evita_estacion_cerrada(self):
        s = SistemaRutas(cerradas=["Calle 26"])
        r, _ = s.ruta("Toberín", "Ricaurte")
        self.assertTrue(r.encontrada)
        self.assertNotIn("Calle 26", r.estaciones)

    def test_16_parametros_coherentes(self):
        self.assertGreaterEqual(bc.VELOCIDAD_MAX_KMH, bc.VELOCIDAD_BUS_KMH)


if __name__ == "__main__":
    unittest.main(verbosity=2)
