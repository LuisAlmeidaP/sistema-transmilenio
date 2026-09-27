"""
sistema.py
==========
Integra los tres componentes del sistema inteligente:

    Base de conocimiento  ──►  Motor de inferencia  ──►  Búsqueda heurística A*
      (hechos + reglas)         (deduce conecta,            (encuentra la mejor
                                 transbordo, alcanzable)     ruta en el espacio
                                                             de estados deducido)
"""
import base_conocimiento as bc
from busqueda import ALGORITMOS, RedTransporte
from motor_inferencia import MotorInferencia

NOMBRES = {"astar": "A*", "dijkstra": "Costo uniforme (Dijkstra)",
           "voraz": "Primero el mejor (voraz)", "bfs": "Amplitud (BFS)"}
PENALIZACION_MIN_TRANSBORDOS = 60.0   # criterio "menos transbordos": un transbordo "cuesta" 1 hora


class SistemaRutas:
    def __init__(self, cerradas=(), criterio="tiempo", lineas=bc.LINEAS):
        self.motor = MotorInferencia(bc.generar_hechos(lineas), bc.REGLAS)
        self.cerradas = []
        # Se crea una red provisional solo para validar los nombres de estaciones cerradas
        coords = {t["?e"] for t in self.motor.consultar(("estacion", "?e", "?a", "?b"))}
        for c in cerradas:
            nombre = self._resolver(c, coords)
            self.motor.afirmar(("cerrada", nombre))
            self.cerradas.append(nombre)
        self.nuevos_hechos = self.motor.inferir()
        penal = PENALIZACION_MIN_TRANSBORDOS if criterio == "transbordos" else bc.PENALIZACION_TRANSBORDO_MIN
        self.criterio = criterio
        self.red = RedTransporte(self.motor, penal)

    @staticmethod
    def _resolver(texto, nombres):
        from busqueda import normalizar
        for n in nombres:
            if normalizar(n) == normalizar(texto):
                return n
        raise ValueError(f"La estación '{texto}' no existe en la base de conocimiento.")

    def estaciones(self):
        return sorted(self.red.coords)

    def lineas(self):
        return sorted({t["?l"] for t in self.motor.consultar(("pertenece", "?e", "?l"))})

    def estaciones_transbordo(self):
        return sorted(t["?x"] for t in self.motor.consultar(("estacion_transbordo", "?x")))

    def ruta(self, origen, destino, algoritmo="astar"):
        """Consulta principal. Devuelve (Resultado, diagnóstico lógico)."""
        o = self.red.resolver_nombre(origen)
        d = self.red.resolver_nombre(destino)
        for e in (o, d):
            if e in self.cerradas:
                raise ValueError(f"La estación '{e}' está cerrada.")
        # 1) Razonamiento lógico: ¿la BC permite deducir alcanzable(o, d)?
        deducible = o == d or self.motor.es_verdad(("alcanzable", o, d))
        # 2) Búsqueda: solo si la lógica garantiza que existe solución
        if not deducible:
            from busqueda import Resultado
            return Resultado(NOMBRES[algoritmo], False), False
        resultado = ALGORITMOS[algoritmo](self.red, o, d)
        return resultado, deducible
