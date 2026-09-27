"""
busqueda.py
===========
Técnicas de búsqueda sobre el espacio de estados deducido por el motor de
inferencia (Benítez, 2014, cap. 9).

Estado:      n = (estación, línea en la que viaja el usuario)
Operadores:  • avanzar por un tramo habilitado      costo = tiempo del tramo
             • hacer transbordo en una estación     costo = penalización
Objetivo:    cualquier estado cuya estación sea el destino.

A*:          f(n) = g(n) + h(n)
             g(n) = tiempo acumulado desde el origen                 [min]
             h(n) = d_haversine(n, destino) / v_max · 60              [min]

h(n) es ADMISIBLE: ningún bus supera v_max y la distancia en línea recta
es la menor posible, luego h(n) ≤ h*(n). También es CONSISTENTE por la
desigualdad triangular: h(x) ≤ c(x,y) + h(y).
"""
import heapq
import itertools
import math
import unicodedata
from collections import deque
from dataclasses import dataclass, field

import base_conocimiento as bc

RADIO_TIERRA_KM = 6371.0


# ---------------------------------------------------------------- utilidades
def normalizar(texto):
    """Minúsculas y sin tildes, para aceptar 'heroes' == 'Héroes'."""
    t = unicodedata.normalize("NFKD", texto.strip().lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia ortodrómica entre dos puntos:
    d = 2R·asin( √( sin²(Δφ/2) + cos φ1 · cos φ2 · sin²(Δλ/2) ) )
    """
    f1, f2 = math.radians(lat1), math.radians(lat2)
    df, dl = f2 - f1, math.radians(lon2 - lon1)
    a = math.sin(df / 2) ** 2 + math.cos(f1) * math.cos(f2) * math.sin(dl / 2) ** 2
    return 2 * RADIO_TIERRA_KM * math.asin(math.sqrt(a))


# ------------------------------------------------------------- red de estados
class RedTransporte:
    """Construye el grafo de estados a partir de los hechos inferidos."""

    def __init__(self, motor, penalizacion_transbordo=bc.PENALIZACION_TRANSBORDO_MIN):
        self.motor = motor
        self.penalizacion = penalizacion_transbordo
        self.coords = {t["?e"]: (t["?lat"], t["?lon"])
                       for t in motor.consultar(("estacion", "?e", "?lat", "?lon"))}
        self.lineas_de = {}
        for t in motor.consultar(("pertenece", "?e", "?l")):
            self.lineas_de.setdefault(t["?e"], set()).add(t["?l"])
        self.tramos = {}          # (x, linea) -> [(y, tiempo)]
        for t in motor.consultar(("tramo_habilitado", "?x", "?y", "?l")):
            x, y, l = t["?x"], t["?y"], t["?l"]
            self.tramos.setdefault((x, l), []).append((y, self.tiempo_tramo(x, y)))
        self.transbordos = {}     # (x, linea) -> [linea destino]
        for t in motor.consultar(("transbordo", "?x", "?l1", "?l2")):
            self.transbordos.setdefault((t["?x"], t["?l1"]), []).append(t["?l2"])
        self._indice = {normalizar(e): e for e in self.coords}

    # --- conocimiento numérico ---
    def distancia_km(self, a, b):
        return haversine_km(*self.coords[a], *self.coords[b])

    def tiempo_tramo(self, a, b):
        """t = d / v · 60 + t_parada   [min]"""
        return self.distancia_km(a, b) / bc.VELOCIDAD_BUS_KMH * 60 + bc.TIEMPO_PARADA_MIN

    def heuristica(self, estacion, destino):
        """h(n) = d(n, destino) / v_max · 60   [min]"""
        return self.distancia_km(estacion, destino) / bc.VELOCIDAD_MAX_KMH * 60

    def resolver_nombre(self, texto):
        """Devuelve el nombre oficial o lanza ValueError con sugerencias."""
        clave = normalizar(texto)
        if clave in self._indice:
            return self._indice[clave]
        parecidas = [e for k, e in self._indice.items() if clave in k]
        if len(parecidas) == 1:
            return parecidas[0]
        msg = f"La estación '{texto}' no existe en la base de conocimiento."
        if parecidas:
            msg += " ¿Quiso decir: " + ", ".join(sorted(parecidas)) + "?"
        raise ValueError(msg)

    # --- generación de sucesores ---
    def sucesores(self, estado):
        estacion, linea = estado
        for destino, t in self.tramos.get((estacion, linea), []):
            yield (destino, linea), t, "viaje"
        for otra in self.transbordos.get((estacion, linea), []):
            if (estacion, otra) in self.tramos:          # solo si la otra línea está operativa ahí
                yield (estacion, otra), self.penalizacion, "transbordo"

    def estados_iniciales(self, origen):
        return [(origen, l) for l in sorted(self.lineas_de.get(origen, ()))
                if (origen, l) in self.tramos]


# ------------------------------------------------------------------ resultado
@dataclass
class Resultado:
    algoritmo: str
    encontrada: bool
    camino: list = field(default_factory=list)   # lista de estados (estacion, linea)
    costo_min: float = 0.0
    nodos_expandidos: int = 0
    nodos_generados: int = 0

    @property
    def transbordos(self):
        return sum(1 for a, b in zip(self.camino, self.camino[1:]) if a[0] == b[0])

    @property
    def estaciones(self):
        vistas = []
        for e, _ in self.camino:
            if not vistas or vistas[-1] != e:
                vistas.append(e)
        return vistas

    def tiempo_real(self, red):
        """Tiempo con la penalización estándar (útil cuando el criterio es 'transbordos')."""
        total = 0.0
        for a, b in zip(self.camino, self.camino[1:]):
            total += bc.PENALIZACION_TRANSBORDO_MIN if a[0] == b[0] else red.tiempo_tramo(a[0], b[0])
        return total

    def instrucciones(self, red):
        """Convierte el camino en indicaciones legibles por el usuario."""
        if not self.encontrada:
            return ["No existe una ruta con las condiciones dadas."]
        if len(self.camino) <= 1:
            return ["Ya se encuentra en el destino."]
        pasos, i = [], 0
        while i < len(self.camino) - 1:
            a, b = self.camino[i], self.camino[i + 1]
            if a[0] == b[0]:
                pasos.append(f"Transbordo en {a[0]}: cambie de la troncal {a[1]} a la troncal {b[1]} "
                             f"(+{bc.PENALIZACION_TRANSBORDO_MIN:.0f} min)")
                i += 1
                continue
            j, t = i, 0.0
            while j < len(self.camino) - 1 and self.camino[j + 1][1] == a[1] \
                    and self.camino[j][0] != self.camino[j + 1][0]:
                t += red.tiempo_tramo(self.camino[j][0], self.camino[j + 1][0])
                j += 1
            paradas = j - i
            pasos.append(f"Tome la troncal {a[1]} desde {a[0]} hasta {self.camino[j][0]} "
                         f"({paradas} parada{'s' if paradas != 1 else ''}, ≈{t:.1f} min)")
            i = j
        return [f"{k}. {p}" for k, p in enumerate(pasos, 1)]


def _reconstruir(padres, final):
    camino = [final]
    while padres[camino[-1]] is not None:
        camino.append(padres[camino[-1]])
    return camino[::-1]


# ------------------------------------------------------------------ algoritmos
def _busqueda_prioridad(red, origen, destino, nombre, prioridad):
    """Esquema general de búsqueda con cola de prioridad (grafo, con lista cerrada).

    prioridad(g, estado) decide el orden de expansión:
        A*       -> g + h
        Dijkstra -> g               (costo uniforme)
        Voraz    -> h               (primero el mejor, no garantiza óptimo)
    """
    contador = itertools.count()
    abiertos, g, padres = [], {}, {}
    for s in red.estados_iniciales(origen):
        g[s], padres[s] = 0.0, None
        heapq.heappush(abiertos, (prioridad(0.0, s), next(contador), s))
    cerrados, expandidos, generados = set(), 0, len(abiertos)
    while abiertos:
        _, _, actual = heapq.heappop(abiertos)
        if actual in cerrados:
            continue
        if actual[0] == destino:
            return Resultado(nombre, True, _reconstruir(padres, actual), g[actual], expandidos, generados)
        cerrados.add(actual)
        expandidos += 1
        for sig, costo, _ in red.sucesores(actual):
            nuevo_g = g[actual] + costo
            if sig not in cerrados and nuevo_g < g.get(sig, math.inf):
                g[sig], padres[sig] = nuevo_g, actual
                heapq.heappush(abiertos, (prioridad(nuevo_g, sig), next(contador), sig))
                generados += 1
    return Resultado(nombre, False, nodos_expandidos=expandidos, nodos_generados=generados)


def a_estrella(red, origen, destino):
    return _busqueda_prioridad(red, origen, destino, "A*",
                               lambda g, s: g + red.heuristica(s[0], destino))


def dijkstra(red, origen, destino):
    return _busqueda_prioridad(red, origen, destino, "Costo uniforme (Dijkstra)",
                               lambda g, s: g)


def voraz(red, origen, destino):
    return _busqueda_prioridad(red, origen, destino, "Primero el mejor (voraz)",
                               lambda g, s: red.heuristica(s[0], destino))


def amplitud(red, origen, destino):
    """Búsqueda en amplitud (BFS): minimiza el número de operadores, no el tiempo."""
    padres, frontera = {}, deque()
    for s in red.estados_iniciales(origen):
        padres[s] = None
        frontera.append(s)
    expandidos, generados = 0, len(frontera)
    while frontera:
        actual = frontera.popleft()
        if actual[0] == destino:
            camino = _reconstruir(padres, actual)
            costo = sum(red.penalizacion if a[0] == b[0] else red.tiempo_tramo(a[0], b[0])
                        for a, b in zip(camino, camino[1:]))
            return Resultado("Amplitud (BFS)", True, camino, costo, expandidos, generados)
        expandidos += 1
        for sig, _, _ in red.sucesores(actual):
            if sig not in padres:
                padres[sig] = actual
                frontera.append(sig)
                generados += 1
    return Resultado("Amplitud (BFS)", False, nodos_expandidos=expandidos, nodos_generados=generados)


ALGORITMOS = {"astar": a_estrella, "dijkstra": dijkstra, "voraz": voraz, "bfs": amplitud}
