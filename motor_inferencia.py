"""
motor_inferencia.py
===================
Motor de inferencia con ENCADENAMIENTO HACIA ADELANTE (forward chaining)
sobre cláusulas de Horn con variables (Benítez, 2014, caps. 2 y 3).

Ciclo reconocer-actuar:
    1. Equiparación: buscar sustituciones θ que hagan verdaderas todas las
       premisas de una regla con los hechos de la memoria de trabajo.
    2. Resolución de conflictos: se disparan todas las reglas en orden
       (estrategia de refracción: un hecho ya conocido no se vuelve a añadir).
    3. Acción: añadir conclusión·θ a la memoria de trabajo.
    El ciclo termina al alcanzar un punto fijo (ningún hecho nuevo).

Además guarda la JUSTIFICACIÓN de cada hecho inferido, lo que permite
explicar "por qué" el sistema sabe algo (módulo de explicación).
"""
from collections import defaultdict


def es_variable(t):
    return isinstance(t, str) and t.startswith("?")


def unificar(patron, hecho, theta):
    """Unifica un patrón (con variables) con un hecho base.

    Devuelve la sustitución θ extendida, o None si no unifican.
    """
    if len(patron) != len(hecho):
        return None
    theta = dict(theta)
    for p, h in zip(patron, hecho):
        if es_variable(p):
            if p in theta:
                if theta[p] != h:
                    return None
            else:
                theta[p] = h
        elif p != h:
            return None
    return theta


def sustituir(patron, theta):
    return tuple(theta.get(t, t) if es_variable(t) else t for t in patron)


class MotorInferencia:
    def __init__(self, hechos, reglas):
        self.reglas = reglas
        self.hechos = set()
        # índices: predicado -> hechos ; (predicado, primer arg) -> hechos
        self._por_pred = defaultdict(set)
        self._por_pred_arg = defaultdict(set)
        self.justificacion = {}   # hecho -> (nombre_regla, [premisas]) ; None si es hecho base
        self.ciclos = 0
        self.disparos = 0
        for h in hechos:
            self._agregar(h, None)

    # ------------------------------------------------------------------ BC
    def _agregar(self, hecho, porque):
        if hecho in self.hechos:
            return False
        self.hechos.add(hecho)
        self._por_pred[hecho[0]].add(hecho)
        if len(hecho) > 1:
            self._por_pred_arg[(hecho[0], hecho[1])].add(hecho)
        self.justificacion[hecho] = porque
        return True

    def afirmar(self, hecho):
        """Añade un hecho base (p. ej. cerrada('Calle 26'))."""
        return self._agregar(tuple(hecho), None)

    def _candidatos(self, patron, theta):
        """Recupera hechos candidatos usando el índice más selectivo."""
        pred = patron[0]
        if len(patron) > 1:
            primero = patron[1]
            if es_variable(primero):
                primero = theta.get(primero, primero)
            if not es_variable(primero):
                return list(self._por_pred_arg.get((pred, primero), ()))
        return list(self._por_pred.get(pred, ()))

    # --------------------------------------------------------- equiparación
    def _satisfacer(self, premisas, theta, usados):
        """Genera todas las sustituciones que satisfacen la lista de premisas."""
        if not premisas:
            yield theta, usados
            return
        primera, resto = premisas[0], premisas[1:]

        if primera[0] == "distinto":                      # predicado incorporado
            a, b = sustituir(primera[1:], theta)
            if a != b:
                yield from self._satisfacer(resto, theta, usados)
            return

        if primera[0] == "no":                            # negación por fallo
            objetivo = sustituir(primera[1], theta)
            if not any(unificar(objetivo, h, {}) is not None
                       for h in self._candidatos(objetivo, {})):
                yield from self._satisfacer(resto, theta, usados)
            return

        for hecho in self._candidatos(primera, theta):
            nuevo = unificar(primera, hecho, theta)
            if nuevo is not None:
                yield from self._satisfacer(resto, nuevo, usados + [hecho])

    # ------------------------------------------------ encadenamiento adelante
    def inferir(self):
        """Ejecuta el ciclo reconocer-actuar hasta el punto fijo.

        Retorna el número de hechos nuevos deducidos.
        """
        antes = len(self.hechos)
        cambio = True
        while cambio:
            cambio = False
            self.ciclos += 1
            for nombre, premisas, conclusion in self.reglas:
                nuevos = []
                for theta, usados in self._satisfacer(premisas, {}, []):
                    nuevos.append((sustituir(conclusion, theta), usados))
                for hecho, usados in nuevos:
                    if self._agregar(hecho, (nombre, usados)):
                        self.disparos += 1
                        cambio = True
        return len(self.hechos) - antes

    # --------------------------------------------------------------- consulta
    def consultar(self, patron):
        """Devuelve las sustituciones que hacen verdadero el patrón.

        Ej.: consultar(('transbordo', '?x', 'NQS', '?l')) -> [{'?x': 'Ricaurte', '?l': 'Américas'}, ...]
        """
        patron = tuple(patron)
        return [t for h in self._candidatos(patron, {})
                if (t := unificar(patron, h, {})) is not None]

    def es_verdad(self, hecho):
        return tuple(hecho) in self.hechos

    # ------------------------------------------------------------ explicación
    def explicar(self, hecho, nivel=0, max_nivel=4):
        """Traza de por qué un hecho es verdadero (árbol de justificación)."""
        hecho = tuple(hecho)
        sangria = "   " * nivel
        if hecho not in self.hechos:
            return [f"{sangria}✗ {formatear(hecho)} NO se puede deducir de la base de conocimiento"]
        porque = self.justificacion[hecho]
        if porque is None:
            return [f"{sangria}• {formatear(hecho)}   [hecho de la base]"]
        regla, premisas = porque
        lineas = [f"{sangria}• {formatear(hecho)}   [por regla {regla}]"]
        if nivel < max_nivel:
            for p in premisas:
                lineas.extend(self.explicar(p, nivel + 1, max_nivel))
        else:
            lineas.append(f"{sangria}   …")
        return lineas

    def estadisticas(self):
        conteo = defaultdict(int)
        for h in self.hechos:
            conteo[h[0]] += 1
        return dict(sorted(conteo.items()))


def formatear(hecho):
    pred, *args = hecho
    return f"{pred}(" + ", ".join(str(a) for a in args) + ")"
