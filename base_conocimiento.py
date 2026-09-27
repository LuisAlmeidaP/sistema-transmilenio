"""
base_conocimiento.py
====================
Base de conocimiento (BC) del sistema inteligente de rutas.

Contiene dos partes, siguiendo la separación clásica de un sistema basado
en reglas (Benítez, 2014, cap. 3):

1. HECHOS  (conocimiento declarativo):  estacion/3, pertenece/2, siguiente/3
2. REGLAS  (conocimiento procedimental): cláusulas de Horn  premisas -> conclusión

Modelo: red troncal SIMPLIFICADA de TransMilenio (Bogotá). Las coordenadas
son aproximadas y los tiempos se estiman a partir de la distancia; es un
modelo académico, no la operación real. Para usar otra ciudad basta con
reemplazar el diccionario LINEAS.

Notación: las variables en las reglas empiezan con '?'  (ej. ?x, ?linea).
"""

# ---------------------------------------------------------------------------
# 1. Datos de la red: cada línea troncal es una secuencia ordenada de
#    estaciones (nombre, latitud, longitud).
# ---------------------------------------------------------------------------
LINEAS = {
    "Caracas-Autonorte": [
        ("Portal del Norte", 4.7546, -74.0462),
        ("Toberín", 4.7466, -74.0470),
        ("Calle 161", 4.7392, -74.0480),
        ("Mazurén", 4.7318, -74.0487),
        ("Calle 146", 4.7240, -74.0495),
        ("Alcalá", 4.7165, -74.0505),
        ("Prado", 4.7090, -74.0519),
        ("Calle 127", 4.7020, -74.0530),
        ("Pepe Sierra", 4.6950, -74.0540),
        ("Calle 106", 4.6890, -74.0555),
        ("Calle 100", 4.6840, -74.0565),
        ("Virrey", 4.6760, -74.0575),
        ("Calle 85", 4.6700, -74.0588),
        ("Héroes", 4.6660, -74.0600),
        ("Calle 76", 4.6620, -74.0620),
        ("Calle 72", 4.6570, -74.0630),
        ("Flores", 4.6520, -74.0640),
        ("Calle 63", 4.6470, -74.0655),
        ("Calle 57", 4.6420, -74.0665),
        ("Marly", 4.6370, -74.0675),
        ("Calle 45", 4.6320, -74.0685),
        ("Av. 39", 4.6270, -74.0695),
        ("Calle 34", 4.6210, -74.0705),
        ("Calle 26", 4.6150, -74.0725),
        ("Calle 22", 4.6100, -74.0745),
        ("Calle 19", 4.6060, -74.0760),
        ("Av. Jiménez", 4.6020, -74.0775),
        ("Tercer Milenio", 4.5975, -74.0815),
        ("Hortúa", 4.5920, -74.0840),
        ("Hospital", 4.5880, -74.0870),
        ("Tygua-San José", 4.5830, -74.0905),
        ("Olaya", 4.5780, -74.0940),
        ("Quiroga", 4.5720, -74.0980),
        ("Calle 40 Sur", 4.5670, -74.1010),
    ],
    "Calle 80": [
        ("Portal 80", 4.7110, -74.1120),
        ("Quirigua", 4.7095, -74.1100),
        ("Carrera 90", 4.7080, -74.1080),
        ("Av. Ciudad de Cali", 4.7050, -74.1050),
        ("Granja-Carrera 77", 4.7020, -74.1020),
        ("Minuto de Dios", 4.6990, -74.0990),
        ("Boyacá", 4.6960, -74.0960),
        ("Ferias", 4.6920, -74.0920),
        ("Av. 68", 4.6880, -74.0880),
        ("Carrera 53", 4.6840, -74.0840),
        ("Carrera 47", 4.6800, -74.0800),
        ("Escuela Militar", 4.6760, -74.0760),
        ("Polo", 4.6720, -74.0700),
        ("Héroes", 4.6660, -74.0600),
    ],
    "NQS": [
        ("Polo", 4.6720, -74.0700),
        ("NQS Calle 75", 4.6690, -74.0730),
        ("Av. Chile", 4.6620, -74.0750),
        ("Simón Bolívar", 4.6560, -74.0770),
        ("Movistar Arena", 4.6510, -74.0780),
        ("El Campín", 4.6460, -74.0780),
        ("U. Nacional", 4.6400, -74.0790),
        ("Av. El Dorado", 4.6340, -74.0800),
        ("CAD", 4.6280, -74.0830),
        ("Paloquemao", 4.6170, -74.0850),
        ("Ricaurte", 4.6100, -74.0900),
        ("Comuneros", 4.6050, -74.0960),
        ("Santa Isabel", 4.6000, -74.1000),
        ("Sena", 4.5950, -74.1050),
        ("NQS Calle 30 Sur", 4.5900, -74.1100),
        ("NQS Calle 38A Sur", 4.5860, -74.1150),
        ("General Santander", 4.5820, -74.1210),
        ("Alquería", 4.5870, -74.1300),
        ("Venecia", 4.5950, -74.1400),
        ("Portal Sur", 4.5960, -74.1520),
    ],
    "Américas": [
        ("Portal Américas", 4.6310, -74.1800),
        ("Patio Bonito", 4.6340, -74.1720),
        ("Biblioteca Tintal", 4.6310, -74.1650),
        ("Transversal 86", 4.6290, -74.1590),
        ("Banderas", 4.6300, -74.1520),
        ("Mandalay", 4.6330, -74.1460),
        ("Mundo Aventura", 4.6350, -74.1400),
        ("Marsella", 4.6330, -74.1340),
        ("Pradera", 4.6310, -74.1260),
        ("Distrito Grafiti", 4.6290, -74.1180),
        ("Puente Aranda", 4.6260, -74.1120),
        ("Carrera 43", 4.6230, -74.1060),
        ("Zona Industrial", 4.6200, -74.1030),
        ("CDS-Carrera 32", 4.6150, -74.0960),
        ("Ricaurte", 4.6100, -74.0900),
        ("San Façon-Carrera 22", 4.6070, -74.0870),
        ("De La Sabana", 4.6050, -74.0830),
        ("Av. Jiménez", 4.6020, -74.0775),
    ],
    "Calle 26": [
        ("Portal El Dorado", 4.6810, -74.1200),
        ("Modelia", 4.6720, -74.1150),
        ("Normandía", 4.6660, -74.1090),
        ("Av. Rojas", 4.6610, -74.1020),
        ("El Tiempo", 4.6570, -74.0980),
        ("Salitre-El Greco", 4.6530, -74.0940),
        ("CAN", 4.6480, -74.0890),
        ("Gobernación", 4.6440, -74.0860),
        ("Quinta Paredes", 4.6400, -74.0840),
        ("Corferias", 4.6350, -74.0830),
        ("Ciudad Universitaria", 4.6310, -74.0820),
        ("Concejo de Bogotá", 4.6280, -74.0800),
        ("Centro Memoria", 4.6230, -74.0770),
        ("Calle 26", 4.6150, -74.0725),
    ],
}

# Parámetros del modelo (unidades entre corchetes)
VELOCIDAD_BUS_KMH = 25.0        # velocidad comercial promedio [km/h]
VELOCIDAD_MAX_KMH = 25.0        # cota de velocidad para la heurística [km/h]; debe ser >= VELOCIDAD_BUS_KMH
                                # para que h(n) sea admisible. Igualarlas da la heurística más informada.
TIEMPO_PARADA_MIN = 0.5         # detención en cada estación [min]
PENALIZACION_TRANSBORDO_MIN = 5.0  # caminar + esperar al cambiar de troncal [min]


# ---------------------------------------------------------------------------
# 2. HECHOS: se generan a partir de LINEAS como tuplas (predicado, args...)
# ---------------------------------------------------------------------------
def generar_hechos(lineas=LINEAS):
    """Traduce la red a hechos lógicos de primer orden.

    estacion(Nombre, Lat, Lon)       -> ('estacion', nombre, lat, lon)
    pertenece(Estacion, Linea)       -> ('pertenece', est, linea)
    siguiente(A, B, Linea)           -> ('siguiente', a, b, linea)
    """
    hechos = set()
    for linea, estaciones in lineas.items():
        for nombre, lat, lon in estaciones:
            hechos.add(("estacion", nombre, lat, lon))
            hechos.add(("pertenece", nombre, linea))
        for (a, _, _), (b, _, _) in zip(estaciones, estaciones[1:]):
            hechos.add(("siguiente", a, b, linea))
    return hechos


# ---------------------------------------------------------------------------
# 3. REGLAS: cláusulas de Horn. Cada regla = (nombre, [premisas], conclusión)
#    Una premisa ('distinto', ?a, ?b) es un predicado incorporado (built-in).
#    Una premisa ('no', patrón) es negación por fallo (closed world assumption).
# ---------------------------------------------------------------------------
REGLAS = [
    # R1  siguiente(X,Y,L)                    -> conecta(X,Y,L)
    ("R1_sentido_ida",
     [("siguiente", "?x", "?y", "?l")],
     ("conecta", "?x", "?y", "?l")),

    # R2  siguiente(X,Y,L)                    -> conecta(Y,X,L)   (las troncales son bidireccionales)
    ("R2_sentido_regreso",
     [("siguiente", "?x", "?y", "?l")],
     ("conecta", "?y", "?x", "?l")),

    # R3  pertenece(X,L1) ∧ pertenece(X,L2) ∧ L1≠L2 -> transbordo(X,L1,L2)
    ("R3_transbordo",
     [("pertenece", "?x", "?l1"), ("pertenece", "?x", "?l2"), ("distinto", "?l1", "?l2")],
     ("transbordo", "?x", "?l1", "?l2")),

    # R4  transbordo(X,L1,L2)                 -> estacion_transbordo(X)
    ("R4_es_estacion_transbordo",
     [("transbordo", "?x", "?l1", "?l2")],
     ("estacion_transbordo", "?x")),

    # R5  conecta(X,Y,L) ∧ ¬cerrada(X) ∧ ¬cerrada(Y) -> tramo_habilitado(X,Y,L)
    ("R5_tramo_habilitado",
     [("conecta", "?x", "?y", "?l"), ("no", ("cerrada", "?x")), ("no", ("cerrada", "?y"))],
     ("tramo_habilitado", "?x", "?y", "?l")),

    # R6  tramo_habilitado(X,Y,L)             -> alcanzable(X,Y)
    ("R6_alcanzable_directo",
     [("tramo_habilitado", "?x", "?y", "?l")],
     ("alcanzable", "?x", "?y")),

    # R7  tramo_habilitado(X,Y,L) ∧ alcanzable(Y,Z) ∧ X≠Z -> alcanzable(X,Z)   (clausura transitiva)
    ("R7_alcanzable_transitivo",
     [("tramo_habilitado", "?x", "?y", "?l"), ("alcanzable", "?y", "?z"), ("distinto", "?x", "?z")],
     ("alcanzable", "?x", "?z")),
]
