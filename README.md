# Sistema inteligente de rutas — TransMilenio (modelo)

Sistema basado en conocimiento que, a partir de una **base de conocimiento escrita en reglas lógicas**, encuentra la **mejor ruta** entre un punto A y un punto B del sistema de transporte masivo. Usa un **motor de inferencia con encadenamiento hacia adelante** y la **búsqueda heurística A\***.

Basado en Benítez, R. (2014). *Inteligencia artificial avanzada*. Editorial UOC: cap. 2 (lógica y representación del conocimiento), cap. 3 (sistemas basados en reglas) y cap. 9 (búsqueda heurística).

---

## 1. Arquitectura

```
┌────────────────────────┐     ┌────────────────────────┐     ┌──────────────────────────┐
│  BASE DE CONOCIMIENTO  │ ──► │  MOTOR DE INFERENCIA   │ ──► │  BÚSQUEDA HEURÍSTICA A*  │
│  hechos + reglas Horn  │     │  encadenamiento hacia  │     │  f(n) = g(n) + h(n)      │
│  base_conocimiento.py  │     │  adelante + explicación│     │  busqueda.py             │
└────────────────────────┘     │  motor_inferencia.py   │     └──────────────────────────┘
                               └────────────────────────┘
```

| Archivo | Contenido |
|---|---|
| `base_conocimiento.py` | Red de 5 troncales y 95 estaciones, parámetros, hechos y las reglas R1–R7 |
| `motor_inferencia.py` | Unificación, encadenamiento hacia adelante, consultas y módulo de explicación |
| `busqueda.py` | Espacio de estados, heurística (distancia haversine) y los algoritmos A\*, Dijkstra, voraz y BFS |
| `sistema.py` | Une los tres componentes |
| `main.py` | Interfaz de línea de comandos y modo interactivo |
| `visualizacion.py` | Dibuja el mapa con la ruta (matplotlib) |
| `tests/test_sistema.py` | 16 pruebas unitarias |
| `docs/Informe_de_pruebas.pdf` | Documento PDF con las pruebas realizadas |
| `docs/generar_informe_pruebas.py` | Regenera el PDF ejecutando todas las pruebas |

### Reglas de la base de conocimiento

```
R1: siguiente(X,Y,L)                               → conecta(X,Y,L)
R2: siguiente(X,Y,L)                               → conecta(Y,X,L)
R3: pertenece(X,L1) ∧ pertenece(X,L2) ∧ L1≠L2      → transbordo(X,L1,L2)
R4: transbordo(X,L1,L2)                            → estacion_transbordo(X)
R5: conecta(X,Y,L) ∧ ¬cerrada(X) ∧ ¬cerrada(Y)     → tramo_habilitado(X,Y,L)
R6: tramo_habilitado(X,Y,L)                        → alcanzable(X,Y)
R7: tramo_habilitado(X,Y,L) ∧ alcanzable(Y,Z) ∧ X≠Z → alcanzable(X,Z)
```

### Heurística

`h(n) = distancia_haversine(n, destino) / v_max · 60` (minutos). Es **admisible** porque ningún bus viaja más rápido que `v_max` ni recorre menos que la distancia en línea recta. También es **consistente**, así que A\* garantiza la ruta de menor tiempo.

> **Aviso:** es un modelo académico simplificado. Las coordenadas son aproximadas y los tiempos se estiman a partir de la distancia (25 km/h, 0,5 min por parada y 5 min por transbordo). Para usar otra ciudad solo hay que editar el diccionario `LINEAS` en `base_conocimiento.py`.

---

## 2. Requisitos e instalación

- Python 3.8 o superior
- El programa principal **no necesita librerías externas**. `matplotlib` solo hace falta para `--graficar`, y `reportlab` solo para regenerar el PDF de pruebas.

```bash
git clone <URL-DEL-REPOSITORIO>
cd ruta_inteligente
pip install -r requirements.txt      # opcional
```

---

## 3. Ejecución

### Modo interactivo
```bash
python main.py
```
Muestra las estaciones numeradas y pide el origen, el destino, el criterio y las estaciones cerradas.

### Línea de comandos

| Comando | Qué hace |
|---|---|
| `python main.py -o "Portal del Norte" -d "Portal Américas"` | Mejor ruta con A\* |
| `python main.py -o "Portal 80" -d "Portal Sur" --comparar` | Compara A\*, Dijkstra, voraz y BFS |
| `python main.py -o "Toberín" -d "Ricaurte" --cerrar "Calle 26"` | Recalcula la ruta sin una estación |
| `python main.py -o "Héroes" -d "Ricaurte" --explicar` | Muestra las reglas que justifican la ruta |
| `python main.py -o "Calle 100" -d "Portal Américas" --cerrar Ricaurte --cerrar "Av. Jiménez" --explicar` | Caso sin solución, con la explicación |
| `python main.py -o "Portal 80" -d "Corferias" --graficar ruta.png` | Guarda un mapa con la ruta |
| `python main.py -o "Toberín" -d "Portal Sur" -c transbordos` | Prioriza tener menos transbordos |
| `python main.py -o "Portal 80" -d "Hospital" -a dijkstra` | Usa otro algoritmo (`astar`, `dijkstra`, `voraz`, `bfs`) |
| `python main.py --listar` | Muestra la base de conocimiento inferida |

Los nombres se aceptan **sin tildes y en minúsculas** (`portal americas`). Si un nombre es ambiguo, el sistema sugiere opciones.

### Ejemplo de salida
```
========================================================================
                RUTA: Portal El Dorado  →  Portal Américas
========================================================================
Criterio de optimización : menor tiempo
Algoritmo de búsqueda    : A*
Inferencia lógica        : alcanzable(Portal El Dorado, Portal Américas) = VERDADERO
------------------------------------------------------------------------
1. Tome la troncal Calle 26 desde Portal El Dorado hasta Calle 26 (13 paradas, ≈28.6 min)
2. Transbordo en Calle 26: cambie de la troncal Calle 26 a la troncal Caracas-Autonorte (+5 min)
3. Tome la troncal Caracas-Autonorte desde Calle 26 hasta Av. Jiménez (3 paradas, ≈5.2 min)
4. Transbordo en Av. Jiménez: cambie de la troncal Caracas-Autonorte a la troncal Américas (+5 min)
5. Tome la troncal Américas desde Av. Jiménez hasta Portal Américas (17 paradas, ≈39.0 min)
------------------------------------------------------------------------
Tiempo estimado          : 82.8 min
Transbordos              : 2
```

### Pruebas
```bash
python -m unittest -v                  # 16 pruebas unitarias
python docs/generar_informe_pruebas.py # regenera docs/Informe_de_pruebas.pdf
```

