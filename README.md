# Sistema inteligente de rutas — TransMilenio (modelo)

Proyecto de Inteligencia Artificial sobre el sistema de transporte masivo de Bogotá:

- **Actividad 2 — Búsqueda y sistemas basados en reglas:** a partir de una **base de conocimiento escrita en reglas lógicas**, encuentra la **mejor ruta** entre un punto A y un punto B. Usa un **motor de inferencia con encadenamiento hacia adelante** y la **búsqueda heurística A\***.
- **Actividad 3 — Aprendizaje supervisado:** un **árbol de decisión** aprende de datos históricos a predecir el **nivel de congestión** (baja, media, alta) de cada estación según la hora, el día, la lluvia y los eventos masivos, y se integra al buscador de rutas. Ver la [sección 4](#4-actividad-3--aprendizaje-supervisado-predicción-de-congestión).

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
| `docs/Informe_de_pruebas.pdf` | Documento PDF con las pruebas realizadas (Actividad 2) |
| `docs/generar_informe_pruebas.py` | Regenera el PDF ejecutando todas las pruebas (Actividad 2) |

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
- El buscador de rutas (Actividad 2) **no necesita librerías externas**. La Actividad 3 usa `numpy`, `pandas`, `scikit-learn` y `matplotlib`; `reportlab` solo hace falta para regenerar los PDF.

```bash
git clone <URL-DEL-REPOSITORIO>
cd ruta_inteligente
pip install -r requirements.txt      # obligatorio para la Actividad 3
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
python -m unittest -v                  # 34 pruebas unitarias (16 de la Act. 2 + 18 de la Act. 3)
python docs/generar_informe_pruebas.py # regenera docs/Informe_de_pruebas.pdf
```

---

## 4. Actividad 3 — Aprendizaje supervisado: predicción de congestión

Basado en Palma Méndez, J. T. (2008). *Inteligencia artificial: métodos, técnicas y aplicaciones*, cap. 17 (Aprendizaje de árboles y reglas de decisión).

**Problema:** clasificación en 3 clases. Dada una estación, un día y una hora (más lluvia y eventos), predecir si la congestión será **baja**, **media** o **alta**.

### Entregables

| # | Entregable | Ubicación |
|---|---|---|
| 1 | Archivos de fuentes de datos | [`datos/afluencia_transmilenio_muestra.csv`](datos/afluencia_transmilenio_muestra.csv) (47 880 registros) y [`datos/estaciones.csv`](datos/estaciones.csv) |
| 2 | Código fuente en Python | [`aprendizaje/`](aprendizaje/) |
| 3 | Descripción de los datos | [`docs/Descripcion_de_los_datos.pdf`](docs/Descripcion_de_los_datos.pdf) y [`datos/README.md`](datos/README.md) |
| 4 | Pruebas del componente | [`docs/Informe_de_pruebas_actividad3.pdf`](docs/Informe_de_pruebas_actividad3.pdf) y [`tests/test_aprendizaje.py`](tests/test_aprendizaje.py) |
| 5 | Video | *(enlace al video)* |

### Estructura

| Archivo | Contenido |
|---|---|
| `aprendizaje/generar_dataset.py` | Construye el dataset de muestra (95 estaciones × 28 días × 18 horas) |
| `aprendizaje/preparar_datos.py` | Carga y valida los datos, elige las variables predictoras y divide por fecha |
| `aprendizaje/id3.py` | Entropía y ganancia de información calculadas a mano (algoritmo ID3) |
| `aprendizaje/entrenar.py` | Entrena y compara 4 modelos, poda el árbol con validación cruzada, extrae reglas y guarda el modelo |
| `aprendizaje/predecir.py` | Predice la congestión y explica la regla que se aplicó |
| `aprendizaje/resultados/` | Métricas, matrices de confusión, árbol, importancia de variables y reglas (`reglas_arbol.txt`) |
| `aprendizaje/modelos/arbol_congestion.joblib` | Árbol de decisión entrenado |
| `tests/test_aprendizaje.py` | 18 pruebas unitarias |
| `docs/generar_documentos_actividad3.py` | Regenera los dos PDF ejecutando el código |

### Comandos

```bash
pip install -r requirements.txt

python -m aprendizaje.generar_dataset          # 1. genera datos/afluencia_transmilenio_muestra.csv
python -m aprendizaje.id3                      # 2. ganancia de información de cada atributo (ID3)
python -m aprendizaje.entrenar                 # 3. entrena, compara y guarda el modelo (~30 s)

# 4. predicciones explicadas
python -m aprendizaje.predecir --estacion "Portal del Norte" --dia lunes --hora 6
python -m aprendizaje.predecir --estacion "Movistar Arena" --dia sabado --hora 20 --evento
python -m aprendizaje.predecir --estacion "Calle 26" --dia martes          # perfil de todo el día

# 5. integración con el buscador de rutas de la Actividad 2
python main.py -o "Portal El Dorado" -d "Portal Américas" --dia lunes --hora 7

# 6. pruebas y documentos
python -m unittest -v
python docs/generar_documentos_actividad3.py
```

### Resultados (semana 4, que el modelo nunca vio)

| Modelo | Exactitud | F1 macro | Recall clase alta |
|---|---|---|---|
| Línea base (clase más frecuente) | 0.610 | 0.253 | 0.000 |
| **Árbol de decisión (entropía)** | **0.857** | **0.814** | **0.839** |
| Random Forest (200 árboles) | 0.866 | 0.825 | 0.833 |
| Regresión logística | 0.616 | 0.304 | 0.026 |

- El árbol queda casi empatado con Random Forest y, a diferencia de este, **se puede leer como reglas SI–ENTONCES**.
- La regresión logística falla porque la congestión no crece linealmente con la hora (hay dos picos).
- `validaciones` e `indice_ocupacion` **no se usan como entradas** porque definen la clase (fuga de información).
- La división es **temporal**: se entrena con las semanas 1–3 y se prueba con la semana 4.

> **Aviso sobre los datos:** las fuentes abiertas reales (salidas del sistema troncal por franja horaria, geografía de estaciones, GTFS) no publican la afluencia por estación y por hora con un nivel de congestión. Por eso, como permite el enunciado, se construyó una **muestra** con la misma estructura, reproducible con semilla fija. Para usar datos reales basta con reemplazar el CSV y volver a entrenar. Detalles en `docs/Descripcion_de_los_datos.pdf`.

