# Datos — Actividad 3

| Archivo | Registros | Descripción |
|---|---|---|
| `afluencia_transmilenio_muestra.csv` | 47 880 | Una fila por estación × día × hora (95 estaciones, 3 al 30 de agosto de 2026, 5:00–22:00) |
| `estaciones.csv` | 95 | Atributos fijos de cada estación (troncal, tipo, coordenadas, capacidad) |

Se generan con `python -m aprendizaje.generar_dataset` (semilla fija, siempre dan los mismos datos).

## Fuentes reales identificadas

| Fuente | Publicador | Uso en el proyecto |
|---|---|---|
| [Consolidado de salidas sistema troncal por franja horaria](https://datosabiertos.bogota.gov.co/dataset/consolidado-de-salidas-sistema-troncal-por-franja-horaria) | Datos Abiertos Bogotá | Forma del perfil horario (picos AM/PM) y diferencia hábil / fin de semana |
| [Transmilenio Trunk Stations](https://bogota.tumidata.org/dataset/transmilenio_trunk_stations_bogota) | TransMilenio S.A. (catálogo TUMI Data Hub) | Coordenadas, troncal y tipo de estación |
| [GTFS TransMilenio y SITP](https://opennetzero.org/dataset/bogota-colombia-transmilenio-and-sitp-transit-services-general-transit-feed-specification) | Banco Mundial (Open Net Zero) | Horario de operación y frecuencias para estimar la capacidad |
| [Flujo de usuarios de TransMilenio](https://bogota.gov.co/mi-ciudad/movilidad/transmilenio/flujo-de-usuarios-de-transmilenio-en-tiempo-real) | Alcaldía de Bogotá | Confirma que la afluencia se mide por estación |
| `base_conocimiento.py` | Este repositorio (Actividad 2) | Las 95 estaciones del modelo |

Ninguna publica la **afluencia por estación y por hora** con un nivel de congestión, así que se construyó una **muestra** con la misma estructura (lo permite el enunciado).

## Diccionario

| Columna | Tipo | Rol |
|---|---|---|
| `fecha` | AAAA-MM-DD | partición entrenamiento / prueba |
| `dia_semana` (0 = lunes), `es_fin_semana`, `es_festivo`, `hora` (5–22) | enteros | predictoras |
| `nombre_dia`, `franja` | texto | informativas (`franja` se usa en el cálculo ID3) |
| `estacion` | texto | identificador |
| `troncal`, `tipo_estacion` (portal / transbordo / sencilla) | categóricas | predictoras |
| `num_troncales`, `latitud`, `longitud`, `distancia_centro_km` | numéricas | predictoras |
| `lluvia_mm`, `evento_masivo` | numéricas | predictoras |
| `capacidad_hora` | entero | auxiliar |
| `validaciones`, `indice_ocupacion` | numéricas | **excluidas** (definen la clase: fuga de información) |
| `nivel_congestion` | baja (< 0,45) · media (0,45–0,75) · alta (≥ 0,75) | **clase objetivo** |

> Datos de muestra para fines académicos: no son mediciones reales de TransMilenio.
