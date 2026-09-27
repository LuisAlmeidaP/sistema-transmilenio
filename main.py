#!/usr/bin/env python3
"""
main.py — Interfaz de línea de comandos del sistema inteligente de rutas.

Ejemplos:
    python main.py --origen "Portal del Norte" --destino "Portal Américas"
    python main.py -o "Portal 80" -d "Portal Sur" --comparar
    python main.py -o "Calle 100" -d "Ricaurte" --cerrar "Av. Jiménez" --explicar
    python main.py -o "Toberín" -d "Portal Américas" --criterio transbordos
    python main.py --listar
    python main.py                 # modo interactivo
"""
import argparse
import sys

from motor_inferencia import formatear
from sistema import SistemaRutas

ANCHO = 72


def titulo(texto):
    print("\n" + "=" * ANCHO)
    print(texto.center(ANCHO))
    print("=" * ANCHO)


def mostrar_ruta(sistema, origen, destino, algoritmo="astar", explicar=False, graficar=None):
    resultado, deducible = sistema.ruta(origen, destino, algoritmo)
    o = sistema.red.resolver_nombre(origen)
    d = sistema.red.resolver_nombre(destino)
    titulo(f"RUTA: {o}  →  {d}")
    print(f"Criterio de optimización : {'menor tiempo' if sistema.criterio == 'tiempo' else 'menos transbordos'}")
    print(f"Algoritmo de búsqueda    : {resultado.algoritmo}")
    if sistema.cerradas:
        print(f"Estaciones cerradas      : {', '.join(sistema.cerradas)}")
    print(f"Inferencia lógica        : alcanzable({o}, {d}) = {'VERDADERO' if deducible else 'FALSO'}")
    print("-" * ANCHO)
    for linea in resultado.instrucciones(sistema.red):
        print(linea)
    if resultado.encontrada:
        print("-" * ANCHO)
        print(f"Tiempo estimado          : {resultado.tiempo_real(sistema.red):.1f} min")
        print(f"Estaciones recorridas    : {len(resultado.estaciones)}")
        print(f"Transbordos              : {resultado.transbordos}")
        print(f"Nodos expandidos         : {resultado.nodos_expandidos}")
        print(f"Recorrido                : {' → '.join(resultado.estaciones)}")
    if explicar:
        titulo("EXPLICACIÓN (¿por qué el sistema sabe que hay ruta?)" if deducible
               else "EXPLICACIÓN (¿por qué NO hay ruta?)")
        for l in sistema.motor.explicar(("alcanzable", o, d), max_nivel=3):
            print(l)
        if not deducible and sistema.cerradas:
            print("   La regla R5 no habilita tramos que toquen estaciones cerradas:")
            for c in sistema.cerradas:
                print(f"   • cerrada({c})   [hecho afirmado por el usuario]")
            print("   Sin esos tramos, R6/R7 no logran encadenar el origen con el destino.")
        if resultado.encontrada:
            for a, b in zip(resultado.camino, resultado.camino[1:]):
                if a[0] == b[0]:
                    print()
                    for l in sistema.motor.explicar(("transbordo", a[0], a[1], b[1])):
                        print(l)
    if graficar and resultado.encontrada:
        from visualizacion import dibujar_red
        ruta = dibujar_red(sistema, resultado, graficar)
        print(f"\nMapa guardado en: {ruta}")
    return resultado


def comparar(sistema, origen, destino):
    titulo("COMPARACIÓN DE ALGORITMOS DE BÚSQUEDA")
    print(f"{'Algoritmo':<28}{'Tiempo(min)':>12}{'Estac.':>8}{'Transb.':>9}{'Expand.':>9}{'Óptimo':>8}")
    print("-" * ANCHO)
    filas = []
    for alg in ("astar", "dijkstra", "voraz", "bfs"):
        r, _ = sistema.ruta(origen, destino, alg)
        filas.append(r)
    optimo = min(r.costo_min for r in filas if r.encontrada)
    for r in filas:
        es_opt = "sí" if abs(r.costo_min - optimo) < 1e-6 else "no"
        print(f"{r.algoritmo:<28}{r.tiempo_real(sistema.red):>12.1f}{len(r.estaciones):>8}"
              f"{r.transbordos:>9}{r.nodos_expandidos:>9}{es_opt:>8}")
    return filas


def listar(sistema):
    titulo("BASE DE CONOCIMIENTO")
    print("Hechos por predicado (después de la inferencia):")
    for pred, n in sistema.motor.estadisticas().items():
        print(f"   {pred:<22}{n:>6}")
    print(f"\nCiclos de inferencia: {sistema.motor.ciclos}   Reglas disparadas: {sistema.motor.disparos}")
    print("\nEstaciones de transbordo deducidas (regla R4):")
    print("   " + ", ".join(sistema.estaciones_transbordo()))
    for linea in sistema.lineas():
        est = [t["?e"] for t in sistema.motor.consultar(("pertenece", "?e", linea))]
        print(f"\nTroncal {linea} ({len(est)} estaciones):")
        print("   " + ", ".join(sorted(est)))


def interactivo():
    sistema = SistemaRutas()
    estaciones = sistema.estaciones()
    titulo("SISTEMA INTELIGENTE DE RUTAS — TransMilenio (modelo)")
    for i, e in enumerate(estaciones, 1):
        print(f"{i:>3}. {e:<24}", end="\n" if i % 3 == 0 else "")
    print()

    def pedir(msg):
        while True:
            txt = input(msg).strip()
            if txt.isdigit() and 1 <= int(txt) <= len(estaciones):
                return estaciones[int(txt) - 1]
            try:
                return sistema.red.resolver_nombre(txt)
            except ValueError as e:
                print("  ", e)

    origen = pedir("\nOrigen (número o nombre): ")
    destino = pedir("Destino (número o nombre): ")
    crit = input("Criterio [1] menor tiempo  [2] menos transbordos (Enter = 1): ").strip()
    cerr = input("Estaciones cerradas separadas por coma (Enter = ninguna): ").strip()
    sistema = SistemaRutas(cerradas=[c for c in cerr.split(",") if c.strip()],
                           criterio="transbordos" if crit == "2" else "tiempo")
    mostrar_ruta(sistema, origen, destino, explicar=input("¿Mostrar explicación? (s/N): ").lower() == "s")


def main(argv=None):
    p = argparse.ArgumentParser(description="Sistema inteligente de rutas (reglas lógicas + A*)")
    p.add_argument("-o", "--origen", help="estación de origen (punto A)")
    p.add_argument("-d", "--destino", help="estación de destino (punto B)")
    p.add_argument("-a", "--algoritmo", default="astar", choices=["astar", "dijkstra", "voraz", "bfs"])
    p.add_argument("-c", "--criterio", default="tiempo", choices=["tiempo", "transbordos"])
    p.add_argument("--cerrar", action="append", default=[], metavar="ESTACION",
                   help="marca una estación como cerrada (se puede repetir)")
    p.add_argument("--explicar", action="store_true", help="muestra la cadena de reglas que justifica la ruta")
    p.add_argument("--comparar", action="store_true", help="compara A*, Dijkstra, voraz y BFS")
    p.add_argument("--graficar", metavar="ARCHIVO.png", help="guarda un mapa con la ruta")
    p.add_argument("--listar", action="store_true", help="muestra la base de conocimiento")
    args = p.parse_args(argv)

    if not (args.origen or args.destino or args.listar):
        interactivo()
        return 0
    try:
        sistema = SistemaRutas(cerradas=args.cerrar, criterio=args.criterio)
        if args.listar:
            listar(sistema)
        if args.origen and args.destino:
            mostrar_ruta(sistema, args.origen, args.destino, args.algoritmo, args.explicar, args.graficar)
            if args.comparar:
                comparar(sistema, args.origen, args.destino)
        elif args.origen or args.destino:
            p.error("debe indicar --origen y --destino")
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
