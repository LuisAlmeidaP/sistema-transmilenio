"""
visualizacion.py — Dibuja la red y resalta la ruta encontrada (requiere matplotlib).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import base_conocimiento as bc

COLORES = {
    "Caracas-Autonorte": "#d7263d",
    "Calle 80": "#f4a261",
    "NQS": "#2a9d8f",
    "Américas": "#6a4c93",
    "Calle 26": "#1d70b8",
}


def dibujar_red(sistema, resultado=None, archivo="ruta.png"):
    fig, ax = plt.subplots(figsize=(9, 9))
    for linea, estaciones in bc.LINEAS.items():
        xs = [lon for _, _, lon in estaciones]
        ys = [lat for _, lat, _ in estaciones]
        ax.plot(xs, ys, "-", color=COLORES.get(linea, "gray"), lw=3, alpha=0.45, label=linea)
        ax.plot(xs, ys, "o", color=COLORES.get(linea, "gray"), ms=3, alpha=0.6)
    for e in sistema.estaciones_transbordo():
        lat, lon = sistema.red.coords[e]
        ax.plot(lon, lat, "s", color="black", ms=6)
        ax.annotate(e, (lon, lat), xytext=(5, 4), textcoords="offset points", fontsize=8)
    for e in sistema.cerradas:
        lat, lon = sistema.red.coords[e]
        ax.plot(lon, lat, "X", color="red", ms=12)
    if resultado is not None and resultado.encontrada:
        est = resultado.estaciones
        xs = [sistema.red.coords[e][1] for e in est]
        ys = [sistema.red.coords[e][0] for e in est]
        ax.plot(xs, ys, "-", color="black", lw=2.5, label="Ruta encontrada")
        for nombre, x, y, marca in ((est[0], xs[0], ys[0], "A"), (est[-1], xs[-1], ys[-1], "B")):
            ax.plot(x, y, "o", color="gold", mec="black", ms=14)
            ax.annotate(f"{marca}: {nombre}", (x, y), xytext=(8, -12), textcoords="offset points",
                        fontsize=10, weight="bold")
        ax.set_title(f"{est[0]} → {est[-1]}  |  {resultado.tiempo_real(sistema.red):.1f} min, "
                     f"{resultado.transbordos} transbordo(s)")
    else:
        ax.set_title("Red troncal modelada")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(archivo, dpi=130)
    plt.close(fig)
    return archivo
