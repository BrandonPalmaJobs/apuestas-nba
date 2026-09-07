"""
Prueba rapida de que las fuentes de datos (ESPN + pbpstats.com) responden
desde ESTA computadora, antes de confiar en el resto del sistema.

Uso:
    python probar_conexion.py
"""

import sys

import nba_data as n


def probar(nombre, fn):
    print(f"Probando: {nombre}...", end=" ", flush=True)
    try:
        resultado = fn()
        print(f"OK ({resultado})")
        return True
    except Exception as e:
        print(f"FALLO -> {type(e).__name__}: {e}")
        return False


def main():
    print("=" * 60)
    print("Prueba de conexion a las fuentes de datos NBA")
    print("=" * 60)
    season = n.current_nba_season()
    print(f"Temporada actual detectada: {season - 1}-{str(season)[2:]}\n")

    ok = []
    lakers = n.resolve_team("Lakers")
    # Se prueba con la temporada 2024-25 (ya jugada completa) en vez de la
    # actual, para no depender de que la temporada nueva ya haya arrancado.
    ok.append(probar(
        "ESPN - calendario/juegos de un equipo (Lakers, 2024-25)",
        lambda: f"{len(n.espn_team_event_ids(lakers['espn_id'], season=2025))} juegos encontrados",
    ))
    ok.append(probar(
        "ESPN - boxscore de un juego real (2024-25)",
        lambda: (lambda ids: f"boxscore OK para el juego {ids[0]}" if ids else "sin juegos")(
            n.espn_team_event_ids(lakers["espn_id"], season=2025)[:1]
        ),
    ))
    ok.append(probar(
        "pbpstats.com - roster de un equipo (Lakers, 2024-25)",
        lambda: f"{len(n.pbp_team_roster(lakers['nba_id'], season=2025))} jugadores",
    ))

    print("\n" + "=" * 60)
    if all(ok):
        print("TODO BIEN - las fuentes de datos responden desde esta compu.")
        print("Ya se puede confiar en el resto del sistema (nba_report.py).")
        print("\nProbando el reporte completo puede tardar 1-3 minutos (recorre")
        print("varios juegos por equipo) y a veces ESPN limita solicitudes si se")
        print("corre muchas veces seguidas en poco tiempo - si un reporte falla")
        print("con error 403, espera unos minutos antes de reintentar.")
    else:
        print("ALGO FALLO - revisa los mensajes de arriba.")
        print("Si fallo con timeout/conexion, puede ser tu red bloqueando estos")
        print("sitios. Si fallo con error 403, es probable que ESPN este")
        print("limitando temporalmente las solicitudes - espera unos minutos e")
        print("intenta de nuevo.")
    print("=" * 60)
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
