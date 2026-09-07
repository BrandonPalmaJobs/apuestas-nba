"""
Corre TODOS los reportes nocturnos de inversionistas: para cada uno de
los 4 niveles ($1,000/$3,000/$5,000/$10,000), arma y manda a CADA
inversionista registrado su correo personalizado con TODAS las apuestas/
movimientos del dia de hoy (hora CDMX) y su saldo actualizado. Si un
inversionista no tuvo ningun movimiento hoy, igual se le manda el correo
(dice "no se registro ningun movimiento hoy").

Lee el historial de TODOS los inversionistas de un nivel en UNA sola
llamada a la API (send_daily_reports_batch) en vez de una llamada por
persona - pensado para escalar a ~100 usuarios sin agotar la cuota de
Google Sheets.

Pensado para correr una vez al dia via GitHub Actions a las 11:50pm hora
de Mexico (ver .github/workflows/investor_reports.yml) - tambien se
puede correr a mano.

Uso:
    python nba_send_investor_reports.py
"""

import os
import sys

import nba_investor_emails as ie
import nba_investors as inv


def main():
    creds_path = os.environ.get("GOOGLE_CREDENTIALS_JSON_PATH")
    gmail_address = os.environ.get("GMAIL_ADDRESS")
    gmail_app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not (creds_path and gmail_address and gmail_app_password):
        print("Faltan variables de entorno: GOOGLE_CREDENTIALS_JSON_PATH, GMAIL_ADDRESS, "
              "GMAIL_APP_PASSWORD", file=sys.stderr)
        sys.exit(1)

    gc = inv.get_client(creds_path)
    fecha = inv.cdmx_today()
    print(f"Enviando reportes del {fecha} (hora CDMX)...")

    total_ok = total_fail = total_sin_correo = 0
    for tier in inv.TIER_SHEET_NAMES:
        try:
            investors = inv.list_investors(gc, tier)
        except Exception as e:
            print(f"Nivel ${tier:,}: ERROR leyendo el Sheet - {e}", file=sys.stderr)
            continue

        con_correo = [i for i in investors if i["correo"]]
        sin_correo = [i for i in investors if not i["correo"]]
        for i in sin_correo:
            print(f"  [omitido] {i['nombre']}: sin correo registrado")
        total_sin_correo += len(sin_correo)

        print(f"\nNivel ${tier:,}: {len(con_correo)} inversionista(s) con correo")
        if not con_correo:
            continue
        try:
            resultados = ie.send_daily_reports_batch(gc, tier, con_correo, gmail_address, gmail_app_password,
                                                       fecha=fecha)
        except Exception as e:
            print(f"  ERROR leyendo el historial en lote de este nivel: {e}", file=sys.stderr)
            total_fail += len(con_correo)
            continue
        for nombre, ok, msg in resultados:
            print(f"  {nombre}: {msg}")
            if ok:
                total_ok += 1
            else:
                total_fail += 1

    print(f"\nListo: {total_ok} enviados, {total_fail} fallidos, {total_sin_correo} sin correo registrado.")
    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
