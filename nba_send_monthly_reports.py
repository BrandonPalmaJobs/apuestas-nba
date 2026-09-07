"""
Manda la RECOPILACION MENSUAL a todos los inversionistas registrados (los
4 niveles): todos los movimientos del mes que acaba de terminar, saldo al
inicio/fin de ese mes, y su record/efectividad de ese mes.

Pensado para correr el dia 1 de cada mes, justo despues de medianoche
hora CDMX, via GitHub Actions (ver
.github/workflows/investor_monthly_reports.yml) - tambien se puede
correr a mano.

Uso:
    python nba_send_monthly_reports.py
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
    print("Enviando recopilaciones mensuales (mes que acaba de terminar, hora CDMX)...")

    total_ok = total_fail = total_sin_correo = 0
    for tier in inv.TIER_SHEET_NAMES:
        try:
            investors = inv.list_investors(gc, tier)
        except Exception as e:
            print(f"Nivel ${tier:,}: ERROR leyendo el Sheet - {e}", file=sys.stderr)
            continue
        print(f"\nNivel ${tier:,}: {len(investors)} inversionista(s)")
        for i in investors:
            if not i["correo"]:
                print(f"  [omitido] {i['nombre']}: sin correo registrado")
                total_sin_correo += 1
                continue
            ok, msg = ie.send_monthly_report(gc, tier, i["nombre"], i["correo"], gmail_address, gmail_app_password)
            print(f"  {i['nombre']}: {msg}")
            if ok:
                total_ok += 1
            else:
                total_fail += 1

    print(f"\nListo: {total_ok} enviados, {total_fail} fallidos, {total_sin_correo} sin correo registrado.")
    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
