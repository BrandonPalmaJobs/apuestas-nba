"""
Manejo de inversionistas y apuestas para el fondo de apuestas NBA: cada
nivel ($1,000/$3,000/$5,000/$10,000) es un Google Sheet separado (ya
existente en la carpeta de Drive del usuario, ej. "Inversores de
$1,000"). Dentro de cada Sheet:
  - Una pestana "Registro": lista de inversionistas de ese nivel
    (Nombre, Telefono, Correo).
  - Una pestana POR INVERSIONISTA (con su nombre): historial de sus
    apuestas y su saldo acumulado.

Cada inversionista tiene su PROPIO monto fijo por apuesta (no es un pool
compartido) - si dos personas estan en el nivel de $1,000, cada una tiene
su propia apuesta identica (mismo momio), cuentas independientes.

Momios en formato AMERICANO (+150, -170), no decimales ni porcentajes:
    momio positivo: ganancia = monto * (momio / 100)
    momio negativo: ganancia = monto * (100 / abs(momio))

Requiere el mismo secret GOOGLE_CREDENTIALS_JSON ya usado en el proyecto
de MLB (cuenta de servicio de Google) - hay que compartir la carpeta
"apuestas financieras" (o cada Sheet individualmente) con el correo de
esa cuenta de servicio, con permiso de Editor.
"""

from datetime import date, datetime

TIER_SHEET_NAMES = {
    1000: "Inversores de $1,000",
    3000: "Inversores de $3,000",
    5000: "Inversores de $5,000",
    10000: "Inversores de $10,000",
}

REGISTRO_TAB = "Registro"
REGISTRO_HEADERS = ["Nombre", "Telefono", "Correo"]
BET_HEADERS = ["Fecha", "Descripcion", "Monto", "Momio", "Resultado", "Ganancia_Perdida", "Saldo"]


def american_odds_profit(stake, odds, result):
    """Ganancia/perdida en dinero para un momio AMERICANO. result: 'Gano',
    'Perdio' o 'Push' (empate, se regresa el monto sin ganar ni perder)."""
    if result == "Push":
        return 0.0
    if result == "Perdio":
        return -abs(stake)
    if odds > 0:
        return stake * (odds / 100)
    else:
        return stake * (100 / abs(odds))


def get_client(credentials_path):
    import gspread
    return gspread.service_account(filename=credentials_path)


def _open_tier_sheet(gc, tier):
    sheet_name = TIER_SHEET_NAMES.get(tier)
    if not sheet_name:
        raise ValueError(f"Nivel de inversion desconocido: {tier}. Validos: {list(TIER_SHEET_NAMES)}")
    return gc.open(sheet_name)


def _get_or_create_tab(sh, tab_name, headers):
    import gspread
    try:
        ws = sh.worksheet(tab_name)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=tab_name, rows=200, cols=len(headers) + 2)
        ws.append_row(headers)
    return ws


def list_investors(gc, tier):
    sh = _open_tier_sheet(gc, tier)
    ws = _get_or_create_tab(sh, REGISTRO_TAB, REGISTRO_HEADERS)
    rows = ws.get_all_records()
    return [{"nombre": r["Nombre"], "telefono": r.get("Telefono", ""), "correo": r.get("Correo", "")}
            for r in rows if r.get("Nombre")]


def add_investor(gc, tier, nombre, telefono, correo):
    sh = _open_tier_sheet(gc, tier)
    registro = _get_or_create_tab(sh, REGISTRO_TAB, REGISTRO_HEADERS)
    existentes = [r["Nombre"].strip().lower() for r in registro.get_all_records() if r.get("Nombre")]
    if nombre.strip().lower() in existentes:
        raise ValueError(f"'{nombre}' ya esta registrado en el nivel ${tier}.")
    registro.append_row([nombre, telefono, correo])

    tab = _get_or_create_tab(sh, nombre, BET_HEADERS)
    if len(tab.get_all_values()) <= 1:
        tab.append_row([date.today().isoformat(), "Saldo inicial", 0, "", "", 0, tier])
    return True


def get_investor_balance(gc, tier, nombre):
    sh = _open_tier_sheet(gc, tier)
    tab = sh.worksheet(nombre)
    rows = tab.get_all_records()
    if not rows:
        return tier
    return float(rows[-1]["Saldo"])


def log_bet(gc, tier, nombre, descripcion, monto, momio, resultado, fecha=None):
    """Agrega una apuesta resuelta (resultado ya conocido: 'Gano'/'Perdio'/
    'Push') al historial del inversionista y actualiza su saldo
    acumulado. Regresa la fila agregada (dict)."""
    if resultado not in ("Gano", "Perdio", "Push"):
        raise ValueError("resultado debe ser 'Gano', 'Perdio' o 'Push'")
    fecha = fecha or date.today().isoformat()

    sh = _open_tier_sheet(gc, tier)
    tab = _get_or_create_tab(sh, nombre, BET_HEADERS)
    saldo_previo = get_investor_balance(gc, tier, nombre)

    ganancia = american_odds_profit(monto, momio, resultado)
    saldo_nuevo = saldo_previo + ganancia

    momio_str = f"+{momio}" if momio > 0 else str(momio)
    row = [fecha, descripcion, monto, momio_str, resultado, round(ganancia, 2), round(saldo_nuevo, 2)]
    tab.append_row(row)
    return dict(zip(BET_HEADERS, row))


def get_bets_for_date(gc, tier, nombre, fecha):
    """Todas las apuestas de un inversionista logueadas en `fecha`
    (YYYY-MM-DD) - para el reporte por correo."""
    sh = _open_tier_sheet(gc, tier)
    try:
        tab = sh.worksheet(nombre)
    except Exception:
        return []
    rows = tab.get_all_records()
    return [r for r in rows if str(r.get("Fecha")) == fecha and r.get("Descripcion") != "Saldo inicial"]


def all_investors_all_tiers(gc):
    """{tier: [ {nombre, telefono, correo}, ... ]} para los 4 niveles."""
    return {tier: list_investors(gc, tier) for tier in TIER_SHEET_NAMES}
