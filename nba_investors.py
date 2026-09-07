"""
Manejo de inversionistas y apuestas para el fondo de apuestas NBA: cada
nivel ($1,000/$3,000/$5,000/$10,000) es un Google Sheet separado (ya
existente en la carpeta de Drive del usuario, ej. "Inversores de
$1,000"). Dentro de cada Sheet:
  - Una pestana "Registro": lista de inversionistas de ese nivel
    (Nombre, Telefono, Correo) - para saber a quien mandarle el correo
    nocturno mas adelante.
  - Una pestana POR INVERSIONISTA (con su nombre, ej. "Mario Palma"):
    historial de sus apuestas y su saldo acumulado, en el MISMO formato
    ya creado a mano por el usuario: columna A vacia, encabezados en la
    fila 2 empezando en la columna B (Partido al que se aposto | Fecha
    en la que se aposto | Apuesta que se realizo | Momio en la que se
    tomo | Inversion actual | Ganada / Perdida | Inversion despues de
    apuesta).

Cada inversionista tiene su PROPIO monto fijo por apuesta (no es un pool
compartido) - si dos personas estan en el nivel de $1,000, cada una tiene
su propia apuesta identica (mismo momio), cuentas independientes.

Momios en formato AMERICANO (+150, -170), no decimales ni porcentajes:
    momio positivo: ganancia = monto * (momio / 100)
    momio negativo: ganancia = monto * (100 / abs(momio))

Requiere el secret GOOGLE_CREDENTIALS_JSON (cuenta de servicio de Google
dedicada a este proyecto) - hay que compartir la carpeta "apuestas
financieras" (o cada Sheet individualmente) con el correo de esa cuenta
de servicio, con permiso de Editor.
"""

from datetime import date

TIER_SHEET_NAMES = {
    1000: "Inversores de $1,000",
    3000: "Inversores de $3,000",
    5000: "Inversores de $5,000",
    10000: "Inversores de $10,000",
}

REGISTRO_TAB = "Registro"
REGISTRO_HEADERS = ["Nombre", "Telefono", "Correo"]

# Mismo formato que las pestanas ya creadas a mano por el usuario (ej.
# "Mario Palma", "Emmanuel Rios" en el Sheet de $1,000): la columna A
# queda vacia, los encabezados van en la fila 2 empezando en la columna B.
HEADER_ROW = 2
DATA_START_COL = "B"
BET_HEADERS = [
    "Partido al que se aposto", "Fecha en la que se aposto", "Apuesta que se realizo",
    "Momio en la que se tomo", "Inversion actual", "Ganada / Perdida", "Inversion despues de apuesta",
]


def _parse_number(value):
    """El Sheet usa formato regional en espanol (coma decimal: '90,91' en
    vez de '90.91') al leer valores YA formateados de vuelta - sin esto,
    float() truena o da un numero equivocado en silencio."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip().replace(",", "."))


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


def _get_or_create_registro(sh):
    import gspread
    try:
        return sh.worksheet(REGISTRO_TAB)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=REGISTRO_TAB, rows=200, cols=len(REGISTRO_HEADERS) + 1)
        ws.append_row(REGISTRO_HEADERS)
        return ws


def _get_or_create_investor_tab(sh, nombre):
    """Pestana de UN inversionista - si ya existe (ej. creada a mano por
    el usuario, como 'Mario Palma'), se usa tal cual sin tocar sus
    encabezados. Si no existe, se crea con el mismo formato exacto
    (columna A vacia, encabezados en la fila 2 desde la columna B)."""
    import gspread
    try:
        return sh.worksheet(nombre)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=nombre, rows=200, cols=len(BET_HEADERS) + 2)
        ws.update(range_name=f"{DATA_START_COL}{HEADER_ROW}", values=[BET_HEADERS])
        return ws


def _read_bet_rows(ws):
    """Filas de datos de la pestana de un inversionista (despues de la
    fila de encabezado, columnas B en adelante), como lista de dicts con
    las llaves de BET_HEADERS. Ignora filas totalmente vacias."""
    all_values = ws.get_all_values()
    data_rows = all_values[HEADER_ROW:]
    out = []
    for row in data_rows:
        cells = row[1:1 + len(BET_HEADERS)]
        if not any(c.strip() for c in cells if c):
            continue
        cells = cells + [""] * (len(BET_HEADERS) - len(cells))
        out.append(dict(zip(BET_HEADERS, cells)))
    return out


def _append_bet_row(ws, values):
    """Escribe una fila nueva justo debajo de la ultima fila con datos,
    en las columnas B..H (deja la columna A vacia, igual que las
    pestanas creadas a mano)."""
    col_b_values = ws.col_values(2)  # columna B
    next_row = len(col_b_values) + 1
    if next_row <= HEADER_ROW:
        next_row = HEADER_ROW + 1
    end_col = chr(ord(DATA_START_COL) + len(BET_HEADERS) - 1)
    ws.update(range_name=f"{DATA_START_COL}{next_row}:{end_col}{next_row}", values=[values])


def list_investors(gc, tier):
    sh = _open_tier_sheet(gc, tier)
    ws = _get_or_create_registro(sh)
    rows = ws.get_all_records()
    return [{"nombre": r["Nombre"], "telefono": r.get("Telefono", ""), "correo": r.get("Correo", "")}
            for r in rows if r.get("Nombre")]


def add_investor(gc, tier, nombre, telefono, correo):
    sh = _open_tier_sheet(gc, tier)
    registro = _get_or_create_registro(sh)
    existentes = [r["Nombre"].strip().lower() for r in registro.get_all_records() if r.get("Nombre")]
    if nombre.strip().lower() in existentes:
        raise ValueError(f"'{nombre}' ya esta registrado en el nivel ${tier}.")
    registro.append_row([nombre, telefono, correo])
    _get_or_create_investor_tab(sh, nombre)
    return True


def update_investor(gc, tier, nombre, telefono=None, correo=None):
    """Edita telefono/correo de un inversionista YA registrado (el nombre
    no se puede cambiar aqui - es el mismo nombre de su pestana de
    apuestas, cambiarlo rompería esa referencia)."""
    sh = _open_tier_sheet(gc, tier)
    registro = _get_or_create_registro(sh)
    cell = registro.find(nombre, in_column=1)
    if not cell:
        raise ValueError(f"'{nombre}' no esta registrado en el nivel ${tier}.")
    if telefono is not None:
        registro.update_cell(cell.row, 2, telefono)
    if correo is not None:
        registro.update_cell(cell.row, 3, correo)
    return True


def delete_investor(gc, tier, nombre):
    """Borra al inversionista del Registro Y borra su pestana completa de
    apuestas (con su historial) - accion IRREVERSIBLE."""
    sh = _open_tier_sheet(gc, tier)
    registro = _get_or_create_registro(sh)
    cell = registro.find(nombre, in_column=1)
    if cell:
        registro.delete_rows(cell.row)
    try:
        ws = sh.worksheet(nombre)
        sh.del_worksheet(ws)
    except Exception:
        pass
    return True


def move_investor_tier(gc, from_tier, to_tier, nombre, nuevo_saldo=None):
    """Mueve a un inversionista de un nivel a otro (ej. de $1,000 a
    $3,000) - SOLO pasa cuando el usuario lo pide explicitamente aqui,
    nunca automatico por que su saldo haya crecido o bajado. Su pestana
    de historial en el nivel VIEJO se deja intacta (no se borra, queda
    como registro), solo se quita del Registro viejo para que ya no
    cuente como activo ahi. En el nivel NUEVO se crea su pestana con el
    saldo inicial indicado (por default, su saldo actual del nivel
    viejo)."""
    if from_tier == to_tier:
        raise ValueError("El nivel origen y destino son el mismo.")

    sh_from = _open_tier_sheet(gc, from_tier)
    registro_from = _get_or_create_registro(sh_from)
    cell = registro_from.find(nombre, in_column=1)
    if not cell:
        raise ValueError(f"'{nombre}' no esta registrado en el nivel ${from_tier}.")
    row_values = registro_from.row_values(cell.row)
    telefono = row_values[1] if len(row_values) > 1 else ""
    correo = row_values[2] if len(row_values) > 2 else ""

    saldo_actual = nuevo_saldo if nuevo_saldo is not None else get_investor_balance(gc, from_tier, nombre)

    sh_to = _open_tier_sheet(gc, to_tier)
    registro_to = _get_or_create_registro(sh_to)
    existentes = [r["Nombre"].strip().lower() for r in registro_to.get_all_records() if r.get("Nombre")]
    if nombre.strip().lower() in existentes:
        raise ValueError(f"'{nombre}' ya esta registrado en el nivel ${to_tier}.")
    registro_to.append_row([nombre, telefono, correo])
    ws_to = _get_or_create_investor_tab(sh_to, nombre)
    values = ["", date.today().isoformat(), f"Traspaso desde nivel ${from_tier:,}",
              "", saldo_actual, 0, round(saldo_actual, 2)]
    _append_bet_row(ws_to, values)

    registro_from.delete_rows(cell.row)
    return True


def log_movement(gc, tier, nombre, tipo, monto, fecha=None):
    """Retiro o deposito de fondos FUERA de una apuesta (ej. el
    inversionista retira ganancias y se queda solo con su monto nominal).
    NO cambia de hoja/nivel al inversionista automaticamente - eso es una
    decision manual del usuario (mover a alguien de nivel significa
    borrarlo de una hoja y agregarlo en otra a mano). tipo: 'Retiro' o
    'Deposito'."""
    if tipo not in ("Retiro", "Deposito"):
        raise ValueError("tipo debe ser 'Retiro' o 'Deposito'")
    fecha = fecha or date.today().isoformat()

    sh = _open_tier_sheet(gc, tier)
    ws = _get_or_create_investor_tab(sh, nombre)
    saldo_previo = get_investor_balance(gc, tier, nombre)

    delta = -abs(monto) if tipo == "Retiro" else abs(monto)
    saldo_nuevo = saldo_previo + delta

    values = ["", fecha, tipo, "", monto, round(delta, 2), round(saldo_nuevo, 2)]
    _append_bet_row(ws, values)
    return dict(zip(BET_HEADERS, values))


def get_investor_balance(gc, tier, nombre):
    """Saldo actual: la 'Inversion despues de apuesta' de la ULTIMA
    apuesta registrada, o el monto nominal del nivel si todavia no tiene
    ninguna apuesta."""
    sh = _open_tier_sheet(gc, tier)
    ws = _get_or_create_investor_tab(sh, nombre)
    rows = _read_bet_rows(ws)
    if not rows:
        return float(tier)
    try:
        return _parse_number(rows[-1]["Inversion despues de apuesta"])
    except (ValueError, KeyError):
        return float(tier)


def log_bet(gc, tier, nombre, partido, apuesta, monto, momio, resultado, fecha=None):
    """Agrega una apuesta resuelta (resultado ya conocido: 'Gano'/'Perdio'/
    'Push') al historial del inversionista y actualiza su saldo
    acumulado. Regresa la fila agregada (dict, llaves de BET_HEADERS)."""
    if resultado not in ("Gano", "Perdio", "Push"):
        raise ValueError("resultado debe ser 'Gano', 'Perdio' o 'Push'")
    fecha = fecha or date.today().isoformat()

    sh = _open_tier_sheet(gc, tier)
    ws = _get_or_create_investor_tab(sh, nombre)
    saldo_previo = get_investor_balance(gc, tier, nombre)

    ganancia = american_odds_profit(monto, momio, resultado)
    saldo_nuevo = saldo_previo + ganancia

    momio_str = f"+{momio}" if momio > 0 else str(momio)
    values = [partido, fecha, apuesta, momio_str, monto, round(ganancia, 2), round(saldo_nuevo, 2)]
    _append_bet_row(ws, values)
    return dict(zip(BET_HEADERS, values))


def get_bets_for_date(gc, tier, nombre, fecha):
    """Todas las apuestas de un inversionista logueadas en `fecha`
    (YYYY-MM-DD) - para el reporte por correo."""
    sh = _open_tier_sheet(gc, tier)
    try:
        ws = sh.worksheet(nombre)
    except Exception:
        return []
    return [r for r in _read_bet_rows(ws) if r.get("Fecha en la que se aposto") == fecha]


def all_investors_all_tiers(gc):
    """{tier: [ {nombre, telefono, correo}, ... ]} para los 4 niveles."""
    return {tier: list_investors(gc, tier) for tier in TIER_SHEET_NAMES}
