"""
Capa de acceso a datos de NBA. stats.nba.com esta bloqueado a nivel de red
para este usuario (confirmado con ERR_CONNECTION_RESET incluso en un
navegador normal, no es bot-detection) - este modulo usa dos fuentes
alternas, ambas verificadas con datos reales:

  - ESPN (site.api.espn.com, sports.core.api.espn.com): boxscores completos
    por equipo y jugador, calendario, marcador por cuarto, lesionados. De
    aqui se CALCULAN las metricas avanzadas (OffRtg/DefRtg/Pace/eFG%/TS%/
    Four Factors) con las formulas estandar de Dean Oliver - son formulas
    publicas, no un invento propio.
  - pbpstats.com (api.pbpstats.com): On/Off Net Rating por jugador (para
    medir impacto de lesionados) y datos de tiro por zona, derivados de las
    jugadas oficiales de la NBA por un proyecto independiente.

Lo que NINGUNA fuente publica tiene (solo existe en el tracking de video
propietario de la NBA/Second Spectrum): Screen Assists, Deflection Rate,
Box-Outs, Loose Balls, Box Creation exacto. Se omiten del reporte en vez de
inventarlos.
"""

import json
import os
import time
from datetime import date, datetime, timedelta

import requests

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_CORE_BASE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
PBP_BASE = "https://api.pbpstats.com"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"}

# ---------------------------------------------------------------------------
# Cache en disco de resumenes de juegos YA jugados - un juego terminado
# nunca cambia, asi que se guarda para siempre. Esto es lo que de verdad
# evita disparar el limite de solicitudes de ESPN: distintos reportes
# comparten muchos de los mismos juegos recientes entre si (ej. el ultimo
# juego de los Lakers aparece en CUALQUIER reporte que incluya a los
# Lakers), asi que con el tiempo la gran mayoria de las solicitudes de un
# reporte nuevo ya estan en cache y no vuelven a pedirse.
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
_SUMMARY_CACHE_PATH = os.path.join(_CACHE_DIR, "game_summaries.json")
_summary_cache = None
_cache_dirty = False


def _load_summary_cache():
    global _summary_cache
    if _summary_cache is None:
        if os.path.exists(_SUMMARY_CACHE_PATH):
            try:
                with open(_SUMMARY_CACHE_PATH, "r", encoding="utf-8") as f:
                    _summary_cache = json.load(f)
            except Exception:
                _summary_cache = {}
        else:
            _summary_cache = {}
    return _summary_cache


def flush_cache():
    """Guarda a disco los juegos nuevos que se agregaron al cache durante
    esta corrida - se llama UNA vez al final (no en cada request) para no
    reescribir el archivo completo decenas de veces."""
    global _cache_dirty
    if not _cache_dirty or _summary_cache is None:
        return
    os.makedirs(_CACHE_DIR, exist_ok=True)
    tmp_path = _SUMMARY_CACHE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_summary_cache, f)
    os.replace(tmp_path, _SUMMARY_CACHE_PATH)
    _cache_dirty = False


def get_json(url, params=None, retries=3, timeout=25, min_interval=0.3):
    """GET con reintentos - mismo patron que apuestas_mlb/mlb_first_inning_report.py:
    con muchas llamadas seguidas, un solo timeout no debe tumbar toda una
    corrida.

    Diferencia importante con MLB: 403/429 (limite de solicitudes) NO se
    reintenta - reintentar un rechazo por limite de solicitudes no lo
    arregla, solo hace que un reporte que falla tarde varios MINUTOS en vez
    de fallar rapido en segundos (nos paso probando este mismo codigo).
    Tambien se espera un mínimo entre llamadas para no disparar ese limite
    en primer lugar durante un reporte normal (~20-40 requests)."""
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
            if r.status_code in (403, 429):
                raise requests.exceptions.HTTPError(
                    f"{r.status_code} - limite de solicitudes o bloqueo (no se reintenta)", response=r)
            r.raise_for_status()
            time.sleep(min_interval)
            return r.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (403, 429):
                raise
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    raise last_err


def current_nba_season():
    """Anio de temporada que espera ESPN/pbpstats: la NBA arranca en
    octubre - antes de eso, 'la temporada actual' sigue siendo la que
    termino en la primavera de este mismo anio calendario."""
    d = date.today()
    year = d.year + 1 if d.month >= 10 else d.year
    return year


def season_label(season_end_year=None):
    """Convierte el anio en que TERMINA la temporada (ej. 2025) al formato
    'YYYY-YY' que usa pbpstats (ej. '2024-25')."""
    y = (season_end_year or current_nba_season()) - 1
    return f"{y}-{str(y + 1)[2:]}"


# ---------------------------------------------------------------------------
# Equipos (lista estatica - los IDs coinciden entre ESPN y pbpstats/stats.nba)
# ---------------------------------------------------------------------------

TEAMS = [
    {"nba_id": 1610612737, "espn_id": 1, "full_name": "Atlanta Hawks", "abbreviation": "ATL"},
    {"nba_id": 1610612738, "espn_id": 2, "full_name": "Boston Celtics", "abbreviation": "BOS"},
    {"nba_id": 1610612751, "espn_id": 17, "full_name": "Brooklyn Nets", "abbreviation": "BKN"},
    {"nba_id": 1610612766, "espn_id": 30, "full_name": "Charlotte Hornets", "abbreviation": "CHA"},
    {"nba_id": 1610612741, "espn_id": 4, "full_name": "Chicago Bulls", "abbreviation": "CHI"},
    {"nba_id": 1610612739, "espn_id": 5, "full_name": "Cleveland Cavaliers", "abbreviation": "CLE"},
    {"nba_id": 1610612742, "espn_id": 6, "full_name": "Dallas Mavericks", "abbreviation": "DAL"},
    {"nba_id": 1610612743, "espn_id": 7, "full_name": "Denver Nuggets", "abbreviation": "DEN"},
    {"nba_id": 1610612765, "espn_id": 8, "full_name": "Detroit Pistons", "abbreviation": "DET"},
    {"nba_id": 1610612744, "espn_id": 9, "full_name": "Golden State Warriors", "abbreviation": "GSW"},
    {"nba_id": 1610612745, "espn_id": 10, "full_name": "Houston Rockets", "abbreviation": "HOU"},
    {"nba_id": 1610612754, "espn_id": 11, "full_name": "Indiana Pacers", "abbreviation": "IND"},
    {"nba_id": 1610612746, "espn_id": 12, "full_name": "LA Clippers", "abbreviation": "LAC"},
    {"nba_id": 1610612747, "espn_id": 13, "full_name": "Los Angeles Lakers", "abbreviation": "LAL"},
    {"nba_id": 1610612763, "espn_id": 29, "full_name": "Memphis Grizzlies", "abbreviation": "MEM"},
    {"nba_id": 1610612748, "espn_id": 14, "full_name": "Miami Heat", "abbreviation": "MIA"},
    {"nba_id": 1610612749, "espn_id": 15, "full_name": "Milwaukee Bucks", "abbreviation": "MIL"},
    {"nba_id": 1610612750, "espn_id": 16, "full_name": "Minnesota Timberwolves", "abbreviation": "MIN"},
    {"nba_id": 1610612740, "espn_id": 3, "full_name": "New Orleans Pelicans", "abbreviation": "NOP"},
    {"nba_id": 1610612752, "espn_id": 18, "full_name": "New York Knicks", "abbreviation": "NYK"},
    {"nba_id": 1610612760, "espn_id": 25, "full_name": "Oklahoma City Thunder", "abbreviation": "OKC"},
    {"nba_id": 1610612753, "espn_id": 19, "full_name": "Orlando Magic", "abbreviation": "ORL"},
    {"nba_id": 1610612755, "espn_id": 20, "full_name": "Philadelphia 76ers", "abbreviation": "PHI"},
    {"nba_id": 1610612756, "espn_id": 21, "full_name": "Phoenix Suns", "abbreviation": "PHX"},
    {"nba_id": 1610612757, "espn_id": 22, "full_name": "Portland Trail Blazers", "abbreviation": "POR"},
    {"nba_id": 1610612758, "espn_id": 23, "full_name": "Sacramento Kings", "abbreviation": "SAC"},
    {"nba_id": 1610612759, "espn_id": 24, "full_name": "San Antonio Spurs", "abbreviation": "SAS"},
    {"nba_id": 1610612761, "espn_id": 28, "full_name": "Toronto Raptors", "abbreviation": "TOR"},
    {"nba_id": 1610612762, "espn_id": 26, "full_name": "Utah Jazz", "abbreviation": "UTA"},
    {"nba_id": 1610612764, "espn_id": 27, "full_name": "Washington Wizards", "abbreviation": "WAS"},
]


def resolve_team(query):
    q = query.strip().lower()
    for t in TEAMS:
        if q == t["full_name"].lower() or q == t["abbreviation"].lower():
            return t
    for t in TEAMS:
        if q == t["full_name"].lower().split()[-1]:
            return t
    for t in TEAMS:
        if q in t["full_name"].lower():
            return t
    raise ValueError(f"No encontre un equipo de NBA que coincida con '{query}'")


# ---------------------------------------------------------------------------
# ESPN: calendario, boxscores, jugadores, lesionados
# ---------------------------------------------------------------------------

def espn_team_event_ids(espn_team_id, season=None):
    """IDs de todos los juegos (regular season) de un equipo en una
    temporada, en orden cronologico. Usa la API 'core' de ESPN (temporada
    en la URL) porque la API 'site' con ?season= en query string regresa
    403 para temporadas que no sean la actual - la 'core' si acepta
    temporadas pasadas, util para practicar antes de que arranque la
    temporada nueva."""
    import re
    data = get_json(
        f"{ESPN_CORE_BASE}/seasons/{season or current_nba_season()}/types/2/teams/{espn_team_id}/events",
        params={"limit": 100},
    )
    ids = []
    for item in data.get("items", []):
        m = re.search(r"/events/(\d+)", item.get("$ref", ""))
        if m:
            ids.append(m.group(1))
    return ids


def espn_played_games(espn_team_id, season=None, last_n=None, request_delay=0.6):
    """Lista de los ultimos N juegos YA jugados de un equipo (id, fecha,
    rival) - usa el API 'core' (ver espn_competition_detail), que no esta
    bloqueado a diferencia del 'site' (site.api.espn.com/summary, el que se
    usaba antes y empezo a dar 403 incluso desde una red residencial normal
    despues de varias pruebas seguidas)."""
    ids = espn_team_event_ids(espn_team_id, season=season)
    ids_to_check = ids[-(last_n * 2):] if last_n else ids  # margen por si algun juego no cargo bien
    games = []
    fallos_403 = 0
    ultimo_error = None
    for eid in reversed(ids_to_check):
        if last_n and len(games) >= last_n:
            break
        time.sleep(request_delay)
        try:
            comp = espn_competition_detail(eid)
        except Exception as e:
            ultimo_error = e
            if "403" in str(e) or "429" in str(e):
                fallos_403 += 1
            continue
        if not comp.get("completed"):
            continue
        opp = next((c for c in comp["competitors"] if str(c["espn_id"]) != str(espn_team_id)), None)
        if not opp:
            continue
        games.append({"id": eid, "date": comp.get("date"), "opponent_espn_id": opp["espn_id"]})
    if not games and (fallos_403 or ultimo_error):
        raise RuntimeError(
            f"No se pudo traer ningun juego ({fallos_403} de ellos por limite de solicitudes/bloqueo). "
            f"Ultimo error: {ultimo_error}. Espera unos minutos antes de reintentar - ESPN limita cuantas "
            f"solicitudes seguidas acepta.")
    games.sort(key=lambda g: g.get("date") or "")
    return games


def espn_competition_detail(event_id):
    """Quienes jugaron (IDs de equipo ESPN) y si el juego ya termino - UNA
    llamada ligera al API 'core' (sin noticias/videos/jugada por jugada
    como el 'summary' viejo). Se cachea en disco solo si el juego ya
    termino (uno en vivo puede seguir cambiando)."""
    global _cache_dirty
    cache = _load_summary_cache()
    key = f"comp:{event_id}"
    if key in cache:
        return cache[key]

    data = get_json(f"{ESPN_CORE_BASE}/events/{event_id}/competitions/{event_id}")
    completed = False
    try:
        status_url = data.get("status", {}).get("$ref", "").replace("http://", "https://")
        if status_url:
            status = get_json(status_url)
            completed = bool(status.get("type", {}).get("completed"))
    except Exception:
        pass

    result = {
        "completed": completed,
        "date": data.get("date"),
        "competitors": [{"espn_id": c.get("id"), "home_away": c.get("homeAway")}
                         for c in data.get("competitors", [])],
    }
    if completed:
        cache[key] = result
        _cache_dirty = True
    return result


def espn_team_game_stats(event_id, espn_team_id):
    """Estadisticas de UN equipo en UN juego ya jugado - valores numericos
    ya limpios (a diferencia del 'summary' viejo, que traia todo como texto
    tipo '46-86' y habia que partir a mano)."""
    global _cache_dirty
    cache = _load_summary_cache()
    key = f"stats:{event_id}:{espn_team_id}"
    if key in cache:
        return cache[key]
    data = get_json(f"{ESPN_CORE_BASE}/events/{event_id}/competitions/{event_id}"
                     f"/competitors/{espn_team_id}/statistics")
    out = {}
    for cat in data.get("splits", {}).get("categories", []):
        for s in cat.get("stats", []):
            out[s["name"]] = s.get("value")
    cache[key] = out
    _cache_dirty = True
    return out


def espn_team_game_linescores(event_id, espn_team_id):
    """Puntos por cuarto de UN equipo en UN juego ya jugado, en orden."""
    global _cache_dirty
    cache = _load_summary_cache()
    key = f"lines:{event_id}:{espn_team_id}"
    if key in cache:
        return cache[key]
    data = get_json(f"{ESPN_CORE_BASE}/events/{event_id}/competitions/{event_id}"
                     f"/competitors/{espn_team_id}/linescores")
    items = sorted(data.get("items", []), key=lambda x: x.get("period", 0))
    quarters = [float(item.get("value", 0) or 0) for item in items]
    cache[key] = quarters
    _cache_dirty = True
    return quarters


def team_game_boxscore(event_id, espn_team_id):
    """Estadisticas + puntos por cuarto de UN equipo en UN juego ya jugado."""
    stats = espn_team_game_stats(event_id, espn_team_id)
    quarters = espn_team_game_linescores(event_id, espn_team_id)
    return stats, quarters


def espn_game_roster(event_id, espn_team_id):
    """Roster de UN equipo en UN juego ya jugado: quien jugo, quien empezo
    (starter), y su playerId - la base para props de jugador (Fase 3).
    NO se cachea en disco completo (solo los stats individuales, ver
    espn_player_game_stats) porque es liviano y rapido de re-pedir."""
    data = get_json(f"{ESPN_CORE_BASE}/events/{event_id}/competitions/{event_id}"
                     f"/competitors/{espn_team_id}/roster", params={"limit": 30})
    out = []
    for entry in data.get("entries", []):
        out.append({
            "player_id": entry.get("playerId"),
            "name": entry.get("displayName"),
            "starter": bool(entry.get("starter")),
            "did_not_play": bool(entry.get("didNotPlay")),
        })
    return out


def espn_player_game_stats(event_id, espn_team_id, player_id):
    """Estadisticas de UN jugador en UN juego ya jugado - mismo formato
    plano que espn_team_game_stats (categories/stats -> {name: value}),
    trae 'points'/'rebounds'/'assists'/'minutes' ya limpios. Cacheado en
    disco igual que los boxscores de equipo."""
    global _cache_dirty
    cache = _load_summary_cache()
    key = f"pstats:{event_id}:{espn_team_id}:{player_id}"
    if key in cache:
        return cache[key]
    data = get_json(f"{ESPN_CORE_BASE}/events/{event_id}/competitions/{event_id}"
                     f"/competitors/{espn_team_id}/roster/{player_id}/statistics/0")
    out = {}
    for cat in data.get("splits", {}).get("categories", []):
        for s in cat.get("stats", []):
            out[s["name"]] = s.get("value")
    cache[key] = out
    _cache_dirty = True
    return out


def player_boxscores(event_id):
    """Estadisticas de todos los jugadores que SI jugaron, de ambos equipos,
    en un juego ya jugado - reescrita sobre el API 'core' (ver
    espn_game_roster/espn_player_game_stats) para props de jugador
    (Fase 3)."""
    comp = espn_competition_detail(event_id)
    out = []
    for c in comp["competitors"]:
        team_espn_id = c["espn_id"]
        for entry in espn_game_roster(event_id, team_espn_id):
            if entry["did_not_play"]:
                continue
            stats = espn_player_game_stats(event_id, team_espn_id, entry["player_id"])
            out.append({**entry, "team_espn_id": team_espn_id, "stats": stats})
    return out


def espn_team_season_roster(espn_team_id, season=None):
    """Roster COMPLETO del equipo para una temporada (no un juego
    especifico) - para resolver el nombre de un jugador a su ID antes de
    poder pedir su historial de juegos (props de jugador, Fase 3). Se
    cachea en disco: el roster no cambia seguido dentro de una temporada, y
    si cambia un poco por un trade no es grave que quede desactualizado
    unos dias."""
    global _cache_dirty
    cache = _load_summary_cache()
    season = season or current_nba_season()
    key = f"roster_season:{espn_team_id}:{season}"
    if key in cache:
        return cache[key]
    data = get_json(f"{ESPN_CORE_BASE}/seasons/{season}/teams/{espn_team_id}/athletes", params={"limit": 50})
    out = []
    for item in data.get("items", []):
        ref = item.get("$ref", "").replace("http://", "https://")
        try:
            detail = get_json(ref)
        except Exception:
            continue
        out.append({"player_id": detail.get("id"), "name": detail.get("displayName")})
    cache[key] = out
    _cache_dirty = True
    return out


def resolve_player_espn(espn_team_id, name_query, season=None):
    roster = espn_team_season_roster(espn_team_id, season=season)
    q = name_query.strip().lower()
    for p in roster:
        if q == (p["name"] or "").lower():
            return p["player_id"], p["name"]
    for p in roster:
        if q in (p["name"] or "").lower():
            return p["player_id"], p["name"]
    return None, None


def espn_team_injuries(espn_team_id):
    """Lesionados actuales de un equipo (roster + status), directo de ESPN -
    no requiere parsear ningun PDF."""
    data = get_json(f"{ESPN_BASE}/teams/{espn_team_id}/injuries")
    return data.get("injuries", [])


def find_next_matchup(team_a, team_b, days_ahead=14):
    """Busca el proximo juego programado entre dos equipos en el calendario
    de ESPN del equipo A. Usa la API 'site' SIN parametro de temporada (asi
    es como responde bien - con ?season= en query string da 403, ver
    espn_team_event_ids) - siempre trae el calendario proximo/actual, que es
    justo lo que hace falta aqui (juegos futuros, no historicos)."""
    data = get_json(f"{ESPN_BASE}/teams/{team_a['espn_id']}/schedule")
    events = data.get("events", [])
    today = date.today()
    for e in sorted(events, key=lambda e: e.get("date", "")):
        comp = e.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        ids = {str(c.get("team", {}).get("id")) for c in competitors}
        if str(team_b["espn_id"]) not in ids:
            continue
        game_date = datetime.fromisoformat(e["date"].replace("Z", "+00:00")).date()
        if game_date < today or game_date > today + timedelta(days=days_ahead):
            continue
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        return {
            "event_id": e["id"], "date": game_date.isoformat(),
            "home_espn_id": home.get("team", {}).get("id") if home else None,
        }
    return None


# ---------------------------------------------------------------------------
# pbpstats.com: On/Off Net Rating (impacto de lesionados) y tiro por zona
# ---------------------------------------------------------------------------

def pbp_team_roster(nba_team_id, season=None):
    """season = anio en que TERMINA la temporada (ej. 2025 para 2024-25),
    igual convencion que el resto del modulo."""
    data = get_json(f"{PBP_BASE}/get-team-players-for-season", params={
        "League": "nba", "Season": season_label(season),
        "SeasonType": "Regular Season", "TeamId": str(nba_team_id),
    })
    return data.get("players", {})  # {player_id: nombre}


def pbp_player_on_off(nba_team_id, player_id, season=None):
    """Net Rating (y otras metricas) del EQUIPO con este jugador en cancha
    vs. fuera - la base para medir el impacto real de una lesion."""
    data = get_json(f"{PBP_BASE}/get-on-off/nba/player", params={
        "Season": season_label(season), "SeasonType": "Regular Season",
        "TeamId": str(nba_team_id), "PlayerId": str(player_id),
    })
    return data.get("results", {})


def pbp_season_schedule(season=None):
    """Calendario COMPLETO de la temporada con marcador final, via
    pbpstats.com (fuente independiente de ESPN, deriva de las jugadas
    oficiales de la NBA) - un solo llamado trae los ~1230 juegos de toda
    la liga con local/visitante y marcador final. Se usa como RESPALDO si
    ESPN se bloquea (ver team_rolling_report en nba_report.py)."""
    data = get_json(f"{PBP_BASE}/get-games/nba", params={
        "Season": season_label(season), "SeasonType": "Regular Season",
    })
    return data.get("results", [])


def pbp_team_game_log(nba_team_id, season=None):
    """Historial de TODOS los juegos ya jugados de un equipo en la
    temporada, con estadisticas avanzadas YA CALCULADAS por pbpstats.com
    (EfgPct, TsPct, Pace, rebote%, etc.) - un solo llamado trae la
    temporada completa, a diferencia de ESPN que pide juego por juego.
    Respaldo independiente si ESPN se bloquea."""
    data = get_json(f"{PBP_BASE}/get-game-logs/nba", params={
        "Season": season_label(season), "SeasonType": "Regular Season",
        "EntityType": "Team", "EntityId": str(nba_team_id),
    })
    return data.get("multi_row_table_data", [])


def resolve_player_id(nba_team_id, name_query, season=None):
    roster = pbp_team_roster(nba_team_id, season=season)
    q = name_query.strip().lower()
    for pid, name in roster.items():
        if q == name.lower():
            return pid, name
    for pid, name in roster.items():
        if q in name.lower():
            return pid, name
    return None, None
