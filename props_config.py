"""
Definicion UNICA de que props de jugador maneja el sistema - la usan la
recoleccion (nba_props_data), el entrenamiento (nba_props_train), la
prediccion (nba_props_predict, streamlit_app) y el registro de apuestas.
Para agregar un prop nuevo basta con agregarlo aqui y reentrenar.
"""

# Estadisticas base por juego: nombre interno -> campo de ESPN
# (espn_player_game_stats). Todas ya vienen en el cache de stats de jugador,
# asi que agregar una nueva NO requiere volver a descargar nada.
BASE_STATS = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "threes": "threePointFieldGoalsMade",
    "steals": "steals",
    "blocks": "blocks",
    "turnovers": "turnovers",
    "minutes": "minutes",
}

# Props combinados que ofrecen las casas: suma de estadisticas base.
COMBOS = {
    "pra": ("points", "rebounds", "assists"),
    "pr": ("points", "rebounds"),
    "pa": ("points", "assists"),
    "ra": ("rebounds", "assists"),
}

# Props a predecir (cada uno tiene su propio modelo model_props_<key>.joblib),
# en el orden en que se muestran.
TARGETS = ["points", "rebounds", "assists", "threes", "pra", "pr", "pa", "ra", "steals", "blocks", "turnovers"]

LABELS = {
    "points": "Puntos", "rebounds": "Rebotes", "assists": "Asistencias", "threes": "Triples",
    "steals": "Robos", "blocks": "Bloqueos", "turnovers": "Perdidas", "minutes": "Minutos",
    "pra": "Pts+Reb+Ast", "pr": "Pts+Reb", "pa": "Pts+Ast", "ra": "Reb+Ast",
}

# Estadisticas de las que se mide "lo que permite el rival por posicion".
ALLOWED_STATS = ["points", "rebounds", "assists", "threes", "steals", "blocks", "turnovers"]


def stat_value(game, key):
    """Valor de un prop (base o combinado) en un juego ya normalizado."""
    if key in COMBOS:
        parts = [game.get(k) for k in COMBOS[key]]
        return None if any(p is None for p in parts) else sum(parts)
    return game.get(key)


def normalize_stats(raw):
    """Stats crudas de ESPN -> dict con las llaves internas de BASE_STATS."""
    return {k: raw.get(espn_key) for k, espn_key in BASE_STATS.items()}


def avg_feature(key):
    return f"avg_{key}"


def allow_feature(key):
    return f"opp_allow_{key}36_pos"
