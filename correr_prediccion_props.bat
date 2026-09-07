@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   PREDICCION DE PROPS DE JUGADOR (Fase 3)
echo ============================================================
echo.
echo Necesita los archivos model_props_points.joblib, model_props_rebounds.joblib
echo y model_props_assists.joblib ya entrenados (corre primero
echo reentrenar_props.bat si todavia no existen).
echo.
set /p JUGADOR="Jugador (nombre o parte del nombre): "
set /p EQUIPO_JUGADOR="Equipo del jugador: "
set /p EQUIPO_RIVAL="Equipo rival: "
set /p SEASON="Anio en que TERMINA la temporada (Enter para temporada actual): "

echo.
if "%SEASON%"=="" (
    python nba_props_predict.py "%JUGADOR%" "%EQUIPO_JUGADOR%" "%EQUIPO_RIVAL%"
) else (
    python nba_props_predict.py "%JUGADOR%" "%EQUIPO_JUGADOR%" "%EQUIPO_RIVAL%" --season %SEASON%
)

echo.
echo ============================================================
echo   LISTO. Presiona una tecla para cerrar.
echo ============================================================
pause
