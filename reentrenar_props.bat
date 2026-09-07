@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   REENTRENAR MODELOS DE PROPS DE JUGADOR (puntos/rebotes/asistencias)
echo ============================================================
echo.
echo Solo cubre jugadores TITULARES (starter) de cada equipo - ver el
echo comentario en nba_props_data.py para el porque.
echo.
set /p SEASON="Anio en que TERMINA la temporada a usar (Enter para temporada actual): "

echo.
echo Paso 1/2: recolectando el dataset de props de TODA la liga...
echo (reusa el cache de reentrenar_modelos.bat si ya lo corriste para
echo  esta misma temporada - si no, tarda bastante mas)
echo.
if "%SEASON%"=="" (
    python nba_props_data.py --out training_data_props.csv
) else (
    python nba_props_data.py --season %SEASON% --out training_data_props.csv
)
if errorlevel 1 (
    echo.
    echo ERROR recolectando el dataset - revisa el mensaje de arriba.
    pause
    exit /b 1
)

echo.
echo Paso 2/2: entrenando los modelos...
echo.
python nba_props_train.py --data training_data_props.csv

echo.
echo ============================================================
echo   LISTO. Revisa arriba si el modelo entrenado le gano al
echo   promedio propio del jugador en cada estadistica.
echo ============================================================
pause
