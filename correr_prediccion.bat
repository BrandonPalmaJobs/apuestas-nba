@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   PREDICCION NBA CON MODELO ENTRENADO (Fase 2)
echo ============================================================
echo.
echo Necesita los archivos model_nba_total.joblib, model_nba_margin.joblib
echo y model_nba_win.joblib ya entrenados (corre primero
echo reentrenar_modelos.bat si todavia no existen).
echo.
set /p EQUIPO_A="Equipo A: "
set /p EQUIPO_B="Equipo B: "
set /p SEASON="Anio en que TERMINA la temporada (Enter para temporada actual): "
set /p LOCAL="Si NO hay juego real programado (receso): cual es local, A o B (Enter = A): "

set ARGS="%EQUIPO_A%" "%EQUIPO_B%"
if not "%SEASON%"=="" set ARGS=%ARGS% --season %SEASON%
if not "%LOCAL%"=="" set ARGS=%ARGS% --local %LOCAL%

echo.
python nba_predict.py %ARGS%

echo.
echo ============================================================
echo   LISTO. Si hubo un juego real programado, la prediccion ya
echo   quedo guardada en predictions_log_nba.csv para evaluarla
echo   despues con evaluar_predicciones.bat. En modo practica (sin
echo   juego real) no se guarda nada.
echo ============================================================
pause
