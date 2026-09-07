@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   EVALUAR PREDICCIONES NBA (contra el resultado real)
echo ============================================================
echo.
echo Revisa predictions_log_nba.csv: para cada juego ya predicho que
echo ya termino, baja el resultado real y compara los modelos contra
echo la formula simple (baseline).
echo.

python nba_track.py --evaluate

echo.
echo ============================================================
echo   LISTO. Presiona una tecla para cerrar.
echo ============================================================
pause
