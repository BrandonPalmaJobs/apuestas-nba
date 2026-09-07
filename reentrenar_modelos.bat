@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   REENTRENAR MODELOS NBA (puntos totales, spread, money line)
echo ============================================================
echo.
set /p SEASON="Anio en que TERMINA la temporada a usar (2025 para 2024-25, Enter para temporada actual): "

echo.
echo Paso 1/2: recolectando el dataset historico de TODA la liga...
echo (la primera vez tarda bastante - se guarda en cache y las
echo  siguientes corridas son mucho mas rapidas)
echo.
if "%SEASON%"=="" (
    python nba_train_data.py --out training_data_nba.csv
) else (
    python nba_train_data.py --season %SEASON% --out training_data_nba.csv
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
python nba_train.py --data training_data_nba.csv

echo.
echo ============================================================
echo   LISTO. Revisa arriba si el modelo entrenado le gano a la
echo   formula simple (baseline) en cada bet. Los .joblib y
echo   training_history_nba.csv ya quedaron actualizados.
echo ============================================================
pause
