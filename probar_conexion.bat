@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo   Prueba de conexion a las fuentes de datos NBA
echo   (ESPN + pbpstats.com)
echo ================================================
echo.
echo Esto corre 3 llamadas reales para confirmar que estas
echo fuentes responden desde ESTA computadora antes de usar
echo el resto del sistema. Tarda unos segundos.
echo.

python probar_conexion.py

echo.
echo ================================================
echo   Listo. Presiona una tecla para cerrar.
echo ================================================
pause >nul
