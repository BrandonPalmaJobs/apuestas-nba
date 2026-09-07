@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   REPORTE NBA - analisis pre-apuesta
echo ============================================================
echo.
echo Este programa te va a pedir un par de datos y despues genera
echo un reporte del juego: ofensiva/defensiva avanzada, lesionados,
echo fatiga de calendario, y proyeccion de puntos por cuarto.
echo.
echo ------------------------------------------------------------
echo PASO 1 de 3: Que equipos juegan
echo ------------------------------------------------------------
echo Puedes escribir el nombre completo, la ciudad, o la
echo abreviacion. Ejemplos validos: "Lakers", "Los Angeles Lakers",
echo "LAL". No importa mayusculas/minusculas.
echo.
set /p EQUIPO_A="  Equipo A: "
set /p EQUIPO_B="  Equipo B: "

echo.
echo ------------------------------------------------------------
echo PASO 2 de 3: Temporada (opcional)
echo ------------------------------------------------------------
echo Escribe el anio en que TERMINA la temporada que quieres
echo analizar (ej. 2025 para la temporada 2024-25, ya jugada
echo completa - util para practicar si la temporada nueva todavia
echo no arranca). Deja vacio para usar la temporada actual.
echo.
set /p TEMPORADA="  Temporada (Enter para la actual): "

set SEASONOPT=
if not "%TEMPORADA%"=="" set SEASONOPT= --season %TEMPORADA%

echo.
echo ------------------------------------------------------------
echo PASO 3 de 3: Forzar un lesionado (opcional)
echo ------------------------------------------------------------
echo El reporte YA muestra los lesionados actuales segun ESPN.
echo Usa esto UNICAMENTE si quieres ademas calcular el impacto
echo (On/Off Net Rating) de un jugador especifico.
echo.
echo Si no quieres forzar nada, deja estas dos preguntas VACIAS y
echo solo presiona Enter.
echo.
set /p LESIONADO_A="  Forzar lesionado del Equipo A (Enter para omitir): "
set /p LESIONADO_B="  Forzar lesionado del Equipo B (Enter para omitir): "

set OVERRIDES=
if not "%LESIONADO_A%"=="" set OVERRIDES=%OVERRIDES% --lesionado-a "%LESIONADO_A%"
if not "%LESIONADO_B%"=="" set OVERRIDES=%OVERRIDES% --lesionado-b "%LESIONADO_B%"

echo.
echo ============================================================
echo   Generando el reporte, esto puede tardar 1-3 minutos...
echo ============================================================
echo.

python nba_report.py "%EQUIPO_A%" "%EQUIPO_B%"%SEASONOPT%%OVERRIDES%

echo.
echo ============================================================
echo   LISTO. Si algo salio mal, revisa arriba el mensaje de error
echo   (los mas comunes: nombre de equipo mal escrito, sin
echo   internet, o error 403 de ESPN por limite de solicitudes -
echo   en ese caso espera unos minutos y reintenta. Corre primero
echo   probar_conexion.bat si no estas seguro de que todo
echo   funciona). Presiona una tecla para cerrar esta ventana.
echo ============================================================
pause >nul
