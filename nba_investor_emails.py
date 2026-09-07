"""
Envio de reportes por correo a los inversionistas: resumen de los
movimientos (apuestas/retiros/depositos) registrados HOY, su saldo
actualizado, y las mismas estadisticas generales de la pestana de
Analisis de la app (efectividad, total apostado, ganancia total) - via
Gmail SMTP, en HTML con el mismo estilo visual del Sheet (encabezado
negro con letras blancas).

Requiere dos secrets:
    GMAIL_ADDRESS = "tu_correo@gmail.com"
    GMAIL_APP_PASSWORD = "xxxx xxxx xxxx xxxx"  (contrasena de aplicacion,
        NO tu contrasena normal - se genera en
        https://myaccount.google.com/apppasswords, requiere verificacion
        en 2 pasos activada en la cuenta)

Uso normal: nba_send_investor_reports.py (corrido por GitHub Actions cada
noche) llama a send_daily_report() para cada inversionista. Tambien se
puede probar a mano desde la app (boton "Enviar reporte de hoy (prueba)").
"""

import html
import io
import smtplib
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import matplotlib
matplotlib.use("Agg")  # sin pantalla - corre en un servidor (Streamlit Cloud / GitHub Actions)
import matplotlib.pyplot as plt

import nba_investors as inv

NEGRO = "#111111"
BLANCO = "#ffffff"
VERDE = "#1e7e34"
ROJO = "#c0392b"
GRIS_CLARO = "#f4f4f4"
GRIS_TEXTO = "#888888"


def _resultado_de(ganada_perdida_str):
    try:
        val = inv._parse_number(ganada_perdida_str)
    except (ValueError, TypeError):
        return "N/D"
    if val > 0:
        return "GANADA"
    if val < 0:
        return "PERDIDA"
    return "PUSH / SIN CAMBIO"


def _stats_generales_de(historial, tier):
    """Mismas estadisticas que la pestana de Analisis de la app (saldo
    actual, ganancia/perdida total, record ganadas-perdidas, efectividad,
    total apostado - todo el historial, no solo hoy), a partir de un
    historial YA LEIDO (individual o de un lote con get_full_history_batch),
    para no volver a leer Sheets aqui."""
    # saldo actual = el saldo de la ULTIMA fila ya leida - se evita una
    # segunda lectura completa de la pestana solo para el saldo (misma
    # lógica que get_investor_balance, pero sin repetir la llamada).
    saldo_actual = historial[-1]["saldo"] if historial else float(tier)
    ganancia_total = saldo_actual - tier
    apuestas = [h for h in historial if h["apuesta"] not in ("Retiro", "Deposito")]
    ganadas = [h for h in apuestas if h["ganada_perdida"] > 0]
    perdidas = [h for h in apuestas if h["ganada_perdida"] < 0]
    total_apostado = sum(h["monto"] for h in apuestas)
    win_rate = (len(ganadas) / len(apuestas) * 100) if apuestas else None
    return {
        "saldo_actual": saldo_actual, "ganancia_total": ganancia_total,
        "n_ganadas": len(ganadas), "n_perdidas": len(perdidas), "win_rate": win_rate,
        "total_apostado": total_apostado, "historial": historial, "apuestas": apuestas,
    }


MESES_ES = ["", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto",
            "Septiembre", "Octubre", "Noviembre", "Diciembre"]


def _movimientos_del_mes(historial, year, month):
    prefijo = f"{year:04d}-{month:02d}"
    return [h for h in historial if (h["fecha"] or "").startswith(prefijo)]


def _stats_del_mes(historial, tier, year, month):
    """Estadisticas ACOTADAS al mes (a diferencia de _stats_generales_de,
    que es todo el historial): saldo al inicio/fin de mes, ganancia del
    mes, record y efectividad SOLO de ese mes."""
    prefijo = f"{year:04d}-{month:02d}"
    del_mes = _movimientos_del_mes(historial, year, month)
    antes_del_mes = [h for h in historial if (h["fecha"] or "") < prefijo]

    saldo_inicio = antes_del_mes[-1]["saldo"] if antes_del_mes else float(tier)
    saldo_fin = del_mes[-1]["saldo"] if del_mes else saldo_inicio
    saldo_actual = historial[-1]["saldo"] if historial else float(tier)

    apuestas_mes = [h for h in del_mes if h["apuesta"] not in ("Retiro", "Deposito")]
    ganadas = [h for h in apuestas_mes if h["ganada_perdida"] > 0]
    perdidas = [h for h in apuestas_mes if h["ganada_perdida"] < 0]
    total_apostado = sum(h["monto"] for h in apuestas_mes)
    win_rate = (len(ganadas) / len(apuestas_mes) * 100) if apuestas_mes else None

    return {
        "saldo_actual": saldo_actual, "saldo_inicio": saldo_inicio, "saldo_fin": saldo_fin,
        "ganancia_total": saldo_fin - saldo_inicio,
        "n_ganadas": len(ganadas), "n_perdidas": len(perdidas), "win_rate": win_rate,
        "total_apostado": total_apostado, "historial": del_mes, "apuestas": apuestas_mes,
    }


def _parse_fecha(fecha_str):
    for fmt in ("%Y-%m-%d",):
        try:
            return datetime.strptime(fecha_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _render_saldo_chart(historial):
    """Grafica de linea del saldo a traves del tiempo - misma info que
    st.line_chart en la pestana de Analisis de la app, como imagen PNG
    para embeber en el correo (los correos no pueden correr Javascript,
    asi que una grafica interactiva no funciona aqui)."""
    puntos = [(f, h["saldo"]) for h in historial if (f := _parse_fecha(h["fecha"]))]
    if not puntos:
        return None
    puntos.sort(key=lambda p: p[0])
    fechas, saldos = zip(*puntos)

    fig, ax = plt.subplots(figsize=(6, 2.8), dpi=130)
    ax.plot(fechas, saldos, marker="o", color=NEGRO, linewidth=2)
    ax.set_title("Saldo a traves del tiempo", fontsize=11, fontweight="bold", loc="left")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_ganancia_chart(apuestas):
    """Grafica de barras de ganancia/perdida por apuesta - verde si gano,
    rojo si perdio, gris si push, misma info que st.bar_chart en la app."""
    puntos = [(f, h["ganada_perdida"]) for h in apuestas if (f := _parse_fecha(h["fecha"]))]
    if not puntos:
        return None
    puntos.sort(key=lambda p: p[0])
    fechas, valores = zip(*puntos)
    colores = [VERDE if v > 0 else ROJO if v < 0 else GRIS_TEXTO for v in valores]

    fig, ax = plt.subplots(figsize=(6, 2.8), dpi=130)
    ax.bar(range(len(valores)), valores, color=colores)
    ax.set_title("Ganancia/Perdida por apuesta", fontsize=11, fontweight="bold", loc="left")
    ax.axhline(0, color="#cccccc", linewidth=1)
    ax.grid(alpha=0.3, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(range(len(fechas)))
    ax.set_xticklabels([f.strftime("%m-%d") for f in fechas], rotation=30, ha="right", fontsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_report_body(nombre, tier, movimientos, saldo_actual, periodo_label="hoy",
                       mensaje_vacio="No se registro ningun movimiento hoy."):
    """Version en texto plano (respaldo para clientes de correo que no
    muestran HTML)."""
    lines = [f"Hola {nombre},", "", f"Resumen de {periodo_label} - nivel ${tier:,}:", ""]
    if not movimientos:
        lines.append(mensaje_vacio)
    else:
        for m in movimientos:
            apuesta = m.get("Apuesta que se realizo", "")
            momio = m.get("Momio en la que se tomo", "")
            monto = m.get("Inversion actual", "")
            ganada_perdida = m.get("Ganada / Perdida", "")
            if apuesta in ("Retiro", "Deposito"):
                lines.append(f"- {apuesta}: ${monto}")
            else:
                partido = m.get("Partido al que se aposto", "")
                resultado = _resultado_de(ganada_perdida)
                lines.append(f"- {partido}: {apuesta} (momio {momio})")
                lines.append(f"  Invertido: ${monto} | Resultado: {resultado} | Ganancia/Perdida: ${ganada_perdida}")
    lines.append("")
    lines.append(f"Saldo actualizado: ${saldo_actual:,.2f}")
    lines.append("")
    lines.append("- Sistema de Apuestas NBA (mensaje automatico, no responder)")
    return "\n".join(lines)


def _stat_card(label, value, color=NEGRO):
    return (
        f'<td style="padding:12px 8px;text-align:center;border:1px solid #e0e0e0;">'
        f'<div style="font-size:12px;color:{GRIS_TEXTO};margin-bottom:4px;">{html.escape(label)}</div>'
        f'<div style="font-size:18px;font-weight:bold;color:{color};">{html.escape(str(value))}</div>'
        f'</td>'
    )


def build_report_html(nombre, tier, periodo, movimientos, stats, tiene_grafica_saldo=False,
                       tiene_grafica_ganancia=False, titulo_movimientos="Movimientos de hoy",
                       mensaje_vacio="No se registro ningun movimiento hoy.",
                       etiqueta_ganancia="Ganancia/Perdida"):
    ganancia_color = VERDE if stats["ganancia_total"] >= 0 else ROJO
    win_rate_txt = f"{stats['win_rate']:.0f}%" if stats["win_rate"] is not None else "N/D"

    graficas_html = ""
    if tiene_grafica_saldo:
        graficas_html += (
            '<tr><td style="padding:0 20px 10px;">'
            '<img src="cid:saldo_chart" width="560" style="width:100%;max-width:560px;display:block;'
            'border-radius:6px;border:1px solid #eee;" alt="Saldo a traves del tiempo"></td></tr>'
        )
    if tiene_grafica_ganancia:
        graficas_html += (
            '<tr><td style="padding:0 20px 10px;">'
            '<img src="cid:ganancia_chart" width="560" style="width:100%;max-width:560px;display:block;'
            'border-radius:6px;border:1px solid #eee;" alt="Ganancia/Perdida por apuesta"></td></tr>'
        )

    filas_mov = ""
    if not movimientos:
        filas_mov = (
            '<tr><td colspan="5" style="padding:16px;text-align:center;color:'
            f'{GRIS_TEXTO};">{html.escape(mensaje_vacio)}</td></tr>'
        )
    else:
        for m in movimientos:
            apuesta = m.get("Apuesta que se realizo", "")
            if apuesta in ("Retiro", "Deposito"):
                monto = m.get("Inversion actual", "")
                color = ROJO if apuesta == "Retiro" else VERDE
                filas_mov += (
                    f'<tr><td colspan="4" style="padding:10px;border-bottom:1px solid #eee;">{html.escape(apuesta)}</td>'
                    f'<td style="padding:10px;border-bottom:1px solid #eee;text-align:right;color:{color};'
                    f'font-weight:bold;">${html.escape(str(monto))}</td></tr>'
                )
                continue
            partido = m.get("Partido al que se aposto", "")
            momio = m.get("Momio en la que se tomo", "")
            monto = m.get("Inversion actual", "")
            ganada_perdida = m.get("Ganada / Perdida", "")
            resultado = _resultado_de(ganada_perdida)
            color = VERDE if resultado == "GANADA" else ROJO if resultado == "PERDIDA" else GRIS_TEXTO
            filas_mov += (
                '<tr>'
                f'<td style="padding:10px;border-bottom:1px solid #eee;">{html.escape(str(partido))}</td>'
                f'<td style="padding:10px;border-bottom:1px solid #eee;">{html.escape(str(apuesta))}</td>'
                f'<td style="padding:10px;border-bottom:1px solid #eee;text-align:center;">{html.escape(str(momio))}</td>'
                f'<td style="padding:10px;border-bottom:1px solid #eee;text-align:right;">${html.escape(str(monto))}</td>'
                f'<td style="padding:10px;border-bottom:1px solid #eee;text-align:right;color:{color};'
                f'font-weight:bold;">{resultado} (${html.escape(str(ganada_perdida))})</td>'
                '</tr>'
            )

    return f"""\
<html>
<body style="margin:0;padding:0;background:{GRIS_CLARO};font-family:Arial,Helvetica,sans-serif;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{GRIS_CLARO};padding:24px 0;">
<tr><td align="center">
<table role="presentation" width="600" cellpadding="0" cellspacing="0"
       style="background:{BLANCO};border-radius:8px;overflow:hidden;">
  <tr>
    <td style="background:{NEGRO};color:{BLANCO};padding:24px;text-align:center;">
      <div style="font-size:20px;font-weight:bold;">🏀 Reporte de Apuestas NBA</div>
      <div style="font-size:14px;opacity:0.75;margin-top:4px;">{html.escape(nombre)} &middot; nivel ${tier:,} &middot; {html.escape(periodo)}</div>
    </td>
  </tr>
  <tr>
    <td style="padding:20px;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
        <tr>
          {_stat_card("Saldo actual", f"${stats['saldo_actual']:,.2f}")}
          {_stat_card(etiqueta_ganancia, f"${stats['ganancia_total']:,.2f}", ganancia_color)}
        </tr>
        <tr>
          {_stat_card("Record (G-P)", f"{stats['n_ganadas']}-{stats['n_perdidas']}")}
          {_stat_card("Efectividad", win_rate_txt)}
        </tr>
      </table>
    </td>
  </tr>
  {graficas_html}
  <tr>
    <td style="padding:0 20px 20px;">
      <div style="font-size:14px;font-weight:bold;color:{NEGRO};margin-bottom:8px;">{html.escape(titulo_movimientos)}</div>
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
        <tr style="background:{NEGRO};color:{BLANCO};">
          <th style="padding:10px;text-align:left;font-size:12px;">Partido</th>
          <th style="padding:10px;text-align:left;font-size:12px;">Apuesta</th>
          <th style="padding:10px;text-align:center;font-size:12px;">Momio</th>
          <th style="padding:10px;text-align:right;font-size:12px;">Invertido</th>
          <th style="padding:10px;text-align:right;font-size:12px;">Resultado</th>
        </tr>
        {filas_mov}
      </table>
    </td>
  </tr>
  <tr>
    <td style="background:{GRIS_CLARO};padding:14px;text-align:center;font-size:11px;color:{GRIS_TEXTO};">
      Sistema de Apuestas NBA &mdash; mensaje automatico, no responder
    </td>
  </tr>
</table>
</td></tr>
</table>
</body>
</html>
"""


def send_email(to_email, subject, body_text, body_html, gmail_address, gmail_app_password, images=None):
    """images: dict {content_id: bytes_png} - se referencian en el HTML
    como src="cid:content_id" (imagenes normales por URL no funcionan en
    la mayoria de los clientes de correo, hay que embeberlas)."""
    msg = MIMEMultipart("related")
    msg["From"] = gmail_address
    msg["To"] = to_email
    msg["Subject"] = subject

    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(body_text, "plain", "utf-8"))
    alt.attach(MIMEText(body_html, "html", "utf-8"))
    msg.attach(alt)

    for cid, img_bytes in (images or {}).items():
        img = MIMEImage(img_bytes)
        img.add_header("Content-ID", f"<{cid}>")
        img.add_header("Content-Disposition", "inline", filename=f"{cid}.png")
        msg.attach(img)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_address, gmail_app_password)
        server.sendmail(gmail_address, to_email, msg.as_string())


def _movimientos_de_hoy(historial, fecha):
    """Filtra el historial YA leido (get_full_history, llaves
    simplificadas) por fecha y lo regresa con los nombres de columna
    originales del Sheet - para no tener que volver a leer la pestana
    solo para sacar los movimientos de hoy."""
    out = []
    for h in historial:
        if h["fecha"] != fecha:
            continue
        out.append({
            "Partido al que se aposto": h["partido"], "Fecha en la que se aposto": h["fecha"],
            "Apuesta que se realizo": h["apuesta"], "Momio en la que se tomo": h["momio"],
            "Inversion actual": h["monto"], "Ganada / Perdida": h["ganada_perdida"],
            "Inversion despues de apuesta": h["saldo"],
        })
    return out


def _build_and_send_daily(historial, tier, nombre, correo, gmail_address, gmail_app_password, fecha):
    """Arma y manda el correo diario a partir de un historial YA LEIDO
    (individual o de un lote) - separado de send_daily_report para poder
    reusarlo desde send_daily_reports_batch sin volver a leer Sheets."""
    stats = _stats_generales_de(historial, tier)
    movimientos = _movimientos_de_hoy(stats["historial"], fecha)

    images = {}
    saldo_png = _render_saldo_chart(stats["historial"])
    if saldo_png:
        images["saldo_chart"] = saldo_png
    ganancia_png = _render_ganancia_chart(stats["apuestas"])
    if ganancia_png:
        images["ganancia_chart"] = ganancia_png

    body_text = build_report_body(nombre, tier, movimientos, stats["saldo_actual"])
    body_html = build_report_html(nombre, tier, fecha, movimientos, stats,
                                   tiene_grafica_saldo=bool(saldo_png),
                                   tiene_grafica_ganancia=bool(ganancia_png))
    subject = f"Reporte de apuestas - {nombre} - {fecha}"

    if not correo:
        return False, f"{nombre} no tiene correo registrado."
    try:
        send_email(correo, subject, body_text, body_html, gmail_address, gmail_app_password, images=images)
        return True, f"Correo enviado a {correo}."
    except Exception as e:
        return False, f"Fallo el envio a {correo}: {e}"


def send_daily_report(gc, tier, nombre, correo, gmail_address, gmail_app_password, fecha=None):
    """Arma y manda el correo de UN inversionista (lee su historial
    aparte) para `fecha` (default hoy, hora de CDMX). Uso individual (ej.
    el boton de prueba en la app) - para mandarle a todo un nivel de una
    vez usa send_daily_reports_batch, que lee en UNA sola llamada en vez
    de una por persona."""
    fecha = fecha or inv.cdmx_today()
    historial = inv.get_full_history(gc, tier, nombre)
    return _build_and_send_daily(historial, tier, nombre, correo, gmail_address, gmail_app_password, fecha)


def send_daily_reports_batch(gc, tier, investors, gmail_address, gmail_app_password, fecha=None):
    """Manda el correo diario a VARIOS inversionistas del mismo nivel,
    leyendo el historial de TODOS en una sola llamada a la API (en vez de
    una llamada por persona) - la mejora clave para escalar a ~100
    usuarios. `investors` es una lista de dicts {nombre, correo, ...}
    (lo que regresa list_investors). Regresa [(nombre, ok, mensaje), ...]."""
    fecha = fecha or inv.cdmx_today()
    nombres = [i["nombre"] for i in investors]
    historiales = inv.get_full_history_batch(gc, tier, nombres)
    resultados = []
    for i in investors:
        historial = historiales.get(i["nombre"], [])
        ok, msg = _build_and_send_daily(
            historial, tier, i["nombre"], i["correo"], gmail_address, gmail_app_password, fecha)
        resultados.append((i["nombre"], ok, msg))
    return resultados


def _mes_a_reportar(year, month):
    if year is not None and month is not None:
        return year, month
    hoy = inv.cdmx_today()
    anio_hoy, mes_hoy, _ = (int(x) for x in hoy.split("-"))
    if mes_hoy == 1:
        return anio_hoy - 1, 12
    return anio_hoy, mes_hoy - 1


def _build_and_send_monthly(historial_completo, tier, nombre, correo, gmail_address, gmail_app_password,
                             year, month):
    """Arma y manda la recopilacion mensual a partir de un historial YA
    LEIDO - separado de send_monthly_report para poder reusarlo desde
    send_monthly_reports_batch sin volver a leer Sheets."""
    nombre_mes = MESES_ES[month]
    periodo_label = f"{nombre_mes} {year}"

    stats = _stats_del_mes(historial_completo, tier, year, month)
    movimientos = [
        {
            "Partido al que se aposto": h["partido"], "Fecha en la que se aposto": h["fecha"],
            "Apuesta que se realizo": h["apuesta"], "Momio en la que se tomo": h["momio"],
            "Inversion actual": h["monto"], "Ganada / Perdida": h["ganada_perdida"],
            "Inversion despues de apuesta": h["saldo"],
        }
        for h in stats["historial"]
    ]

    images = {}
    saldo_png = _render_saldo_chart(stats["historial"])
    if saldo_png:
        images["saldo_chart"] = saldo_png
    ganancia_png = _render_ganancia_chart(stats["apuestas"])
    if ganancia_png:
        images["ganancia_chart"] = ganancia_png

    body_text = build_report_body(
        nombre, tier, movimientos, stats["saldo_actual"], periodo_label=nombre_mes.lower(),
        mensaje_vacio=f"No se registraron movimientos en {nombre_mes} {year}.")
    body_html = build_report_html(
        nombre, tier, periodo_label, movimientos, stats,
        tiene_grafica_saldo=bool(saldo_png), tiene_grafica_ganancia=bool(ganancia_png),
        titulo_movimientos=f"Movimientos de {nombre_mes} {year}",
        mensaje_vacio=f"No se registraron movimientos en {nombre_mes} {year}.",
        etiqueta_ganancia=f"Ganancia/Perdida ({nombre_mes})")
    subject = f"Recopilacion mensual - {nombre} - {nombre_mes} {year}"

    if not correo:
        return False, f"{nombre} no tiene correo registrado."
    try:
        send_email(correo, subject, body_text, body_html, gmail_address, gmail_app_password, images=images)
        return True, f"Correo mensual enviado a {correo}."
    except Exception as e:
        return False, f"Fallo el envio mensual a {correo}: {e}"


def send_monthly_report(gc, tier, nombre, correo, gmail_address, gmail_app_password, year=None, month=None):
    """Arma y manda la recopilacion mensual de UN inversionista (lee su
    historial aparte): movimientos de `month`/`year` (default el mes que
    acaba de terminar, hora CDMX), saldo al inicio/fin de ese mes, record
    y efectividad SOLO de ese mes. Uso individual - para todo un nivel de
    una vez usa send_monthly_reports_batch."""
    year, month = _mes_a_reportar(year, month)
    historial_completo = inv.get_full_history(gc, tier, nombre)
    return _build_and_send_monthly(
        historial_completo, tier, nombre, correo, gmail_address, gmail_app_password, year, month)


def send_monthly_reports_batch(gc, tier, investors, gmail_address, gmail_app_password, year=None, month=None):
    """Manda la recopilacion mensual a VARIOS inversionistas del mismo
    nivel, leyendo el historial de TODOS en una sola llamada a la API -
    ver send_daily_reports_batch, mismo patron. Regresa
    [(nombre, ok, mensaje), ...]."""
    year, month = _mes_a_reportar(year, month)
    nombres = [i["nombre"] for i in investors]
    historiales = inv.get_full_history_batch(gc, tier, nombres)
    resultados = []
    for i in investors:
        historial = historiales.get(i["nombre"], [])
        ok, msg = _build_and_send_monthly(
            historial, tier, i["nombre"], i["correo"], gmail_address, gmail_app_password, year, month)
        resultados.append((i["nombre"], ok, msg))
    return resultados
