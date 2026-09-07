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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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


def _stats_generales(gc, tier, nombre):
    """Mismas estadisticas que la pestana de Analisis de la app: saldo
    actual, ganancia/perdida total, record ganadas-perdidas, efectividad,
    total apostado - se calculan sobre el historial COMPLETO, no solo el
    dia de hoy."""
    historial = inv.get_full_history(gc, tier, nombre)
    saldo_actual = inv.get_investor_balance(gc, tier, nombre)
    ganancia_total = saldo_actual - tier
    apuestas = [h for h in historial if h["apuesta"] not in ("Retiro", "Deposito")]
    ganadas = [h for h in apuestas if h["ganada_perdida"] > 0]
    perdidas = [h for h in apuestas if h["ganada_perdida"] < 0]
    total_apostado = sum(h["monto"] for h in apuestas)
    win_rate = (len(ganadas) / len(apuestas) * 100) if apuestas else None
    return {
        "saldo_actual": saldo_actual, "ganancia_total": ganancia_total,
        "n_ganadas": len(ganadas), "n_perdidas": len(perdidas), "win_rate": win_rate,
        "total_apostado": total_apostado,
    }


def build_report_body(nombre, tier, movimientos, saldo_actual):
    """Version en texto plano (respaldo para clientes de correo que no
    muestran HTML)."""
    lines = [f"Hola {nombre},", "", f"Resumen de hoy - nivel ${tier:,}:", ""]
    if not movimientos:
        lines.append("No se registro ningun movimiento hoy.")
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


def build_report_html(nombre, tier, fecha, movimientos, stats):
    ganancia_color = VERDE if stats["ganancia_total"] >= 0 else ROJO
    win_rate_txt = f"{stats['win_rate']:.0f}%" if stats["win_rate"] is not None else "N/D"

    filas_mov = ""
    if not movimientos:
        filas_mov = (
            '<tr><td colspan="5" style="padding:16px;text-align:center;color:'
            f'{GRIS_TEXTO};">No se registro ningun movimiento hoy.</td></tr>'
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
      <div style="font-size:14px;opacity:0.75;margin-top:4px;">{html.escape(nombre)} &middot; nivel ${tier:,} &middot; {fecha}</div>
    </td>
  </tr>
  <tr>
    <td style="padding:20px;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border-collapse:collapse;">
        <tr>
          {_stat_card("Saldo actual", f"${stats['saldo_actual']:,.2f}")}
          {_stat_card("Ganancia/Perdida", f"${stats['ganancia_total']:,.2f}", ganancia_color)}
        </tr>
        <tr>
          {_stat_card("Record (G-P)", f"{stats['n_ganadas']}-{stats['n_perdidas']}")}
          {_stat_card("Efectividad", win_rate_txt)}
        </tr>
      </table>
    </td>
  </tr>
  <tr>
    <td style="padding:0 20px 20px;">
      <div style="font-size:14px;font-weight:bold;color:{NEGRO};margin-bottom:8px;">Movimientos de hoy</div>
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


def send_email(to_email, subject, body_text, body_html, gmail_address, gmail_app_password):
    msg = MIMEMultipart("alternative")
    msg["From"] = gmail_address
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    msg.attach(MIMEText(body_html, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_address, gmail_app_password)
        server.sendmail(gmail_address, to_email, msg.as_string())


def send_daily_report(gc, tier, nombre, correo, gmail_address, gmail_app_password, fecha=None):
    """Arma y manda el correo de un inversionista para `fecha` (default
    hoy, hora de CDMX - no la hora del servidor). Regresa (enviado: bool,
    mensaje: str)."""
    fecha = fecha or inv.cdmx_today()

    movimientos = inv.get_bets_for_date(gc, tier, nombre, fecha)
    stats = _stats_generales(gc, tier, nombre)
    body_text = build_report_body(nombre, tier, movimientos, stats["saldo_actual"])
    body_html = build_report_html(nombre, tier, fecha, movimientos, stats)
    subject = f"Reporte de apuestas - {nombre} - {fecha}"

    if not correo:
        return False, f"{nombre} no tiene correo registrado."
    try:
        send_email(correo, subject, body_text, body_html, gmail_address, gmail_app_password)
        return True, f"Correo enviado a {correo}."
    except Exception as e:
        return False, f"Fallo el envio a {correo}: {e}"
