"""
Envio de reportes por correo a los inversionistas: resumen de los
movimientos (apuestas/retiros/depositos) registrados HOY, con su saldo
actualizado - via Gmail SMTP.

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

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import nba_investors as inv


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


def build_report_body(nombre, tier, movimientos, saldo_actual):
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


def send_email(to_email, subject, body, gmail_address, gmail_app_password):
    msg = MIMEMultipart()
    msg["From"] = gmail_address
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_address, gmail_app_password)
        server.sendmail(gmail_address, to_email, msg.as_string())


def send_daily_report(gc, tier, nombre, correo, gmail_address, gmail_app_password, fecha=None):
    """Arma y manda el correo de un inversionista para `fecha` (default
    hoy, hora de CDMX - no la hora del servidor). Regresa (enviado: bool,
    mensaje: str)."""
    fecha = fecha or inv.cdmx_today()

    movimientos = inv.get_bets_for_date(gc, tier, nombre, fecha)
    saldo = inv.get_investor_balance(gc, tier, nombre)
    body = build_report_body(nombre, tier, movimientos, saldo)
    subject = f"Reporte de apuestas - {nombre} - {fecha}"

    if not correo:
        return False, f"{nombre} no tiene correo registrado."
    try:
        send_email(correo, subject, body, gmail_address, gmail_app_password)
        return True, f"Correo enviado a {correo}."
    except Exception as e:
        return False, f"Fallo el envio a {correo}: {e}"
