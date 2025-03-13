import qrcode
import socket
from datetime import datetime
import os
import webbrowser

def get_local_ip():
    """Obtener la dirección IP local del ordenador"""
    try:
        # Crear un socket para determinar qué dirección IP usar para conectarse a internet
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        # Fallback a un método alternativo
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)

def generate_access_qr(port=8050):
    """Generar código QR para acceso rápido desde móvil"""
    ip = get_local_ip()
    url = f"http://{ip}:{port}"
    
    # Carpeta para guardar el QR
    qr_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
    os.makedirs(qr_dir, exist_ok=True)
    
    # Generar y guardar el QR
    qr_file = os.path.join(qr_dir, f"dashboard_access_qr.png")
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(qr_file)
    
    print(f"✅ Acceso Dashboard - IP: {ip}:{port}")
    print(f"📱 Escanea el código QR para acceder rápidamente desde tu móvil")
    print(f"   QR guardado en: {qr_file}")
    
    # Intentar abrir el QR automáticamente
    try:
        webbrowser.open(qr_file)
    except:
        pass
    
    return url, qr_file

if __name__ == "__main__":
    generate_access_qr()
