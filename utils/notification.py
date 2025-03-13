import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import logging

# Cargar variables de entorno si existe un archivo .env
env_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

def send_telegram_alert(message, image_path=None):
    """
    Envía un mensaje de alerta a través de Telegram con una imagen opcional
    
    Args:
        message (str): El mensaje a enviar
        image_path (str, optional): Ruta a una imagen para enviar
        
    Returns:
        bool: True si se envió con éxito, False en caso contrario
    """
    try:
        # Obtener token y chat_id desde variables de entorno
        token = os.getenv('TELEGRAM_TOKEN')
        chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if not token or not chat_id:
            logging.error("Configuración de Telegram no encontrada. Verifica variables de entorno.")
            return False
        
        # Preparar la URL para el API de Telegram
        base_url = f"https://api.telegram.org/bot{token}"
        
        # Si hay una imagen, enviar como photo
        if image_path and os.path.exists(image_path):
            url = f"{base_url}/sendPhoto"
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': chat_id, 'caption': message, 'parse_mode': 'Markdown'}
                response = requests.post(url, data=data, files=files)
        else:
            # Si no hay imagen, enviar como mensaje de texto
            url = f"{base_url}/sendMessage"
            data = {'chat_id': chat_id, 'text': message, 'parse_mode': 'Markdown'}
            response = requests.post(url, data=data)
        
        # Verificar respuesta
        if response.status_code == 200 and response.json().get('ok'):
            logging.info("Mensaje enviado exitosamente a Telegram")
            return True
        else:
            logging.error(f"Error enviando mensaje: {response.text}")
            return False
    
    except Exception as e:
        logging.error(f"Error en send_telegram_alert: {e}")
        return False
