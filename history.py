import json
import math
import os
from datetime import datetime


def _sanitize(obj):
    """
    Recorre recursivamente un objeto (dict, list, float) y reemplaza
    valores NaN e Inf por None, que es válido en JSON (null).
    Esto evita el ValueError 'Out of range float values are not JSON compliant'.
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(item) for item in obj]
    return obj

# Ruta relativa desde api/backend al directorio de datos
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
HISTORY_DIR = os.path.join(DATA_DIR, 'history')

def get_history_file(username):
    """Obtiene la ruta del archivo de historial para un usuario."""
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR, exist_ok=True)
    return os.path.join(HISTORY_DIR, f"{username}_history.json")

def load_history(username):
    """Carga el historial de un usuario."""
    filepath = get_history_file(username)
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('analyses', [])
    except:
        return []

def save_history_file(username, history):
    """Guarda el historial completo de un usuario, sanitizando NaN/Inf."""
    filepath = get_history_file(username)
    try:
        clean_history = _sanitize(history)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"analyses": clean_history}, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error guardando historial: {e}")
        return False

def save_analysis(username, image_name, prediction, confidence, features):
    """Guarda un nuevo análisis en el historial."""
    history = load_history(username)
    
    new_entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image_name,
        "prediction": prediction,
        "confidence": float(confidence),
        "top_features": features
    }
    
    # Agregar al inicio de la lista
    history.insert(0, new_entry)
    
    return save_history_file(username, history)

def delete_analysis(username, analysis_id):
    """Elimina un análisis específico del historial."""
    history = load_history(username)
    history = [entry for entry in history if entry.get('id') != analysis_id]
    return save_history_file(username, history)

def clear_history(username):
    """Elimina todo el historial de un usuario."""
    return save_history_file(username, [])

def get_analysis_count(username):
    """Retorna el número total de análisis realizados por el usuario."""
    history = load_history(username)
    return len(history)
