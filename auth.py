import json
import os
from datetime import datetime, timedelta
import uuid

# Ruta relativa desde api/backend al directorio de datos
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
USERS_FILE = os.path.join(DATA_DIR, 'users.json')

def load_users():
    """Carga los usuarios desde el archivo JSON."""
    if not os.path.exists(USERS_FILE):
        return []
    
    try:
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('users', [])
    except Exception as e:
        print(f"Error cargando usuarios: {e}")
        return []

def authenticate(username, password):
    """
    Verifica las credenciales del usuario.
    Retorna el objeto usuario si es válido, None si no.
    """
    users = load_users()
    for user in users:
        if user['username'] == username and user['password'] == password:
            return user
    return None

def register_user(username, password, name, role="doctor"):
    """Registra un nuevo usuario en el sistema."""
    users = load_users()
    
    # Verificar si el username ya existe
    if any(u['username'] == username for u in users):
        return False, "El nombre de usuario ya existe"
    
    # Crear nuevo usuario
    new_user = {
        "username": username,
        "password": password,
        "name": name,
        "role": role
    }
    
    users.append(new_user)
    
    # Asegurar que el directorio existe
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
    
    # Guardar en archivo
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump({"users": users}, f, indent=2, ensure_ascii=False)
        return True, "Usuario registrado exitosamente"
    except Exception as e:
        return False, f"Error al guardar: {e}"

def get_user_by_username(username):
    """Obtiene un usuario por su username."""
    users = load_users()
    for user in users:
        if user['username'] == username:
            return user
    return None
