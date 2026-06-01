import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(300, 300)):
    """
    Carga y preprocesa una imagen para el modelo.
    
    Args:
        image_path (str): Ruta a la imagen.
        target_size (tuple): Tamaño objetivo (ancho, alto).
        
    Returns:
        np.ndarray: Imagen preprocesada con forma (1, altura, ancho, 3).
    """
    # Cargar imagen
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Asegurar que la imagen tiene 3 canales (RGB)
    if len(image.shape) == 2:  # Imagen en escala de grises
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        # Convertir BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Redimensionar
    image = cv2.resize(image, target_size)
    
    # Normalizar a [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Añadir dimensión de batch
    image = np.expand_dims(image, axis=0)
    
    return image

def preprocess_image_from_array(image_array, target_size=(300, 300)):
    """
    Preprocesa una imagen desde un array numpy para el modelo.
    
    Args:
        image_array (np.ndarray): Array de la imagen en formato BGR.
        target_size (tuple): Tamaño objetivo (ancho, alto).
        
    Returns:
        np.ndarray: Imagen preprocesada con forma (1, altura, ancho, 3).
    """
    # Trabajar con una copia de la imagen
    image = image_array.copy()
    
    # Asegurar que la imagen tiene 3 canales (RGB)
    if len(image.shape) == 2:  # Imagen en escala de grises
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:  # Imagen en escala de grises con dimensión explícita
        image = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 3:
        # Convertir BGR a RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.shape[2] == 4:  # Imagen RGBA
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    
    # Redimensionar
    image = cv2.resize(image, target_size)
    
    # Normalizar a [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Añadir dimensión de batch
    image = np.expand_dims(image, axis=0)
    
    return image


def find_image_path(image_id, image_dir):
    """
    Busca la ruta de una imagen por su ID en el directorio especificado.
    
    Args:
        image_id (str): ID de la imagen.
        image_dir (str): Directorio donde buscar la imagen.
        
    Returns:
        str: Ruta completa a la imagen.
        
    Raises:
        FileNotFoundError: Si no se encuentra la imagen.
    """
    # Convertir image_id a cadena si es necesario
    image_id = str(image_id)
    
    # Extensiones comunes
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Formatos comunes de nombre
    formats = [
        image_id,  # Nombre original
        f"{image_id}",  # Sin cambios
        f"img_{image_id}",  # Prefijo img_
        f"{image_id}_image",  # Sufijo _image
        f"{image_id}.jpg",  # Con extensión directa
        f"img_{image_id}.jpg",  # Prefijo + extensión
    ]
    
    # Buscar en todas las combinaciones
    for fmt in formats:
        for ext in extensions:
            # Caso 1: fmt ya incluye extensión
            if any(fmt.lower().endswith(e) for e in extensions):
                path = os.path.join(image_dir, fmt)
                if os.path.exists(path):
                    return path
                continue
            
            # Caso 2: fmt no incluye extensión
            path = os.path.join(image_dir, f"{fmt}{ext}")
            if os.path.exists(path):
                return path
    
    # Si no se encuentra, lanzar error
    raise FileNotFoundError(f"No se encontró imagen para ID '{image_id}' en '{image_dir}'")

def load_images(image_ids, image_dir, target_size=(300, 300)):
    """
    Carga y preprocesa múltiples imágenes.
    
    Args:
        image_ids (list): Lista de IDs de imágenes.
        image_dir (str): Directorio donde se encuentran las imágenes.
        target_size (tuple): Tamaño objetivo (ancho, alto).
        
    Returns:
        np.ndarray: Array de imágenes preprocesadas con forma (N, altura, ancho, 3).
    """
    images = []
    for image_id in image_ids:
        try:
            image_path = find_image_path(image_id, image_dir)
            image = preprocess_image(image_path, target_size)
            images.append(image)
        except FileNotFoundError as e:
            print(f"Advertencia: {e}")
            # Crear una imagen de ruido con la forma correcta
            dummy_image = np.random.rand(1, target_size[0], target_size[1], 3).astype(np.float32)
            images.append(dummy_image)
    return np.vstack(images)

if __name__ == '__main__':
    # Ejemplo de uso
    print("Función de preprocesamiento lista.")