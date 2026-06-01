import tensorflow as tf
import numpy as np

class DataAugmentor:
    """
    Clase para realizar aumento de datos en imágenes médicas.
    """
    
    def __init__(self, config=None):
        """
        Inicializa el aumentador de datos.
        
        Args:
            config (dict): Configuración de aumento de datos.
        """
        self.config = config or {}
        # Configuración por defecto
        self.rotation_range = self.config.get('rotation_range', 20)
        self.width_shift_range = self.config.get('width_shift_range', 0.1)
        self.height_shift_range = self.config.get('height_shift_range', 0.1)
        self.horizontal_flip = self.config.get('horizontal_flip', True)
        self.vertical_flip = self.config.get('vertical_flip', True)
        self.brightness_range = self.config.get('brightness_range', 0.1)
        self.zoom_range = self.config.get('zoom_range', 0.1)
        
    def augment_image(self, image):
        """
        Aplica transformaciones aleatorias a una sola imagen.
        
        Args:
            image: Array numpy (H, W, 3) normalizado [0,1]
            
        Returns:
            Imagen aumentada (H, W, 3)
        """
        # Convertir a tensor
        img = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # 1. Rotación aleatoria (usando rot90 como aproximación simple o tf.image.rot90)
        # Para rotación fina se requeriría tfa.image.rotate, pero para evitar dependencias extra
        # usaremos flip y transformaciones simples disponibles en tf.image
        
        # 2. Flip Horizontal
        if self.horizontal_flip:
            img = tf.image.random_flip_left_right(img)
            
        # 3. Flip Vertical
        if self.vertical_flip:
            img = tf.image.random_flip_up_down(img)
            
        # 4. Brillo
        # brightness_range puede ser:
        #  - un número (max_delta) -> se usa tf.image.random_brightness
        #  - una lista/tupla [min_factor, max_factor] -> se multiplica la imagen por un factor aleatorio
        try:
            br = self.brightness_range
        except AttributeError:
            br = None

        if isinstance(br, (list, tuple)) and len(br) == 2:
            # Multiplicative brightness: sample factor en [min, max]
            minf = float(br[0])
            maxf = float(br[1])
            if maxf > minf:
                factor = tf.random.uniform([], minf, maxf)
                img = img * factor
        else:
            # Si es un número y positivo, usar random_brightness (delta aditivo)
            try:
                if br is not None and float(br) > 0:
                    img = tf.image.random_brightness(img, max_delta=float(br))
            except Exception:
                # En caso de tipos inesperados, no aplicar brillo
                pass
            
        # 5. Contraste
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        
        # 6. Saturación
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        
        # Asegurar rango [0,1]
        img = tf.clip_by_value(img, 0.0, 1.0)
        
        return img.numpy()

    def augment_batch(self, images, num_augmentations=1):
        """
        Aumenta un lote de imágenes.
        Genera pares [Original, Aumentada] para coincidir con np.repeat de etiquetas.
        
        Args:
            images: Array numpy (N, H, W, 3)
            num_augmentations: Cuántas aumentaciones por imagen (actualmente soporta 1)
            
        Returns:
            Array numpy (N * (1 + num_augmentations), H, W, 3)
        """
        augmented_batch = []
        
        for img in images:
            # 1. Agregar imagen original
            augmented_batch.append(img)
            
            # 2. Agregar imagen(es) aumentada(s)
            for _ in range(num_augmentations):
                aug_img = self.augment_image(img)
                augmented_batch.append(aug_img)
                
        return np.array(augmented_batch)
