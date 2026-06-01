import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import StratifiedKFold
from src.extraccion_caracteristicas import FEATURE_COLUMNS

class UnifiedDataLoader:
    """
    Cargador de datos unificado que puede manejar múltiples conjuntos de datos,
    incluyendo conjuntos de datos con y sin etiquetas.
    """
    
    def __init__(self, datasets_config):
        """
        Inicializa el cargador de datos unificado.
        
        Args:
            datasets_config (dict): Configuración de los conjuntos de datos.
                                  Debe contener información sobre rutas, directorios de imágenes, etc.
        """
        self.datasets_config = datasets_config
        self.labeled_data = None
        self.unlabeled_data = None
        self._project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _resolve_path(self, p):
        if p is None:
            return None
        if isinstance(p, list):
            return [self._resolve_path(x) for x in p]
        if os.path.isabs(p):
            return p
        return os.path.join(self._project_root, p)
        
    def load_labeled_dataset(self, dataset_name, csv_path, image_dir):
        """
        Carga un conjunto de datos etiquetado.
        
        Args:
            dataset_name (str): Nombre del conjunto de datos.
            csv_path (str): Ruta al archivo CSV con metadatos.
            image_dir (str): Directorio con las imágenes.
            
        Returns:
            pd.DataFrame: DataFrame con los datos etiquetados.
        """
        print(f"Cargando conjunto de datos etiquetado: {dataset_name}")
        resolved_csv = self._resolve_path(csv_path)
        if not os.path.exists(resolved_csv):
            raise FileNotFoundError(f"No se encuentra el CSV: {resolved_csv}")
        df = pd.read_csv(resolved_csv, index_col=False)
        df['dataset'] = dataset_name
        df['has_labels'] = True
        # Handle multiple image directories
        if isinstance(image_dir, list):
            # Multiple directories - search in all of them
            df['image_path'] = df['image_id'].apply(lambda x: self._find_image_in_directories(x, image_dir))
        else:
            # Single directory
            df['image_path'] = df['image_id'].apply(lambda x: self._find_image_in_subdirectories(x, image_dir))
        return df
    
    def load_unlabeled_dataset(self, dataset_name, image_dir, label_mapping=None):
        """
        Carga un conjunto de datos no etiquetado (solo imágenes).
        
        Args:
            dataset_name (str): Nombre del conjunto de datos.
            image_dir (str or list): Directorio(s) con las imágenes.
            label_mapping (dict): Diccionario opcional para mapear image_ids a etiquetas.
            
        Returns:
            pd.DataFrame: DataFrame con los datos no etiquetados.
        """
        print(f"Cargando conjunto de datos no etiquetado: {dataset_name}")
        
        # Obtener todos los archivos de imagen en el directorio(s)
        image_files = []
        image_paths = []
        
        # Handle multiple directories
        directories = image_dir if isinstance(image_dir, list) else [image_dir]
        
        for directory in directories:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_files.append(file)
                        image_paths.append(os.path.join(root, file))
        
        # Crear DataFrame con información básica
        image_ids = [os.path.splitext(f)[0] for f in image_files]
        df = pd.DataFrame({
            'image_id': image_ids,
            'image_filename': image_files,
            'dataset': dataset_name,
            'has_labels': False
        })
        df['image_path'] = image_paths
        
        # Si se proporciona un mapeo de etiquetas, intentar asignar etiquetas
        if label_mapping is not None:
            df['label'] = df['image_id'].map(label_mapping)
            # Marcar como etiquetado si se encontraron etiquetas
            df['has_labels'] = df['label'].notna()
            print(f"  Etiquetas recuperadas para {df['has_labels'].sum()} imágenes")
        
        return df
    
    def load_all_datasets(self):
        """
        Carga todos los conjuntos de datos según la configuración.
        """
        all_dataframes = []
        
        for dataset_name, config in self.datasets_config.items():
            # Handle multiple image directories
            image_dir = config['image_dir']
            if isinstance(image_dir, str) and ',' in image_dir:
                image_dir = [path.strip() for path in image_dir.split(',')]
            image_dir = self._resolve_path(image_dir)
            
            if config.get('has_labels', True):
                # Conjunto de datos etiquetado
                df = self.load_labeled_dataset(
                    dataset_name, 
                    config['csv_path'], 
                    image_dir
                )
                all_dataframes.append(df)
            else:
                # Conjunto de datos no etiquetado
                label_mapping = config.get('label_mapping', None)
                df = self.load_unlabeled_dataset(
                    dataset_name, 
                    image_dir,
                    label_mapping
                )
                all_dataframes.append(df)
        
        # Combinar todos los DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        return combined_df
    
    def prepare_labeled_data(self, df):
        """
        Prepara los datos etiquetados para el entrenamiento.
        
        Args:
            df (pd.DataFrame): DataFrame con datos etiquetados.
            
        Returns:
            tuple: (X_features, y_labels, image_paths)
        """
        labeled_df = df[df['has_labels'] == True].copy()
        
        if labeled_df.empty:
            return None, None, None
            
        # Extraer características
        X_features = np.vstack([
            labeled_df[FEATURE_COLUMNS].values.astype(np.float32)
        ])
        
        # Etiquetas
        y_labels = labeled_df['label'].values.astype(np.float32)
        
        # Rutas de imágenes
        image_paths = labeled_df['image_path'].tolist()
        
        return X_features, y_labels, image_paths
    
    def prepare_unlabeled_data(self, df):
        """
        Prepara los datos no etiquetados para el entrenamiento semi-supervisado.
        
        Args:
            df (pd.DataFrame): DataFrame con datos no etiquetados.
            
        Returns:
            list: Lista de rutas de imágenes no etiquetadas.
        """
        unlabeled_df = df[df['has_labels'] == False].copy()
        
        if unlabeled_df.empty:
            return None
            
        # Solo necesitamos las rutas de las imágenes
        image_paths = unlabeled_df['image_path'].tolist()
        return image_paths
    
    def _find_image_in_directories(self, image_id, directories):
        """
        Busca una imagen en múltiples directorios.
        
        Args:
            image_id (str): ID de la imagen.
            directories (list): Lista de directorios donde buscar.
            
        Returns:
            str: Ruta completa a la imagen, o None si no se encuentra.
        """
        image_id_lower = str(image_id).lower()
        for directory in directories:
            for root, dirs, files in os.walk(directory):
                for f in files:
                    name, ext = os.path.splitext(f)
                    if ext.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'):
                        if name.lower() == image_id_lower:
                            return os.path.join(root, f)
        return None
    
    def _find_image_in_subdirectories(self, image_id, base_directory):
        """
        Busca una imagen en subdirectorios.
        
        Args:
            image_id (str): ID de la imagen.
            base_directory (str): Directorio base donde buscar en subdirectorios.
            
        Returns:
            str: Ruta completa a la imagen, o None si no se encuentra.
        """
        image_id_lower = str(image_id).lower()
        for root, dirs, files in os.walk(base_directory):
            for f in files:
                name, ext = os.path.splitext(f)
                if ext.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'):
                    if name.lower() == image_id_lower:
                        return os.path.join(root, f)
        return None

# Función auxiliar para crear folds estratificados considerando múltiples datasets
def create_stratified_folds(df, n_splits=5, random_state=42):
    """
    Crea folds estratificados para validación cruzada.
    
    Args:
        df (pd.DataFrame): DataFrame con datos etiquetados.
        n_splits (int): Número de divisiones para validación cruzada.
        random_state (int): Semilla para reproducibilidad.
        
    Returns:
        list: Lista de tuplas (train_idx, val_idx) para cada fold.
    """
    labeled_df = df[df['has_labels'] == True].copy()
    
    if labeled_df.empty:
        raise ValueError("No hay datos etiquetados disponibles para crear folds estratificados.")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    
    for train_idx, val_idx in skf.split(labeled_df, labeled_df['label']):
        # Ajustar índices al DataFrame completo
        train_idx_global = labeled_df.index[train_idx].tolist()
        val_idx_global = labeled_df.index[val_idx].tolist()
        folds.append((train_idx_global, val_idx_global))
    
    return folds

if __name__ == '__main__':
    print("Módulo de carga de datos unificada listo.")
