import pandas as pd
import numpy as np

# Lista de 25 características clínicas usadas (debe coincidir con el modelo)
FEATURE_COLUMNS = [
    'asymmetry_score', 'border_irregularity', 'color_variation', 'diameter',
    'contrast', 'energy', 'homogeneity', 'correlation',
    'eccentricity', 'compactness', 'area_ratio',
    'age', 'gender', 'family_history', 'sun_exposure',
    'texture_roughness', 'lesion_shape', 'color_uniformity',
    'edge_sharpness', 'surface_smoothness', 'pattern_symmetry',
    'vascularity', 'pigment_network', 'streaks', 'regression_structures'
]

def load_metadata(csv_path):
    """
    Carga el archivo de metadatos.
    
    Args:
        csv_path (str): Ruta al archivo CSV.
        
    Returns:
        pd.DataFrame: DataFrame con los datos.
    """
    df = pd.read_csv(csv_path, index_col=False)
    # Asegurarse de que la columna 'label' sea de tipo entero
    df['label'] = df['label'].astype(int)
    return df

def get_features_for_image(metadata_df, image_id):
    """
    Obtiene el vector de características para una imagen específica.
    
    Args:
        metadata_df (pd.DataFrame): DataFrame con todos los metadatos.
        image_id (str): ID único de la imagen (nombre del archivo sin extensión).
        
    Returns:
        np.ndarray: Vector de características (shape: [1, num_features]).
    """
    # Convertir image_id a cadena si es necesario
    image_id = str(image_id)
    
    row = metadata_df[metadata_df['image_id'] == image_id]
    if row.empty:
        raise ValueError(f"No se encontró la imagen {image_id} en el metadata.")
    
    features = row[FEATURE_COLUMNS].values.astype(np.float32)
    return features

if __name__ == '__main__':
    print("Funciones de extracción de características listas.")