import argparse
import yaml
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import tensorflow as tf

from src.preprocesamiento import load_images
from src.extraccion_caracteristicas import load_metadata, get_features_for_image

def evaluate(config):
    """Evalúa el rendimiento de los modelos entrenados."""
    print(f"Cargando configuración desde: {config}")
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. Cargar metadatos
    print("Cargando metadatos...")
    metadata_path = cfg['dataset']['path']
    metadata_df = load_metadata(metadata_path)
    
    # 2. Crear una partición balanceada para prueba
    # Tomar una muestra balanceada: 50 nevos y 50 melanomas (o el máximo disponible)
    nevus_df = metadata_df[metadata_df['label'] == 0]
    melanoma_df = metadata_df[metadata_df['label'] == 1]
    
    # Tomar hasta 50 de cada clase
    n_samples = min(50, len(nevus_df), len(melanoma_df))
    nevus_sample = nevus_df.sample(n=n_samples, random_state=42)
    melanoma_sample = melanoma_df.sample(n=n_samples, random_state=42)
    
    # Combinar y mezclar
    test_df = pd.concat([nevus_sample, melanoma_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Usando {len(test_df)} muestras balanceadas como conjunto de prueba ({n_samples} nevos, {n_samples} melanomas)")
    
    # 3. Cargar datos de test
    print("Cargando imágenes de prueba...")
    test_image_ids = test_df['image_id'].tolist()
    X_test_img = load_images(test_image_ids, cfg['dataset']['image_dir'], (cfg['dataset']['image_size'], cfg['dataset']['image_size']))
    
    print("Extrayendo características de prueba...")
    X_test_feat = np.vstack([get_features_for_image(metadata_df, img_id) for img_id in test_image_ids])
    
    y_test = test_df['label'].values.astype(np.float32)
    
    # 4. Cargar modelos y evaluar
    results_dir = cfg['output']['results_dir']
    all_preds = []
    
    # Cargar y evaluar cada modelo fold
    for fold in range(cfg['dataset']['n_splits']):
        model_path = f'{results_dir}best_model_fold_{fold+1}.h5'
        if os.path.exists(model_path):
            print(f"Evaluando modelo del fold {fold+1}...")
            try:
                # Cargar modelo con compatibilidad
                model = load_model(model_path, compile=False)
                # Recompilar con las mismas opciones del entrenamiento
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                preds = model.predict([X_test_img, X_test_feat], verbose=0)
                all_preds.append(preds)
            except Exception as e:
                print(f"Error al cargar modelo del fold {fold+1}: {e}")
                # Intentar cargar con custom objects si es necesario
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    preds = model.predict([X_test_img, X_test_feat], verbose=0)
                    all_preds.append(preds)
                except Exception as e2:
                    print(f"Error al cargar modelo del fold {fold+1} (segundo intento): {e2}")
        else:
            print(f"Modelo del fold {fold+1} no encontrado: {model_path}")
    
    if not all_preds:
        print("No se encontraron modelos para evaluar")
        return
    
    # 5. Calcular predicciones promedio
    avg_preds = np.mean(all_preds, axis=0)
    binary_preds = (avg_preds > 0.5).astype(int)
    
    # 6. Calcular métricas
    accuracy = accuracy_score(y_test, binary_preds)
    precision = precision_score(y_test, binary_preds, zero_division=0)
    recall = recall_score(y_test, binary_preds, zero_division=0)
    f1 = f1_score(y_test, binary_preds, zero_division=0)
    auc = roc_auc_score(y_test, avg_preds)
    
    print("\n--- Métricas de Evaluación ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Sensibilidad (Recall): {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_test, binary_preds, target_names=['Nevus', 'Melanoma'], zero_division=0))
    
    # 7. Matriz de Confusión
    cm = confusion_matrix(y_test, binary_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Nevus', 'Melanoma'], yticklabels=['Nevus', 'Melanoma'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.savefig(f"{results_dir}confusion_matrix.png")
    plt.show()
    print(f"\nMatriz de confusión guardada en '{results_dir}confusion_matrix.png'")
    
    # 8. Guardar resultados en CSV
    results_df = pd.DataFrame({
        'image_id': test_image_ids,
        'real': y_test,
        'predicho': binary_preds.flatten(),
        'probabilidad': avg_preds.flatten()
    })
    results_df.to_csv(f"{results_dir}resultados_prueba.csv", index=False)
    print(f"Resultados detallados guardados en '{results_dir}resultados_prueba.csv'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluar modelos de melanoma.')
    parser.add_argument('--config', type=str, required=True, help='Ruta al archivo de configuración YAML.')
    args = parser.parse_args()
    evaluate(args.config)