import argparse
import yaml
import pandas as pd
import numpy as np
import os
import psutil
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

try:
    from src.modelos import create_hybrid_model
    from src.preprocesamiento import preprocess_image, load_images
    from src.extraccion_caracteristicas import load_metadata, get_features_for_image, FEATURE_COLUMNS
    from src.carga_datos_unificada import UnifiedDataLoader, create_stratified_folds
    from src.aumento_datos import DataAugmentor
except ModuleNotFoundError:
    import sys as _sys, os as _os, importlib.util as _ilu
    _project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _project_root not in _sys.path:
        _sys.path.insert(0, _project_root)
    try:
        from src.modelos import create_hybrid_model
        from src.preprocesamiento import preprocess_image, load_images
        from src.extraccion_caracteristicas import load_metadata, get_features_for_image, FEATURE_COLUMNS
        from src.carga_datos_unificada import UnifiedDataLoader, create_stratified_folds
        from src.aumento_datos import DataAugmentor
    except ModuleNotFoundError:
        _pkg_dir = _os.path.join(_project_root, 'src')
        def _load_module(_name, _path):
            _spec = _ilu.spec_from_file_location(_name, _path)
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            return _mod
        _modelos = _load_module('modelos', _os.path.join(_pkg_dir, 'modelos.py'))
        _pre = _load_module('preprocesamiento', _os.path.join(_pkg_dir, 'preprocesamiento.py'))
        _ext = _load_module('extraccion_caracteristicas', _os.path.join(_pkg_dir, 'extraccion_caracteristicas.py'))
        _carga = _load_module('carga_datos_unificada', _os.path.join(_pkg_dir, 'carga_datos_unificada.py'))
        _aug = _load_module('aumento_datos', _os.path.join(_pkg_dir, 'aumento_datos.py'))
        create_hybrid_model = _modelos.create_hybrid_model
        preprocess_image = _pre.preprocess_image
        load_images = _pre.load_images
        load_metadata = _ext.load_metadata
        get_features_for_image = _ext.get_features_for_image
        FEATURE_COLUMNS = _ext.FEATURE_COLUMNS
        UnifiedDataLoader = _carga.UnifiedDataLoader
        create_stratified_folds = _carga.create_stratified_folds
        DataAugmentor = _aug.DataAugmentor

def get_memory_usage():
    """Retorna el uso de RAM en MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_unlabeled_features(unlabeled_image_paths, target_size=(300, 300)):
    """
    Carga características para imágenes no etiquetadas (usando valores promedio como placeholder).
    
    Args:
        unlabeled_image_paths (list): Lista de rutas de imágenes no etiquetadas.
        target_size (tuple): Tamaño objetivo para las imágenes.
        
    Returns:
        tuple: (imagenes_preprocesadas, caracteristicas_dummy)
    """
    # Cargar imágenes
    images = []
    for img_path in unlabeled_image_paths:
        try:
            image = preprocess_image(img_path, target_size)
            images.append(image)
        except Exception as e:
            print(f"Advertencia: Error al cargar imagen {img_path}: {e}")
            # Crear una imagen de ruido con la forma correcta
            dummy_image = np.random.rand(1, target_size[0], target_size[1], 3).astype(np.float32)
            images.append(dummy_image)
    
    if images:
        X_images = np.vstack(images)
    else:
        X_images = np.array([]).reshape(0, target_size[0], target_size[1], 3)
    
    # Crear características dummy (valores promedio)
    # En una implementación real, aquí se podrían extraer características reales
    num_samples = len(unlabeled_image_paths)
    if num_samples > 0:
        # Usar valores promedio de las características como placeholder
        dummy_features = np.full((num_samples, len(FEATURE_COLUMNS)), 0.5, dtype=np.float32)
    else:
        dummy_features = np.array([]).reshape(0, len(FEATURE_COLUMNS))
    
    return X_images, dummy_features

def train_enhanced(config):
    """Función principal para el entrenamiento mejorado del modelo con múltiples datasets."""
    print(f"Cargando configuración desde: {config}")
    with open(config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"\n{'='*60}")
    print(f"ENTRENAMIENTO MEJORADO CON DATASETS MÚLTIPLES")
    print(f"{'='*60}")
    print(f"RAM inicial: {get_memory_usage():.2f} MB\n")
    
    # 1. Configurar y cargar datasets múltiples
    print("Configurando cargador de datos unificado...")
    # `cfg` normalmente contiene una clave `datasets` con un mapeo por nombre (recomendado).
    # Sin embargo, algunos configs temporales (o notebooks) podrían usar una estructura distinta.
    # Intentamos resolver ambas posibilidades de forma tolerante y dar un mensaje claro si falta.
    if 'datasets' in cfg:
        datasets_config = cfg['datasets']
    else:
        # Detectar si `dataset` contiene en realidad un mapeo por dataset (ej. BALD -> {csv_path:...})
        ds_candidate = cfg.get('dataset', None)
        datasets_config = {}
        if isinstance(ds_candidate, dict):
            first_val = next(iter(ds_candidate.values()), None)
            if isinstance(first_val, dict) and any(k in first_val for k in ('csv_path', 'has_labels', 'image_dir')):
                datasets_config = ds_candidate

    if not datasets_config:
        raise KeyError(
            "No per-dataset configuration found in config. Ensure your YAML contains a `datasets:` mapping "
            "pointing to each dataset (e.g., BALD, AMD) with `csv_path` and `image_dir`. "
            "If running in Colab, make sure you unpacked the project ZIP into `/content/proyecto`."
        )

    data_loader = UnifiedDataLoader(datasets_config)
    
    print("Cargando todos los datasets...")
    combined_df = data_loader.load_all_datasets()
    print(f"Total de muestras cargadas: {len(combined_df)}")
    print(f"  - Etiquetadas: {len(combined_df[combined_df['has_labels']==True])}")
    print(f"  - No etiquetadas: {len(combined_df[combined_df['has_labels']==False])}")
    
    # 2. Preparar datos etiquetados
    print("\nPreparando datos etiquetados...")
    X_labeled_features, y_labeled, labeled_image_paths = data_loader.prepare_labeled_data(combined_df)
    print(f"Datos etiquetados preparados: {len(y_labeled) if y_labeled is not None else 0} muestras")
    
    # Verificar si hay datos etiquetados
    if X_labeled_features is None or len(X_labeled_features) == 0:
        print("ADVERTENCIA: No se encontraron datos etiquetados. Asegúrate de que al menos un dataset tenga etiquetas.")
        return
    
    print("Cargando imágenes etiquetadas por fold (memoria optimizada)...")
    
    # 3. Preparar datos no etiquetados
    unlabeled_image_paths = data_loader.prepare_unlabeled_data(combined_df)
    X_unlabeled_images = None
    X_unlabeled_features = None
    
    if unlabeled_image_paths:
        print(f"\nPreparando datos no etiquetados: {len(unlabeled_image_paths)} muestras")
        X_unlabeled_images, X_unlabeled_features = load_unlabeled_features(
            unlabeled_image_paths, 
            (cfg['dataset']['image_size'], cfg['dataset']['image_size'])
        )
        print(f"Datos no etiquetados preparados:")
        print(f"  - Imágenes: {X_unlabeled_images.shape}")
        print(f"  - Características: {X_unlabeled_features.shape}")
    else:
        print("\nNo hay datos no etiquetados disponibles.")
    
    # Verificar que tengamos al menos algunos datos para entrenar
    if (X_labeled_features is None or len(X_labeled_features) == 0) and \
       (X_unlabeled_images is None or len(X_unlabeled_images) == 0):
        print("ERROR: No hay datos disponibles para entrenamiento (ni etiquetados ni no etiquetados)")
        return
    
    # 4. Validación Cruzada
    n_splits = cfg['dataset']['n_splits']
    print(f"\nCreando {n_splits} folds para validación cruzada...")
    
    # Crear folds solo con datos etiquetados
    labeled_df = combined_df[combined_df['has_labels'] == True].copy()
    folds = create_stratified_folds(labeled_df, n_splits=n_splits, random_state=42)
    
    # Placeholder para almacenar historiales
    histories = []
    all_val_probs = []
    all_val_labels = []
    
    # 5. Entrenamiento por folds
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{n_splits}")
        print(f"{'='*50}")
        
        # Separar datos de entrenamiento y validación
        train_paths = [labeled_image_paths[i] for i in train_idx]
        val_paths = [labeled_image_paths[i] for i in val_idx]
        train_images = []
        val_images = []
        for p in train_paths:
            try:
                img = preprocess_image(p, (cfg['dataset']['image_size'], cfg['dataset']['image_size']))
                train_images.append(img)
            except Exception as e:
                print(f"Advertencia: Error al cargar imagen {p}: {e}")
                di = np.random.rand(1, cfg['dataset']['image_size'], cfg['dataset']['image_size'], 3).astype(np.float32)
                train_images.append(di)
        for p in val_paths:
            try:
                img = preprocess_image(p, (cfg['dataset']['image_size'], cfg['dataset']['image_size']))
                val_images.append(img)
            except Exception as e:
                print(f"Advertencia: Error al cargar imagen {p}: {e}")
                di = np.random.rand(1, cfg['dataset']['image_size'], cfg['dataset']['image_size'], 3).astype(np.float32)
                val_images.append(di)
        X_train_img = np.vstack(train_images) if train_images else np.empty((0, cfg['dataset']['image_size'], cfg['dataset']['image_size'], 3), dtype=np.float32)
        X_val_img = np.vstack(val_images) if val_images else np.empty((0, cfg['dataset']['image_size'], cfg['dataset']['image_size'], 3), dtype=np.float32)
        X_train_feat = X_labeled_features[train_idx]
        X_val_feat = X_labeled_features[val_idx]
        y_train = y_labeled[train_idx]
        y_val = y_labeled[val_idx]
        
        print(f"  Train: {len(y_train)} muestras")
        print(f"  Val: {len(y_val)} muestras")
        
        # 6. Crear aumentador de datos
        augmentor = DataAugmentor(cfg.get('augmentation', {}))
        
        # 7. Aplicar aumento de datos al conjunto de entrenamiento
        print("Aplicando aumento de datos... (desactivado en memoria para ahorrar RAM)")
        # Para evitar OOM en entornos con RAM limitada, desactivar generación completa
        # en memoria. num_augmentations=0 -> no se duplica el dataset en RAM.
        num_augmentations = 0
        X_train_img_aug = augmentor.augment_batch(X_train_img, num_augmentations=num_augmentations)

        scaler = StandardScaler()
        X_train_feat_sc = scaler.fit_transform(X_train_feat)
        X_val_feat_sc = scaler.transform(X_val_feat)

        # Replicar las características y etiquetas según el número de aumentaciones por muestra
        if num_augmentations > 0:
            reps = 1 + int(num_augmentations)
            X_train_feat_aug = np.repeat(X_train_feat_sc, reps, axis=0)
            y_train_aug = np.repeat(y_train, reps, axis=0)
        else:
            X_train_feat_aug = X_train_feat_sc
            y_train_aug = y_train
        
        print(f"  Datos originales: {len(y_train)} muestras")
        print(f"  Datos aumentados: {len(y_train_aug)} muestras")
        
        # 8. Crear y compilar el modelo
        print("Creando el modelo híbrido...")
        model = create_hybrid_model(
            image_shape=(cfg['dataset']['image_size'], cfg['dataset']['image_size'], 3),
            num_features=len(FEATURE_COLUMNS)
        )
        
        # Seleccionar método de entrenamiento (siempre supervisado ahora)
        print("Usando entrenamiento supervisado estándar...")
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Callbacks
        results_dir = cfg['output']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        checkpoint = ModelCheckpoint(f'{results_dir}best_model_fold_{fold+1}.keras', save_best_only=True)
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        
        # Entrenar con datos aumentados
        print("\nIniciando entrenamiento con datos aumentados...")
        cw = compute_class_weight('balanced', classes=np.array([0,1]), y=y_train_aug.astype(int))
        class_weight = {0: float(cw[0]), 1: float(cw[1])}
        history = model.fit(
            [X_train_img_aug, X_train_feat_aug],
            y_train_aug,
            batch_size=cfg['training']['batch_size'],
            epochs=cfg['training']['epochs'],
            validation_data=([X_val_img, X_val_feat_sc], y_val),
            callbacks=[checkpoint, early_stopping],
            class_weight=class_weight,
            verbose=1
        )
        histories.append(history)
        
        # Evaluación en conjunto de validación
        print("\nEvaluando en conjunto de validación...")
        val_probs = model.predict([X_val_img, X_val_feat_sc]).reshape(-1)
        all_val_probs.extend(val_probs.tolist())
        all_val_labels.extend(y_val.astype(int).tolist())
        
        default_thresh = 0.5
        val_preds = (val_probs >= default_thresh).astype(int)
        
        try:
            tn, fp, fn, tp = confusion_matrix(y_val, val_preds, labels=[0,1]).ravel()
            print(f"  Fold {fold+1}: threshold=0.5 -> TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        except Exception as e:
            print(f"  Error calculando matriz de confusión: {e}")

        try:
            p, r, t = precision_recall_curve(y_val.astype(int), val_probs)
            f1s = 2*p[1:]*r[1:]/(p[1:]+r[1:]+1e-8)
            if len(f1s) > 0:
                bi = int(np.argmax(f1s))
                bt = float(t[bi])
                val_preds_opt = (val_probs >= bt).astype(int)
                tn2, fp2, fn2, tp2 = confusion_matrix(y_val, val_preds_opt, labels=[0,1]).ravel()
                print(f"  Fold {fold+1}: threshold={bt:.3f} -> TP={tp2}, FP={fp2}, TN={tn2}, FN={fn2}")
            print(f"  Fold {fold+1}: prob_stats min={float(np.min(val_probs)):.4f}, max={float(np.max(val_probs)):.4f}, mean={float(np.mean(val_probs)):.4f}")
        except Exception as e:
            print(f"  Error analizando probabilidades: {e}")
    
    # --- Umbral global recomendado basado en F1 ---
    try:
        labels_arr = np.array(all_val_labels)
        probs_arr = np.array(all_val_probs)
        precisions, recalls, thresholds = precision_recall_curve(labels_arr, probs_arr)
        f1s = 2*precisions[1:]*recalls[1:]/(precisions[1:]+recalls[1:]+1e-8)
        if len(f1s) > 0:
            best_idx = int(np.argmax(f1s))
            best_threshold = float(thresholds[best_idx])
            print(f"\nUmbral recomendado (max F1): {best_threshold:.3f}")
        else:
            print("\nNo se pudo calcular umbral óptimo (datos insuficientes).")
    except Exception as e:
        print(f"\nError calculando umbral recomendado: {e}")

def main(config_path=None):
    """Entry point compatible con CLI y llamadas directas desde notebooks.

    - Si `config_path` es proporcionado (string), se usa directamente.
    - Si `config_path` es None, se parsean argumentos de línea de comandos (CLI).
    """
    if config_path is not None:
        # Llamada programática desde notebook u otro módulo
        train_enhanced(config_path)
        return

    # Compatibilidad con ejecución desde línea de comandos
    parser = argparse.ArgumentParser(description='Entrenar modelo de melanoma mejorado.')
    parser.add_argument('--config', type=str, required=True, help='Ruta al archivo de configuración YAML.')
    args = parser.parse_args()
    train_enhanced(args.config)

if __name__ == '__main__':
    main()
