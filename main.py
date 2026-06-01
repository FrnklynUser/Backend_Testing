from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import numpy as np
import cv2
import base64
import math
from io import BytesIO
import os
import tensorflow as tf
import time
import sys

# Directorio raíz del backend (donde está este archivo)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Importar funciones del proyecto
from tensorflow.keras.models import load_model
from src.preprocesamiento import preprocess_image, preprocess_image_from_array
from src.extraccion_caracteristicas import load_metadata, get_features_for_image, FEATURE_COLUMNS
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.explicabilidad import generate_grad_cam, overlay_heatmap
from src.extraccion_auto import extract_features_from_image, validate_dermatoscopic_image

# Importar módulos locales del backend
import auth
import history
from history import _sanitize

app = FastAPI(
    title="API Avanzada de Detección de Melanoma Acral",
    description="Backend unificado para autenticación, historial y predicción.",
    version="1.0.0"
)

# Configurar CORS para React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar el dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos de Datos ---
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    name: str

# --- Carga del Modelo ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model_final.keras')
# Buscar metadata en subdirectorio data/ del propio backend
METADATA_BALD = os.path.join(ROOT_DIR, 'data', 'BALD', 'metadata_completo.csv')
METADATA_AMD  = os.path.join(ROOT_DIR, 'data', 'AMD',  'metadata_amd.csv')

model = None
metadata_df = None
scaler = StandardScaler()

# --- Cargar modelo ---
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("✅ Modelo cargado correctamente.")
    else:
        print(f"⚠️  Modelo no encontrado en: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")

# --- Cargar metadata y ajustar scaler (opcional — fallback si no existen los CSVs) ---
try:
    dfs = []
    if os.path.exists(METADATA_BALD):
        dfs.append(load_metadata(METADATA_BALD))
        print("✅ Metadata BALD cargada.")
    else:
        print(f"⚠️  Metadata BALD no encontrada: {METADATA_BALD}")

    if os.path.exists(METADATA_AMD):
        dfs.append(load_metadata(METADATA_AMD))
        print("✅ Metadata AMD cargada.")
    else:
        print(f"⚠️  Metadata AMD no encontrada: {METADATA_AMD}")

    if dfs:
        metadata_df = pd.concat(dfs, ignore_index=True)
        features_data = metadata_df[FEATURE_COLUMNS].values
        scaler.fit(features_data)
        print("✅ Metadatos y Scaler configurados desde CSV.")
    else:
        # Sin CSVs: ajustar scaler con valores sintéticos para que no falle
        print("⚠️  Sin metadata CSV — usando scaler con valores por defecto (imágenes externas funcionarán, dataset interno no).")
        dummy = np.zeros((2, len(FEATURE_COLUMNS)), dtype=np.float32)
        dummy[1] = 1.0
        scaler.fit(dummy)
except Exception as e:
    print(f"❌ Error al cargar metadata: {e}")
    # Fallback de emergencia para el scaler
    dummy = np.zeros((2, len(FEATURE_COLUMNS)), dtype=np.float32)
    dummy[1] = 1.0
    scaler.fit(dummy)

# --- Endpoints de Autenticación ---

@app.post("/auth/login")
async def login(req: LoginRequest):
    user = auth.authenticate(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")
    
    # En un sistema real usaríamos JWT, aquí retornamos el perfil básico
    return {
        "user": {
            "username": user['username'],
            "name": user['name'],
            "role": user['role']
        },
        "message": "Login exitoso"
    }

@app.post("/auth/register")
async def register(req: RegisterRequest):
    success, message = auth.register_user(req.username, req.password, req.name)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}

# --- Endpoints de Historial ---

@app.get("/history/{username}")
async def get_user_history(username: str):
    user_history = history.load_history(username)
    # Sanear datos corruptos (NaN/Inf) de entradas antiguas en el archivo JSON
    clean_history = _sanitize(user_history)
    return {"analyses": clean_history}

@app.delete("/history/{username}/{analysis_id}")
async def delete_history_item(username: str, analysis_id: str):
    success = history.delete_analysis(username, analysis_id)
    if not success:
        raise HTTPException(status_code=500, detail="Error al eliminar el registro")
    return {"message": "Análisis eliminado correctamente"}

# --- Endpoint de Predicción ---

@app.post("/predict")
async def predict(
    username: str,
    file: UploadFile = File(...),
    age: Optional[float] = Form(None),
    gender: Optional[float] = Form(None),
    family_history: Optional[float] = Form(None),
    sun_exposure: Optional[float] = Form(None)
):
    start_time = time.time()
    
    try:
        # Validaciones de imagen
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo no es una imagen.")
        
        contents = await file.read()
        image_array = np.frombuffer(contents, dtype=np.uint8)
        original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if original_image is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")
        
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        if model is None or metadata_df is None:
             raise HTTPException(status_code=500, detail="Modelo no cargado en el servidor")

        # ——— VALIDACIÓN DERMATOSCÓPICA ———
        validation = validate_dermatoscopic_image(original_image)
        print(f"DEBUG: Validación → score={validation['score']}, válida={validation['is_valid']}")
        if not validation["is_valid"]:
            detail = {
                "error": "imagen_no_dermatoscopica",
                "message": validation["reason"],
                "validation_score": validation["score"],
                "criteria": validation["details"],
                "suggestion": (
                    "Por favor, cargue una imagen dermatoscópica de una lesión "
                    "de piel capturada con dermatoscopio o cámara médica."
                ),
            }
            raise HTTPException(
                status_code=422,
                detail=_sanitize(detail)
            )


        # 1. Preprocesar
        processed_image = preprocess_image_from_array(original_image, target_size=(300, 300))

        # 2. Extraer características
        image_filename = file.filename
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            if image_filename.endswith(ext):
                image_filename = image_filename[:-len(ext)]
                break
        
        try:
            features_vector = get_features_for_image(metadata_df, image_filename)
            print(f"✅ Features del CSV para: {image_filename}")
        except:
            # Imagen externa → calcular features automáticamente desde la imagen
            print(f"⚠️  Imagen no en dataset, calculando features automáticamente...")
            features_vector = extract_features_from_image(original_image)

        # Sobrescribir features clínicas con datos reales si el usuario los proporcionó
        # Índices: age=11, gender=12, family_history=13, sun_exposure=14
        if age is not None:
            features_vector[0][11] = float(age)
        if gender is not None:
            features_vector[0][12] = float(gender)
        if family_history is not None:
            features_vector[0][13] = float(family_history)
        if sun_exposure is not None:
            features_vector[0][14] = float(sun_exposure)

        # 3. Escalar y Predecir
        features_vector_scaled = scaler.transform(features_vector)
        pred = model.predict([processed_image, features_vector_scaled], verbose=0)
        pred_value = float(pred[0][0])

        # 4. Grad-CAM
        try:
            target_layer = 'efficientnetb3'
            grad_cam_heatmap = generate_grad_cam(model, [processed_image, features_vector_scaled], 0, target_layer)
        except:
            grad_cam_heatmap = np.ones((300, 300)) * 0.5

        # 5. Superponer y Codificar
        superimposed_image = overlay_heatmap(original_image_rgb, grad_cam_heatmap)
        _, buffer_gradcam = cv2.imencode('.jpg', cv2.cvtColor(superimposed_image, cv2.COLOR_RGB2BGR))
        grad_cam_base64 = base64.b64encode(buffer_gradcam).decode('utf-8')

        prediction = 'Melanoma' if pred_value >= 0.5 else 'Nevus'
        confidence = pred_value if prediction == 'Melanoma' else 1 - pred_value

        # Preparar características para el historial (sanitizando NaN/Inf)
        features_dict = {}
        for i, feature_name in enumerate(FEATURE_COLUMNS):
            raw = float(features_vector[0][i])
            features_dict[feature_name] = None if (math.isnan(raw) or math.isinf(raw)) else raw

        # ——— ALERTA DE INCONSISTENCIA CLÍNICA ———
        # Umbrales calibrados para alta especificidad:
        # se requieren 3+ criterios para evitar falsas alarmas en imágenes externas.
        clinical_alert = None
        if prediction == 'Nevus' and features_dict:
            high_border       = (features_dict.get('border_irregularity') or 0) >= 0.82   # Borde muy irregular
            large_diameter    = (features_dict.get('diameter') or 0)             >= 11.0   # > 11mm (claramente grande)
            high_asymmetry    = (features_dict.get('asymmetry_score') or 0)      >= 0.45   # Asimetría marcada
            high_eccentricity = (features_dict.get('eccentricity') or 0)         >= 0.85   # Muy elíptica
            high_color_var    = (features_dict.get('color_variation') or 0)      >= 0.55   # Alta variación de color

            suspicious_flags = sum([high_border, large_diameter, high_asymmetry,
                                    high_eccentricity, high_color_var])
            if suspicious_flags >= 3:   # Mínimo 3/5 criterios
                risk_level = "ALTO" if suspicious_flags >= 4 else "MODERADO"
                triggered = []
                if high_border:       triggered.append(f"borde muy irregular ({features_dict.get('border_irregularity', 0):.2f} ≥ 0.82)")
                if large_diameter:    triggered.append(f"diámetro notable ({features_dict.get('diameter', 0):.1f}mm ≥ 11mm)")
                if high_asymmetry:    triggered.append(f"asimetría marcada ({features_dict.get('asymmetry_score', 0):.2f} ≥ 0.45)")
                if high_eccentricity: triggered.append(f"excentricidad alta ({features_dict.get('eccentricity', 0):.2f} ≥ 0.85)")
                if high_color_var:    triggered.append(f"variación de color alta ({features_dict.get('color_variation', 0):.2f} ≥ 0.55)")

                clinical_alert = {
                    "level": risk_level,
                    "suspicious_flags": suspicious_flags,
                    "message": (
                        f"El modelo CNN clasifica como Nevus, pero {suspicious_flags}/5 "
                        f"indicadores morfológicos ABCDE presentan valores de riesgo {risk_level}."
                    ),
                    "triggered_criteria": triggered,
                    "recommendation": (
                        "Se recomienda evaluación presencial por un dermatólogo. "
                        "Para mayor precisión, utilice imágenes dermatoscópicas del dataset de entrenamiento."
                    ),
                }
                print(f"⚠️  ALERTA CLÍNICA [{risk_level}]: {suspicious_flags} criterios ABCDE → {triggered}")

        # 6. Guardar en Historial
        history.save_analysis(
            username=username,
            image_name=file.filename,
            prediction=prediction,
            confidence=confidence,
            features=features_dict
        )

        return {
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "grad_cam_image": f"data:image/jpeg;base64,{grad_cam_base64}",
            "top_features": features_dict,
            "clinical_alert": clinical_alert,
            "metrics": {
                "inference_time_ms": round((time.time() - start_time) * 1000, 2),
                "image_size_kb": float(len(contents) / 1024.0),
                "size_bytes": int(len(contents)),
                "confidence_percent": round(confidence * 100, 1)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "API PDA Activa", "status": "online"}
