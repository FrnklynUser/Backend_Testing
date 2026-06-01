"""
extraccion_auto.py
------------------
Extrae automáticamente las 25 características clínicas del FEATURE_COLUMNS
directamente desde el array de la imagen (BGR de OpenCV).
Se usa como fallback cuando la imagen no está en el CSV de metadatos,
permitiendo analizar imágenes descargadas de internet.

También incluye validate_dermatoscopic_image() para verificar que la imagen
sea una imagen dermatoscópica válida antes de ejecutar la inferencia.
"""

import cv2
import numpy as np


def _segmentar_lesion(img_rgb: np.ndarray):
    """
    Segmenta la región de la lesión usando umbralización adaptativa en el
    canal de saturación (HSV). Devuelve la máscara binaria.
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = img_hsv[:, :, 1]
    # Umbral de Otsu sobre la saturación
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # Quedarse sólo con el contorno más grande
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest], -1, 255, -1)
        return clean_mask, largest
    return mask, None


def _asymmetry_score(mask: np.ndarray) -> float:
    """Calcula asimetría comparando mitades superior/inferior e izquierda/derecha."""
    h, w = mask.shape
    top = mask[:h // 2, :]
    bot = np.flipud(mask[h // 2:, :])
    left = mask[:, :w // 2]
    right = np.fliplr(mask[:, w // 2:])
    rows = min(top.shape[0], bot.shape[0])
    cols = min(left.shape[1], right.shape[1])
    diff_v = np.sum(top[:rows] != bot[:rows]) / (h * w + 1e-8)
    diff_h = np.sum(left[:, :cols] != right[:, :cols]) / (h * w + 1e-8)
    return float(np.clip((diff_v + diff_h) / 2, 0, 1))


def _border_irregularity(contour) -> float:
    """Índice de compacidad del contorno — cuanto más irregular, más alto."""
    if contour is None:
        return 0.5
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1e-6:
        return 0.5
    compactness = (4 * np.pi * area) / (perimeter ** 2)
    return float(np.clip(1 - compactness, 0, 1))


def _color_variation(img_rgb: np.ndarray, mask: np.ndarray) -> float:
    """Desviación estándar normalizada de color dentro de la lesión."""
    pixels = img_rgb[mask > 0].astype(np.float32)
    if len(pixels) == 0:
        return 0.5
    std = np.std(pixels, axis=0)
    return float(np.clip(np.mean(std) / 128.0, 0, 1))


def _diameter_estimate(contour, img_shape) -> float:
    """Estima el diámetro relativo de la lesión (normalizado ~3–12 mm)."""
    if contour is None:
        return 7.0
    _, radius = cv2.minEnclosingCircle(contour)
    h, w = img_shape[:2]
    max_dim = max(h, w)
    # Estimamos que la imagen completa ≈ 20 mm
    diameter_mm = (2 * radius / max_dim) * 20.0
    return float(np.clip(diameter_mm, 2.0, 15.0))


def _glcm_features(gray: np.ndarray, mask: np.ndarray):
    """
    Calcula contraste, energía, homogeneidad y correlación de forma aproximada
    usando estadísticas de vecindad sin scikit-image (solo numpy/cv2).
    """
    roi = gray.copy().astype(np.float32)
    roi[mask == 0] = np.nan
    # Gradiente local como proxy de contraste
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    contrast = float(np.clip(np.nanmean(mag) / 255.0, 0, 1))

    # Energía: uniformidad del histograma
    hist = cv2.calcHist([gray], [0], mask, [256], [0, 256])
    hist_norm = hist / (hist.sum() + 1e-8)
    energy = float(np.clip(float(np.sum(hist_norm ** 2)) * 10, 0, 1))

    # Homogeneidad: inverso del contraste
    homogeneity = float(np.clip(1 - contrast, 0, 1))

    # Correlación: basada en la correlación de Pearson del gradiente
    if gx[mask > 0].std() > 1e-6 and gy[mask > 0].std() > 1e-6:
        corr = float(np.corrcoef(gx[mask > 0].flatten(), gy[mask > 0].flatten())[0, 1])
        corr = float(np.clip((corr + 1) / 2, 0, 1))
    else:
        corr = 0.5

    return contrast, energy, homogeneity, corr


def _shape_features(contour, mask: np.ndarray):
    """Excentricidad, compacidad y area_ratio."""
    if contour is None or len(contour) < 5:
        return 0.5, 0.5, 0.5
    try:
        ellipse = cv2.fitEllipse(contour)
        a = ellipse[1][1] / 2  # semi-eje mayor
        b = ellipse[1][0] / 2  # semi-eje menor
        if a < 1e-6:
            eccentricity = 0.5
        else:
            eccentricity = float(np.clip(np.sqrt(1 - (b / a) ** 2), 0, 1))
    except Exception:
        eccentricity = 0.5

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1e-6:
        compactness = 0.5
    else:
        compactness = float(np.clip((4 * np.pi * area) / (perimeter ** 2), 0, 1))

    total_pixels = mask.shape[0] * mask.shape[1]
    area_ratio = float(np.clip(area / total_pixels, 0, 1))
    return eccentricity, compactness, area_ratio


def _texture_features(gray: np.ndarray, mask: np.ndarray):
    """Rugosidad de textura y suavidad de superficie."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    roi_lap = laplacian[mask > 0]
    roughness = float(np.clip(np.std(roi_lap) / 100.0, 0, 1))
    smoothness = float(np.clip(1 - roughness, 0, 1))
    return roughness, smoothness


def _color_uniformity(img_rgb: np.ndarray, mask: np.ndarray) -> float:
    """Uniformidad de color dentro de la lesión (inverso de la varianza)."""
    pixels = img_rgb[mask > 0].astype(np.float32)
    if len(pixels) == 0:
        return 0.5
    var = np.var(pixels)
    return float(np.clip(1 - var / (255 ** 2), 0, 1))


def _edge_sharpness(gray: np.ndarray, mask: np.ndarray) -> float:
    """Nitidez del borde de la lesión."""
    contour_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8)) - cv2.erode(mask, np.ones((5, 5), np.uint8))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge_vals = mag[contour_mask > 0]
    if len(edge_vals) == 0:
        return 0.5
    return float(np.clip(np.mean(edge_vals) / 255.0, 0, 1))


def _pattern_symmetry(mask: np.ndarray) -> float:
    """Simetría del patrón de la máscara (opuesto a asimetría)."""
    return float(np.clip(1 - _asymmetry_score(mask), 0, 1))


def _vascularity_proxy(img_rgb: np.ndarray, mask: np.ndarray) -> float:
    """
    Proxy de vascularidad: presencia de tonos rojizos oscuros
    en la lesión que pueden indicar estructuras vasculares.
    """
    pixels = img_rgb[mask > 0].astype(np.float32)
    if len(pixels) == 0:
        return 0.3
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    # Ratio R/(G+B+1) como indicador de rojez
    redness = r / (g + b + 1)
    return float(np.clip(np.mean(redness) - 0.3, 0, 1))


def _pigment_network_proxy(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Proxy de red de pigmento: contraste local de alta frecuencia
    que sugiere estructura reticular.
    """
    # Diferencia entre imagen original y versión suavizada
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    diff = cv2.absdiff(gray, blur)
    roi = diff[mask > 0]
    if len(roi) == 0:
        return 0.4
    return float(np.clip(np.mean(roi) / 80.0, 0, 1))


def _streaks_proxy(gray: np.ndarray, mask: np.ndarray) -> float:
    """
    Proxy de streaks: detección de estructuras lineales mediante
    filtros Gabor simplificados.
    """
    kernel_h = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
    kernel_v = kernel_h.T
    h = cv2.filter2D(gray.astype(np.float32), -1, kernel_h)
    v = cv2.filter2D(gray.astype(np.float32), -1, kernel_v)
    response = np.sqrt(h ** 2 + v ** 2)
    roi = response[mask > 0]
    if len(roi) == 0:
        return 0.3
    return float(np.clip(np.mean(roi) / 150.0, 0, 1))


def _regression_structures_proxy(img_rgb: np.ndarray, mask: np.ndarray) -> float:
    """
    Proxy de estructuras de regresión: presencia de áreas blancas/cicatriciales
    dentro de la lesión.
    """
    pixels = img_rgb[mask > 0].astype(np.float32)
    if len(pixels) == 0:
        return 0.3
    brightness = np.mean(pixels, axis=1)
    # Porcentaje de píxeles muy brillantes (>200) dentro de la lesión
    white_ratio = np.mean(brightness > 200)
    return float(np.clip(white_ratio, 0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# VALIDADOR DE IMAGEN DERMATOSCÓPICA
# ─────────────────────────────────────────────────────────────────────────────

def _score_area_ratio(mask: np.ndarray) -> tuple[float, str]:
    """
    Verifica que la región segmentada ocupe entre el 5% y el 70% de la imagen.
    Las lesiones dermatoscópicas típicamente ocupan entre el 15% y el 60%.
    Una imagen completamente oscura o sin región clara falla este criterio.
    """
    total_pixels = mask.shape[0] * mask.shape[1]
    lesion_pixels = np.sum(mask > 0)
    ratio = lesion_pixels / (total_pixels + 1e-8)

    if ratio < 0.05:
        return 0.0, f"región segmentada demasiado pequeña ({ratio*100:.1f}% del área)"
    if ratio > 0.75:
        return 0.0, f"región segmentada demasiado grande ({ratio*100:.1f}% del área) — posible imagen sin lesión central"

    # Score máximo en el rango óptimo [0.10, 0.60]
    if 0.10 <= ratio <= 0.60:
        return 1.0, "ok"
    # Degradar suavemente en los extremos [0.05,0.10) y (0.60,0.75]
    if ratio < 0.10:
        return float((ratio - 0.05) / 0.05), "ok"
    return float((0.75 - ratio) / 0.15), "ok"


def _score_lesion_compactness(contour) -> tuple[float, str]:
    """
    Las lesiones dermatoscópicas son regiones compactas (aproximadamente
    circulares u ovales). Una imagen aleatoria tiende a producir contornos
    muy irregulares o dispersos al segmentar.
    """
    if contour is None:
        return 0.0, "no se encontró un contorno definido en la imagen"
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 1e-6 or area < 100:
        return 0.0, "contorno demasiado pequeño para ser una lesión"
    # Índice de circularidad: 1.0 = círculo perfecto
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    if circularity < 0.10:
        return 0.0, f"contorno muy irregular (circularidad={circularity:.2f}) — probablemente no es una lesión"
    return float(np.clip(circularity, 0, 1)), "ok"


def _score_skin_tones(img_rgb: np.ndarray, mask: np.ndarray) -> tuple[float, str]:
    """
    Verifica la presencia de tonos de piel en el fondo (región fuera de la lesión).
    Las imágenes dermatoscópicas siempre tienen piel alrededor de la lesión.
    Usa el modelo YCrCb para detección de piel (robusto a variaciones de iluminación).
    """
    background_mask = (mask == 0).astype(np.uint8) * 255
    background_pixels = np.sum(background_mask > 0)

    if background_pixels < 1000:
        # Si no hay suficiente fondo, es sospechoso (no es una lesión centrada en piel)
        return 0.0, "no se detectó suficiente área de piel circundante"

    img_ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    skin_lower = np.array([0, 133, 77], dtype=np.uint8)
    skin_upper = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask_full = cv2.inRange(img_ycrcb, skin_lower, skin_upper)

    skin_in_background = cv2.bitwise_and(skin_mask_full, background_mask)
    skin_ratio = np.sum(skin_in_background > 0) / (background_pixels + 1e-8)

    if skin_ratio < 0.12:
        return 0.0, f"no se detectaron tonos de piel en el fondo ({skin_ratio*100:.1f}%)"
    return float(np.clip(skin_ratio * 1.5, 0, 1)), "ok"



def _score_dark_center(img_rgb: np.ndarray, mask: np.ndarray) -> tuple[float, str]:
    """
    En imágenes dermatoscópicas la lesión es típicamente más oscura que
    el fondo de piel circundante. Penaliza si el centro es más brillante
    que el borde, lo que indicaría una imagen no dermatoscópica.
    """
    background_mask = (mask == 0).astype(np.uint8)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    if np.sum(background_mask > 0) < 1000:
        return 0.0, "ausencia de fondo para contraste de brillo"

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Verificación de seguridad para evitar NaN
    mask_pixels = gray[mask > 0]
    bg_pixels = gray[background_mask > 0]
    
    if len(mask_pixels) == 0 or len(bg_pixels) == 0:
        return 0.0, "ausencia de píxeles para contraste de brillo"

    lesion_brightness = np.mean(mask_pixels)
    background_brightness = np.mean(bg_pixels)


    diff = background_brightness - lesion_brightness
    if diff < -15:
        return 0.0, f"la región central es más brillante que el fondo (Δ={diff:.0f})"
    return float(np.clip(diff / 60.0, 0, 1)), "ok"



def _score_color_profile(img_rgb: np.ndarray) -> tuple[float, str]:
    """
    Las imágenes dermatoscópicas tienen un perfil de color dominante
    en tonos cálidos (marrones, rosados, beige) + oscuros (negros, grises).
    Penaliza imágenes con colores muy saturados no relacionados con piel
    (azul cielo, verde césped, rojo intenso, etc.).
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = img_hsv[:, :, 0]           # 0-179
    saturation = img_hsv[:, :, 1]    # 0-255

    # Rangos de hue NO dermatoscópicos: azul (90-130), verde (40-80), amarillo puro (25-35)
    non_derm_mask = (
        ((hue >= 90) & (hue <= 130)) |    # Azul
        ((hue >= 45) & (hue <= 80))        # Verde
    )
    # Solo contar píxeles con saturación alta (colores vivos)
    vivid_mask = saturation > 80
    vivid_non_derm = np.sum(non_derm_mask & vivid_mask)
    total_vivid = np.sum(vivid_mask)
    
    if total_vivid < 100:
        # Si casi no hay colores vivos, el perfil es neutro/aceptable
        return 0.8, "ok"

    non_derm_ratio = vivid_non_derm / (total_vivid + 1e-8)

    if non_derm_ratio > 0.45:

        return 0.0, f"perfil de color no dermatoscópico ({non_derm_ratio*100:.0f}% colores no relacionados con piel)"
    return float(np.clip(1 - non_derm_ratio * 2, 0, 1)), "ok"


def validate_dermatoscopic_image(
    img_bgr: np.ndarray,
    threshold: float = 0.45
) -> dict:
    """
    Valida si una imagen es dermatoscópica antes de ejecutar la inferencia.

    Evalúa 5 criterios heurísticos y calcula un score global ponderado [0, 1]:
      - Ratio de área de la lesión segmentada  (peso: 0.25)
      - Compacidad del contorno                (peso: 0.25)
      - Tonos de piel en el fondo              (peso: 0.20)
      - Centro más oscuro que el fondo         (peso: 0.20)
      - Perfil de color general                (peso: 0.10)

    Args:
        img_bgr:   Imagen en formato BGR (salida directa de cv2.imdecode).
        threshold: Score mínimo para considerar la imagen válida (por defecto 0.38).

    Returns:
        dict con:
          - is_valid   (bool)  : True si el score >= threshold
          - score      (float) : Score global [0, 1]
          - reason     (str)   : Motivo de rechazo (si aplica)
          - details    (dict)  : Scores individuales por criterio
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Segmentar para obtener máscara y contorno
    mask, contour = _segmentar_lesion(img_rgb)

    # Evaluar cada criterio
    s_area,        r_area        = _score_area_ratio(mask)
    s_compact,     r_compact     = _score_lesion_compactness(contour)
    s_skin,        r_skin        = _score_skin_tones(img_rgb, mask)
    s_dark,        r_dark        = _score_dark_center(img_rgb, mask)
    s_color,       r_color       = _score_color_profile(img_rgb)

    # Pesos por criterio
    weights = {
        "area_ratio":          (s_area,    0.25),
        "lesion_compactness":  (s_compact, 0.25),
        "skin_tones":          (s_skin,    0.20),
        "dark_center":         (s_dark,    0.20),
        "color_profile":       (s_color,   0.10),
    }
    reasons = {
        "area_ratio":         r_area,
        "lesion_compactness": r_compact,
        "skin_tones":         r_skin,
        "dark_center":        r_dark,
        "color_profile":      r_color,
    }

    # Nombres de criterios en español
    criteria_names = {
        "area_ratio": "Ratio de área",
        "lesion_compactness": "Compacidad de la lesión",
        "skin_tones": "Tonos de piel",
        "dark_center": "Centro oscuro",
        "color_profile": "Perfil de color"
    }

    score = sum(s * w for s, w in weights.values())
    details = {criteria_names.get(k, k): round(v[0], 3) for k, v in weights.items()}

    # Identificar razones de fallo (criterios con score < 0.30)
    failed_reasons = [
        f"{criteria_names.get(k, k)}: {reasons[k]}"
        for k, (s, _) in weights.items()
        if s < 0.30 and reasons[k] != "ok"
    ]

    if score >= threshold:
        return {
            "is_valid": True,
            "score": round(score, 3),
            "reason": "",
            "details": details,
        }
    else:
        main_reason = failed_reasons[0] if failed_reasons else "score global insuficiente"
        return {
            "is_valid": False,
            "score": round(score, 3),
            "reason": (
                f"Imagen; Rechazada\n"
                f"La imagen no parece ser una imagen dermatoscópica de una lesión de piel "
                f"(score de validación: {score:.2f}/{threshold:.2f}). "
                f"Criterio principal: {main_reason}."
            ),
            "details": details,
        }


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACCIÓN DE CARACTERÍSTICAS
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extrae automáticamente las 25 características clínicas desde la imagen.

    Args:
        img_bgr: Array de imagen en formato BGR (como lo devuelve cv2.imdecode).

    Returns:
        np.ndarray de shape (1, 25) con las características normalizadas [0, 1]
        o en el rango esperado por FEATURE_COLUMNS:
        ['asymmetry_score', 'border_irregularity', 'color_variation', 'diameter',
         'contrast', 'energy', 'homogeneity', 'correlation',
         'eccentricity', 'compactness', 'area_ratio',
         'age', 'gender', 'family_history', 'sun_exposure',
         'texture_roughness', 'lesion_shape', 'color_uniformity',
         'edge_sharpness', 'surface_smoothness', 'pattern_symmetry',
         'vascularity', 'pigment_network', 'streaks', 'regression_structures']
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_rgb.shape[:2]

    # Segmentar la lesión
    mask, contour = _segmentar_lesion(img_rgb)

    # ---- Calcular cada feature ----
    asymmetry_score = _asymmetry_score(mask)
    border_irregularity = _border_irregularity(contour)
    color_variation = _color_variation(img_rgb, mask)
    diameter = _diameter_estimate(contour, img_rgb.shape)
    contrast, energy, homogeneity, correlation = _glcm_features(gray, mask)
    eccentricity, compactness, area_ratio = _shape_features(contour, mask)

    # Características clínico-demográficas: se usan valores medios del dataset
    # (calculados a partir del CSV de entrenamiento)
    age = 57.0          # Mediana del dataset
    gender = 0.5        # 0=masculino, 1=femenino → neutro
    family_history = 1  # Sin información → valor medio del dataset
    sun_exposure = 1    # Sin información → valor medio del dataset

    texture_roughness = _texture_features(gray, mask)[0]
    lesion_shape = float(np.clip(compactness, 0, 1))
    color_uniformity = _color_uniformity(img_rgb, mask)
    edge_sharpness = _edge_sharpness(gray, mask)
    surface_smoothness = _texture_features(gray, mask)[1]
    pattern_symmetry = _pattern_symmetry(mask)
    vascularity = _vascularity_proxy(img_rgb, mask)
    pigment_network = _pigment_network_proxy(gray, mask)
    streaks = _streaks_proxy(gray, mask)
    regression_structures = _regression_structures_proxy(img_rgb, mask)

    features = np.array([[
        asymmetry_score,
        border_irregularity,
        color_variation,
        diameter,
        contrast,
        energy,
        homogeneity,
        correlation,
        eccentricity,
        compactness,
        area_ratio,
        age,
        gender,
        family_history,
        sun_exposure,
        texture_roughness,
        lesion_shape,
        color_uniformity,
        edge_sharpness,
        surface_smoothness,
        pattern_symmetry,
        vascularity,
        pigment_network,
        streaks,
        regression_structures
    ]], dtype=np.float32)

    return features


if __name__ == '__main__':
    print("Módulo de extracción automática de características listo.")
    print(f"Produce un vector de 25 features por imagen.")
