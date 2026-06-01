from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Concatenate, Dropout
from tensorflow.keras.models import Model

def create_hybrid_model(image_shape=(300, 300, 3), num_features=25):
    """
    Crea un modelo híbrido que combina una CNN (EfficientNet-B3) para imágenes
    y una MLP para características clínicas.
    """
    # --- Rama CNN para la imagen ---
    base_cnn = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=image_shape
    )
    base_cnn.trainable = False  # Congelar pesos preentrenados

    cnn_input = Input(shape=image_shape, name='image_input')
    cnn_output = base_cnn(cnn_input)
    pooled_cnn_output = GlobalAveragePooling2D()(cnn_output)

    # --- Rama MLP para las características ---
    feature_input = Input(shape=(num_features,), name='feature_input')
    mlp_output = Dense(512, activation='relu')(feature_input)
    mlp_output = Dropout(0.5)(mlp_output)
    mlp_output = Dense(128, activation='relu')(mlp_output)
    mlp_output = Dropout(0.5)(mlp_output)

    # --- Combinar y clasificar ---
    concatenated = Concatenate()([pooled_cnn_output, mlp_output])
    combined_output = Dense(512, activation='relu')(concatenated)
    combined_output = Dropout(0.5)(combined_output)
    combined_output = Dense(128, activation='relu')(combined_output)
    final_output = Dense(1, activation='sigmoid', name='final_output')(combined_output)

    # Crear el modelo final con dos entradas
    model = Model(inputs=[cnn_input, feature_input], outputs=final_output)

    return model

if __name__ == '__main__':
    print("Creando el modelo híbrido con EfficientNet-B3...")
    hybrid_model = create_hybrid_model()
    hybrid_model.summary()
    print("\nModelo creado exitosamente.")