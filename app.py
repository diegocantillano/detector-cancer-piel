import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Configuración de la página
st.set_page_config(
    page_title="Detector de Cáncer de Piel",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .benign-result {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .malignant-result {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Clase detector
class SkinCancerDetector:
    def __init__(self, model_path='models/skin_cancer_model.h5'):
        self.model = None
        self.model_path = model_path
        self.img_size = (224, 224)
        self.load_model()

    @st.cache_resource
    def load_model(self):
        """Cargar el modelo entrenado"""
        try:
            self.model = load_model(self.model_path)
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")

    def preprocess_image(self, image):
        """Preprocesar la imagen para el modelo"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize(self.img_size)
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            st.error(f"Error al procesar la imagen: {str(e)}")
            return None

    def predict(self, image):
        """Hacer predicción sobre la imagen"""
        if self.model is None:
            return None, None, None

        processed_image = self.preprocess_image(image)
        if processed_image is None:
            return None, None, None

        try:
            prediction = self.model.predict(processed_image)
            confidence = float(prediction[0][0])
            result = "Maligno" if confidence > 0.5 else "Benigno"
            risk_level = "Alto" if confidence > 0.5 else "Bajo"
            return result, confidence, risk_level
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")
            return None, None, None

# Función principal
def main():
    st.markdown('<h1 class="main-header">🧪 Detector de Cáncer de Piel</h1>', unsafe_allow_html=True)
    st.markdown("Sube una imagen de una lesión en la piel para evaluar el riesgo de que sea benigna o maligna.")

    uploaded_file = st.file_uploader("📤 Sube tu imagen aquí", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="🖼️ Imagen cargada", use_column_width=True)

            detector = SkinCancerDetector()

            if st.button("🔍 Analizar"):
                result, confidence, risk_level = detector.predict(image)

                if result:
                    st.markdown(
                        f"""<div class='result-box {"malignant-result" if result == "Maligno" else "benign-result"}'>
                            <h3>Resultado: {result}</h3>
                            <p>Confianza del modelo: {confidence*100:.2f}%</p>
                            <p>Nivel de riesgo: {risk_level}</p>
                        </div>""",
                        unsafe_allow_html=True
                    )
                else:
                    st.error("No se pudo obtener un resultado. Verifica la imagen o intenta nuevamente.")
        except Exception as e:
            st.error(f"❌ Error al procesar la imagen: {str(e)}")

# Ejecutar app
if __name__ == "__main__":
    main()
