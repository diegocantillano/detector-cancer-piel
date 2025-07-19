import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Configurar la página
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
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
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
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class SkinCancerDetector:
    def __init__(self, model_path='models/skin_cancer_model.h5'):
        self.model = None
        self.model_path = model_path
        self.img_size = (224, 224)
        self.load_model()

    @st.cache_resource
    def load_model(_self):
        """Cargar el modelo entrenado"""
        try:
            _self.model = load_model(_self.model_path)
            return True
        except Exception as e:
            st.error(f"Error al cargar el modelo: {str(e)}")
            return False

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
        try:
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None, None, None
            prediction = self.model.predict(processed_image)
            confidence = float(prediction[0][0])
            if confidence > 0.5:
                return "Maligno", confidence, "Alto"
            else:
                return "Benigno", confidence, "Bajo"
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")
            return None, None, None


def main():
    st.markdown('<h1 class="main-header">🔬 Detector de Cáncer de Piel</h1>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Sube una imagen de una lesión de piel para predecir si es benigna o maligna.</div>', unsafe_allow_html=True)

    detector = SkinCancerDetector()

    uploaded_file = st.file_uploader("📤 Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        if st.button("🔍 Analizar imagen"):
            result, confidence, risk_level = detector.predict(image)

            if result:
                if result == "Maligno":
                    st.markdown(f'<div class="result-box malignant-result"><strong>Resultado:</strong> {result}<br><strong>Confianza:</strong> {confidence:.2%}<br><strong>Nivel de riesgo:</strong> {risk_level}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-box benign-result"><strong>Resultado:</strong> {result}<br><strong>Confianza:</strong> {confidence:.2%}<br><strong>Nivel de riesgo:</strong> {risk_level}</div>', unsafe_allow_html=True)
            else:
                st.error("No se pudo realizar la predicción.")

if __name__ == "__main__":
    main()


