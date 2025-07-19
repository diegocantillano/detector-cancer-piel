import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(
    page_title="Detector de Cáncer de Piel",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
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
    def __init__(self, model_path='skin_cancer_model.h5'):
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
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Redimensionar
            image = image.resize(self.img_size)

            # Convertir a array numpy
            img_array = np.array(image)

            # Normalizar
            img_array = img_array / 255.0

            # Expandir dimensiones para el batch
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
            # Preprocesar imagen
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None, None, None

            # Hacer predicción
            prediction = self.model.predict(processed_image)
            confidence = float(prediction[0][0])

            # Interpretar resultado
            if confidence > 0.5:
                result = "Maligno"
                risk_level = "Alto"
            else:
                result = "Benigno"
                risk_level = "Bajo"

            return result, confidence, risk_level
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")
            return None, None, None


def main():
    # Título principal
    st.markdown('<h1 class=
