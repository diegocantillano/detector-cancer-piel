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
    def __init__(self, model_path='models/skin_cancer_model.h5'):
        self.model_path = model_path
        self.img_size = (224, 224)
        self.model = load_skin_cancer_model(self.model_path)
    
    @st.cache_resource
    def load_skin_cancer_model(model_path='models/skin_cancer_model.h5'):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None
    
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
            return None, None
        
        try:
            # Preprocesar imagen
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None, None
            
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
    st.markdown('<h1 class="main-header">🔬 Detector de Cáncer de Piel</h1>', unsafe_allow_html=True)
    
    # Información sobre la aplicación
    st.markdown("""
    <div class="info-box">
        <h3>ℹ️ Información Importante</h3>
        <p>Esta aplicación utiliza inteligencia artificial para analizar imágenes de lunares y lesiones cutáneas, 
        ayudando en la detección temprana de melanomas. Sin embargo, <strong>NO reemplaza el diagnóstico médico profesional</strong>.</p>
        <p><strong>Siempre consulte con un dermatólogo para un diagnóstico definitivo.</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Inicializar el detector
    detector = SkinCancerDetector()
    
    # Sidebar con información
    with st.sidebar:
        st.markdown('<h2 class="sub-header">📋 Instrucciones</h2>', unsafe_allow_html=True)
        st.markdown("""
        1. **Suba una imagen** de la lesión cutánea
        2. **Asegúrese** de que la imagen sea clara y bien iluminada
        3. **Revise** el resultado de la predicción
        4. **Consulte** con un médico especialista
        """)
        
        st.markdown('<h2 class="sub-header">⚠️ Limitaciones</h2>', unsafe_allow_html=True)
        st.markdown("""
        - Solo para fines educativos
        - Precisión limitada
        - No reemplaza diagnóstico médico
        - Requiere imágenes de alta calidad
        """)
        
        st.markdown('<h2 class="sub-header">📊 Estadísticas del Modelo</h2>', unsafe_allow_html=True)
        st.markdown("""
        - **Datos de entrenamiento**: 6,000 imágenes
        - **Datos de prueba**: 1,000 imágenes
        - **Arquitectura**: CNN profunda
        - **Precisión estimada**: ~85-90%
        """)
    
    # Área principal de la aplicación
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Subir Imagen</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Seleccione una imagen de la lesión cutánea",
            type=['jpg', 'jpeg', 'png'],
            help="Formatos soportados: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Mostrar imagen original
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen cargada", use_column_width=True)
            
            # Información de la imagen
            st.write(f"**Tamaño original:** {image.size}")
            st.write(f"**Formato:** {image.format}")
            st.write(f"**Modo:** {image.mode}")
    
    with col2:
        st.markdown('<h2 class="sub-header">🔍 Resultado del Análisis</h2>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Botón para analizar
            if st.button("🔬 Analizar Imagen", type="primary"):
                with st.spinner("Analizando imagen..."):
                    # Hacer predicción
                    result, confidence, risk_level = detector.predict(image)
                    
                    if result is not None:
                        # Mostrar resultado
                        if result == "Maligno":
                            st.markdown(f"""
                            <div class="result-box malignant-result">
                                <h3>⚠️ Resultado: {result}</h3>
                                <p><strong>Nivel de riesgo:</strong> {risk_level}</p>
                                <p><strong>Confianza:</strong> {confidence:.2%}</p>
                                <p><strong>Recomendación:</strong> Consulte inmediatamente con un dermatólogo</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-box benign-result">
                                <h3>✅ Resultado: {result}</h3>
                                <p><strong>Nivel de riesgo:</strong> {risk_level}</p>
                                <p><strong>Confianza:</strong> {(1-confidence):.2%}</p>
                                <p><strong>Recomendación:</strong> Mantenga observación periódica</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Gráfico de confianza
                        st.markdown('<h3 class="sub-header">📊 Nivel de Confianza</h3>', unsafe_allow_html=True)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        if result == "Maligno":
                            probs = [1-confidence, confidence]
                            colors = ['#28a745', '#dc3545']
                        else:
                            probs = [1-confidence, confidence]
                            colors = ['#28a745', '#dc3545']
                        
                        bars = ax.bar(['Benigno', 'Maligno'], probs, color=colors, alpha=0.7)
                        ax.set_ylabel('Probabilidad')
                        ax.set_title('Probabilidad de Clasificación')
                        ax.set_ylim(0, 1)
                        
                        # Agregar valores en las barras
                        for i, (bar, prob) in enumerate(zip(bars, probs)):
                            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                                   f'{prob:.2%}', ha='center', va='bottom')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Información adicional
                        st.markdown("""
                        <div class="info-box">
                            <h4>🩺 Próximos Pasos Recomendados</h4>
                            <ul>
                                <li>Consulte con un dermatólogo profesional</li>
                                <li>Lleve esta imagen a su cita médica</li>
                                <li>Documente cualquier cambio en la lesión</li>
                                <li>Mantenga un seguimiento regular</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("No se pudo analizar la imagen. Inténtelo nuevamente.")
        else:
            st.info("👆 Por favor, suba una imagen para comenzar el análisis")
    
    # Información adicional
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📚 Información Adicional</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Sobre el Melanoma
        El melanoma es el tipo más serio de cáncer de piel. Se desarrolla en las células que producen melanina (pigmento). La detección temprana es crucial para el tratamiento exitoso.
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Regla ABCDE
        - **A**simetría
        - **B**ordes irregulares
        - **C**olor variado
        - **D**iámetro > 6mm
        - **E**volución/cambios
        """)
    
    with col3:
        st.markdown("""
        ### 📞 Cuándo Consultar
        - Lunares nuevos
        - Cambios en lunares existentes
        - Picazón o sangrado
        - Crecimiento rápido
        - Cualquier preocupación
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🔬 Desarrollado con TensorFlow y Streamlit</p>
        <p>⚠️ Solo para fines educativos - No reemplaza el diagnóstico médico profesional</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
