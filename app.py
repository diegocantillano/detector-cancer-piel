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
import os
import warnings

# Suprimir warnings específicos
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", message="Compiled the loaded model")

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
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stSpinner > div > div {
        border-top-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def create_demo_model():
    """Crear un modelo de demostración simple para pruebas"""
    try:
        # Suprimir logs de TensorFlow durante la creación
        tf.get_logger().setLevel('ERROR')
        
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compilar el modelo correctamente
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Construir el modelo con datos dummy para evitar warnings
        dummy_input = np.zeros((1, 224, 224, 3))
        _ = model.predict(dummy_input, verbose=0)
        
        return model
    except Exception as e:
        st.error(f"Error creando modelo de demostración: {str(e)}")
        return None

@st.cache_resource
def load_skin_cancer_model():
    """Cargar el modelo entrenado o crear uno de demostración"""
    # Suprimir logs de TensorFlow
    tf.get_logger().setLevel('ERROR')
    
    model_paths = [
        'models/skin_cancer_model.h5',
        'skin_cancer_model.h5',
        './models/skin_cancer_model.h5',
        'model.h5'
    ]
    
    # Intentar cargar el modelo desde diferentes rutas
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                # Cargar modelo sin mostrar warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = load_model(model_path, compile=False)
                
                # Recompilar el modelo para evitar warnings
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Construir métricas con predicción dummy
                dummy_input = np.zeros((1, 224, 224, 3))
                _ = model.predict(dummy_input, verbose=0)
                
                return model, True
            except Exception as e:
                continue
    
    # Si no se puede cargar el modelo real, crear uno de demostración
    demo_model = create_demo_model()
    return demo_model, False

class SkinCancerDetector:
    def __init__(self):
        self.model = None
        self.is_real_model = False
        self.img_size = (224, 224)
        self.load_model()
    
    def load_model(self):
        """Cargar el modelo"""
        try:
            with st.spinner("Cargando modelo de IA..."):
                self.model, self.is_real_model = load_skin_cancer_model()
                if self.model is None:
                    raise Exception("No se pudo cargar ningún modelo")
        except Exception as e:
            st.error(f"Error crítico al cargar el modelo: {str(e)}")
            self.model = None
            self.is_real_model = False
    
    def preprocess_image(self, image):
        """Preprocesar la imagen para el modelo"""
        try:
            # Convertir a RGB si es necesario
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Redimensionar usando LANCZOS para mejor calidad
            image = image.resize(self.img_size, Image.Resampling.LANCZOS)
            
            # Convertir a array numpy
            img_array = np.array(image, dtype=np.float32)
            
            # Normalizar a [0, 1]
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
            
            # Hacer predicción sin verbose
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction = self.model.predict(processed_image, verbose=0)
            
            if self.is_real_model:
                confidence = float(prediction[0][0])
            else:
                # Para el modelo demo, generar una predicción semi-realista
                # basada en características de la imagen
                img_mean = np.mean(processed_image)
                img_std = np.std(processed_image)
                
                # Simulación más realista basada en características de la imagen
                base_prob = (img_mean + img_std) * 0.5
                noise = np.random.normal(0, 0.1)
                confidence = np.clip(base_prob + noise, 0.05, 0.95)
            
            # Interpretar resultado
            if confidence > 0.5:
                result = "Maligno"
                risk_level = "Alto"
                recommendation = "Consulte inmediatamente con un dermatólogo"
            else:
                result = "Benigno"
                risk_level = "Bajo"
                recommendation = "Mantenga observación periódica y consulte con su médico"
            
            return result, confidence, risk_level, recommendation
        except Exception as e:
            st.error(f"Error en la predicción: {str(e)}")
            return None, None, None, None

def create_confidence_chart(result, confidence):
    """Crear gráfico de confianza mejorado"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if result == "Maligno":
        probs = [1-confidence, confidence]
        labels = ['Benigno', 'Maligno']
        colors = ['#28a745', '#dc3545']
    else:
        probs = [1-confidence, confidence]
        labels = ['Benigno', 'Maligno']
        colors = ['#28a745', '#dc3545']
    
    bars = ax.bar(labels, probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax.set_ylabel('Probabilidad', fontsize=12)
    ax.set_title('Distribución de Probabilidades', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar valores en las barras
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Mejorar el estilo
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    return fig

def main():
    # Título principal
    st.markdown('<h1 class="main-header">🔬 Detector de Cáncer de Piel</h1>', unsafe_allow_html=True)
    
    # Inicializar detector
    if 'detector' not in st.session_state:
        st.session_state.detector = SkinCancerDetector()
    
    detector = st.session_state.detector
    
    # Advertencia si no hay modelo real
    if detector.model is None:
        st.error("❌ No se pudo cargar ningún modelo. La aplicación no puede funcionar.")
        st.stop()
    elif not detector.is_real_model:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Modo de Demostración Activo</h3>
            <p>Esta aplicación está ejecutándose en <strong>modo de demostración</strong> porque no se encontró el modelo entrenado.</p>
            <p>Para usar el modelo real, asegúrese de que el archivo del modelo esté disponible en las rutas especificadas.</p>
            <p><strong>⚠️ Los resultados mostrados son simulaciones y NO son diagnósticos médicos reales.</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Información sobre la aplicación
    st.markdown("""
    <div class="info-box">
        <h3>ℹ️ Información Importante</h3>
        <p>Esta herramienta utiliza inteligencia artificial para analizar imágenes de lunares y lesiones cutáneas, 
        como apoyo en la detección temprana de melanomas.</p>
        <p><strong>🩺 IMPORTANTE: Esta herramienta NO reemplaza el diagnóstico médico profesional. 
        Siempre consulte con un dermatólogo para obtener un diagnóstico definitivo.</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar con información
    with st.sidebar:
        st.markdown('<h2 class="sub-header">📋 Guía de Uso</h2>', unsafe_allow_html=True)
        st.markdown("""
        **Pasos a seguir:**
        1. 📤 **Suba una imagen** de la lesión cutánea
        2. 🔍 **Verifique** que la imagen sea clara y bien iluminada
        3. 🔬 **Haga clic en "Analizar"** para obtener el resultado
        4. 📊 **Revise** la predicción y el nivel de confianza
        5. 🩺 **Consulte** con un dermatólogo profesional
        """)
        
        st.markdown('<h2 class="sub-header">⚠️ Limitaciones</h2>', unsafe_allow_html=True)
        st.markdown("""
        - ✋ **Solo para fines educativos e informativos**
        - 🎯 **Precisión limitada** - no es 100% confiable
        - 🚫 **No reemplaza** el criterio médico profesional
        - 📸 **Requiere imágenes** de alta calidad y buena iluminación
        - 🔬 **Resultados orientativos** - siempre confirme con especialista
        """)
        
        st.markdown('<h2 class="sub-header">📊 Estado del Sistema</h2>', unsafe_allow_html=True)
        if detector.model is not None:
            if detector.is_real_model:
                st.success("✅ Modelo entrenado cargado correctamente")
                model_status = "Modelo Real"
                accuracy = "~85-90%"
            else:
                st.warning("⚠️ Modo demostración activo")
                model_status = "Modo Demostración"
                accuracy = "N/A (simulación)"
        else:
            st.error("❌ Error en la carga del modelo")
            model_status = "Error"
            accuracy = "N/A"
        
        st.markdown(f"""
        **Información técnica:**
        - **Estado**: {model_status}
        - **Arquitectura**: Red Neuronal Convolucional
        - **Entrada**: 224×224 píxeles
        - **Precisión estimada**: {accuracy}
        - **Clases**: Benigno / Maligno
        """)
    
    # Área principal de la aplicación
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Subir Imagen</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Seleccione una imagen de la lesión cutánea",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Formatos soportados: JPG, JPEG, PNG, BMP. Tamaño máximo: 200MB"
        )
        
        if uploaded_file is not None:
            try:
                # Mostrar imagen original
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen cargada", use_container_width=True)
                
                # Información de la imagen
                file_size = len(uploaded_file.getvalue())
                st.markdown(f"""
                **Detalles de la imagen:**
                - **Dimensiones**: {image.size[0]} × {image.size[1]} píxeles
                - **Formato**: {image.format}
                - **Modo de color**: {image.mode}
                - **Tamaño del archivo**: {file_size / 1024:.1f} KB
                """)
            except Exception as e:
                st.error(f"Error al cargar la imagen: {str(e)}")
                uploaded_file = None
    
    with col2:
        st.markdown('<h2 class="sub-header">🔍 Resultado del Análisis</h2>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Botón para analizar
            analyze_button = st.button(
                "🔬 Analizar Imagen", 
                type="primary", 
                disabled=(detector.model is None),
                use_container_width=True
            )
            
            if analyze_button:
                if detector.model is None:
                    st.error("❌ No hay modelo disponible para realizar la predicción.")
                else:
                    with st.spinner("🔍 Analizando imagen... Por favor espere..."):
                        # Hacer predicción
                        result_data = detector.predict(image)
                        
                        if len(result_data) == 4 and result_data[0] is not None:
                            result, confidence, risk_level, recommendation = result_data
                            
                            # Mostrar resultado principal
                            if result == "Maligno":
                                st.markdown(f"""
                                <div class="result-box malignant-result">
                                    <h3>⚠️ Resultado: {result}</h3>
                                    <p><strong>Nivel de riesgo:</strong> {risk_level}</p>
                                    <p><strong>Confianza del modelo:</strong> {confidence:.1%}</p>
                                    <p><strong>Recomendación:</strong> {recommendation}</p>
                                    {'<p><em>⚠️ Este es un resultado de demostración - NO es un diagnóstico médico real</em></p>' if not detector.is_real_model else ''}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="result-box benign-result">
                                    <h3>✅ Resultado: {result}</h3>
                                    <p><strong>Nivel de riesgo:</strong> {risk_level}</p>
                                    <p><strong>Confianza del modelo:</strong> {(1-confidence):.1%}</p>
                                    <p><strong>Recomendación:</strong> {recommendation}</p>
                                    {'<p><em>ℹ️ Este es un resultado de demostración - NO es un diagnóstico médico real</em></p>' if not detector.is_real_model else ''}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Gráfico de confianza mejorado
                            st.markdown('<h3 class="sub-header">📊 Distribución de Probabilidades</h3>', unsafe_allow_html=True)
                            
                            fig = create_confidence_chart(result, confidence)
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)  # Liberar memoria
                            
                            # Información adicional sobre el resultado
                            st.markdown("""
                            <div class="info-box">
                                <h4>🩺 Pasos Recomendados</h4>
                                <ol>
                                    <li><strong>Consulta médica:</strong> Programe una cita con un dermatólogo</li>
                                    <li><strong>Documentación:</strong> Guarde esta imagen para mostrar al especialista</li>
                                    <li><strong>Seguimiento:</strong> Documente cualquier cambio en la lesión</li>
                                    <li><strong>Prevención:</strong> Use protector solar y examine su piel regularmente</li>
                                </ol>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Interpretación del resultado
                            confidence_level = "alta" if abs(confidence - 0.5) > 0.3 else "moderada" if abs(confidence - 0.5) > 0.1 else "baja"
                            st.info(f"💡 **Interpretación**: La confianza del modelo es **{confidence_level}** para esta predicción. "
                                   f"{'Recuerde que este es solo un resultado de demostración.' if not detector.is_real_model else 'Siempre consulte con un profesional médico para confirmación.'}")
                        else:
                            st.error("❌ No se pudo analizar la imagen. Por favor, inténtelo nuevamente con una imagen diferente.")
        else:
            st.info("👆 **Instrucciones**: Suba una imagen en la columna izquierda para comenzar el análisis")
            
            # Mostrar ejemplos de buenas imágenes
            st.markdown("""
            ### 📸 Consejos para una buena imagen:
            - **Iluminación**: Use luz natural o buena iluminación artificial
            - **Enfoque**: Asegúrese de que la lesión esté enfocada y nítida
            - **Distancia**: Tome la foto lo suficientemente cerca para ver detalles
            - **Fondo**: Use un fondo contrastante (claro para lesiones oscuras)
            - **Estabilidad**: Evite imágenes borrosas o movidas
            """)
    
    # Información educativa adicional
    st.markdown("---")
    st.markdown('<h2 class="sub-header">📚 Información Educativa</h2>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🎯 Sobre el Melanoma", "🔍 Regla ABCDE", "📞 Cuándo Consultar"])
    
    with tab1:
        st.markdown("""
        ### ¿Qué es el melanoma?
        
        El **melanoma** es el tipo más agresivo de cáncer de piel, aunque también es el menos común. 
        Se desarrolla en los **melanocitos**, las células que producen melanina (el pigmento que da color a la piel).
        
        **Datos importantes:**
        - 🎯 **Detección temprana**: Es clave para un tratamiento exitoso
        - 📈 **Incidencia**: Ha aumentado significativamente en las últimas décadas
        - ☀️ **Principal causa**: Exposición excesiva a radiación UV
        - 👥 **Factores de riesgo**: Piel clara, muchos lunares, antecedentes familiares
        - 💪 **Pronóstico**: Excelente si se detecta temprano (>90% supervivencia a 5 años)
        """)
    
    with tab2:
        st.markdown("""
        ### La Regla ABCDE para evaluar lunares
        
        Esta regla te ayuda a identificar cambios sospechosos en lunares:
        
        - **🅰️ Asimetría**: Una mitad no coincide con la otra
        - **🅱️ Bordes**: Bordes irregulares, dentados o mal definidos
        - **🎨 Color**: Color no uniforme con tonos marrones, negros, rojos, blancos o azules
        - **📏 Diámetro**: Mayor a 6mm (tamaño de un borrador de lápiz)
        - **🔄 Evolución**: Cambios en tamaño, forma, color, elevación o síntomas nuevos
        
        **⚠️ Importante**: Si observa cualquiera de estos signos, consulte con un dermatólogo.
        """)
    
    with tab3:
        st.markdown("""
        ### ¿Cuándo debe consultar a un médico?
        
        **Consulte inmediatamente si nota:**
        - 🆕 **Lunares nuevos** que aparecen después de los 30 años
        - 🔄 **Cambios** en lunares existentes (tamaño, forma, color)
        - 🩸 **Sangrado** o supuración de un lunar
        - 🔥 **Picazón, ardor** o sensibilidad
        - 📈 **Crecimiento rápido** de una lesión
        - 🟡 **Costras** que no cicatrizan
        
        **Revisiones regulares:**
        - 👀 **Autoexamen mensual** de toda la piel
        - 🩺 **Examen dermatológico anual** (o más frecuente si tiene factores de riesgo)
        - 📅 **Seguimiento** de lunares sospechosos con fotografías
        
        **📞 Recuerde**: Es mejor una consulta de más que un diagnóstico tardío.
        """)
    
    # Sección de solución de problemas
    with st.expander("🔧 Solución de Problemas y FAQ"):
        st.markdown("""
        ### Preguntas Frecuentes:
        
        **❓ ¿Por qué veo "Modo de Demostración"?**
        - El modelo entrenado no se encontró en el repositorio
        - Los resultados son simulaciones para propósitos educativos
        - Para usar el modelo real, contacte al desarrollador
        
        **❓ ¿Qué tan precisa es la herramienta?**
        - Los modelos reales pueden alcanzar 85-90% de precisión
        - La precisión depende de la calidad de la imagen
        - NUNCA reemplaza el diagnóstico médico profesional
        
        **❓ ¿Qué tipo de imágenes funcionan mejor?**
        - Imágenes claras, bien iluminadas y enfocadas
        - Tomadas con buena cámara (smartphone moderno es suficiente)
        - Sin filtros ni ediciones
        - Con buen contraste entre la lesión y la piel circundante
        
        **❓ ¿Puedo confiar 100% en los resultados?**
        - NO. Esta herramienta es solo de apoyo educativo
        - Los resultados pueden tener falsos positivos o negativos
        - SIEMPRE consulte con un dermatólogo para diagnóstico definitivo
        
        **❓ La aplicación es lenta, ¿qué puedo hacer?**
        - Las primeras predicciones pueden tardar más
        - Use imágenes más pequeñas si es posible
        - La velocidad depende de los recursos del servidor
        """)
    
    # Footer mejorado
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
        <h4>🔬 Detector de Cáncer de Piel</h4>
        <p><strong>Tecnologías utilizadas:</strong> TensorFlow • Streamlit • Python • OpenCV</p>
        <p><strong>⚠️ Aviso médico importante:</strong> Esta herramienta es solo para fines educativos e informativos. 
        No constituye asesoramiento médico y no debe utilizarse como sustituto del diagnóstico profesional.</p>
        <p><strong>🩺 Recomendación:</strong> Siempre consulte con un dermatólogo certificado para evaluaciones médicas.</p>
        <p><small>Versión 2.0 • Última actualización: Julio 2025</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    # Configurar logging de TensorFlow antes de ejecutar la app
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    main()
