# 🔬 Detector de Cáncer de Piel con IA

Una aplicación web desarrollada con **Streamlit** y **TensorFlow** que utiliza deep learning para detectar melanomas malignos y benignos en imágenes de lesiones cutáneas.

## ⚠️ Advertencia Importante

**Esta aplicación es solo para fines educativos y de investigación. NO reemplaza el diagnóstico médico profesional. Siempre consulte con un dermatólogo para un diagnóstico definitivo.**

## 🎯 Características

- **Interfaz intuitiva** con Streamlit
- **Modelo CNN profundo** entrenado con 6,000 imágenes
- **Análisis en tiempo real** de imágenes de lesiones cutáneas
- **Visualización de resultados** con gráficos de confianza
- **Información educativa** sobre melanomas
- **Diseño responsive** y fácil de usar

## 📊 Datos del Modelo

- **Entrenamiento**: 6,000 imágenes (3,000 benignas + 3,000 malignas)
- **Prueba**: 1,000 imágenes (500 benignas + 500 malignas)
- **Arquitectura**: CNN con 5 capas convolucionales
- **Precisión estimada**: ~85-90%

## 🚀 Instalación y Uso

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/detector-cancer-piel.git
cd detector-cancer-piel
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Organizar los datos
```bash
python setup_data.py
```

### 4. Entrenar el modelo
```bash
python train_model.py
```

### 5. Ejecutar la aplicación
```bash
streamlit run app.py
```

## 📁 Estructura del Proyecto

```
detector-cancer-piel/
├── app.py                 # Aplicación Streamlit
├── train_model.py         # Script de entrenamiento
├── setup_data.py          # Organización de datos
├── requirements.txt       # Dependencias
├── README.md             # Este archivo
├── skin_cancer_model.h5  # Modelo entrenado (generado)
└── data/                 # Datos de entrenamiento y test
    ├── train/
    │   ├── benign/       # 3,000 imágenes benignas
    │   └── malignant/    # 3,000 imágenes malignas
    └── test/
        ├── benign/       # 500 imágenes benignas
        └── malignant/    # 500 imágenes malignas
```

## 🔧 Configuración para Streamlit Cloud

### 1. Subir a GitHub
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 2. Deployment en Streamlit Cloud
1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Conecta tu cuenta de GitHub
3. Selecciona el repositorio
4. Configura:
   - **Main file path**: `app.py`
   - **Python version**: 3.9
5. Haz clic en "Deploy"

### 3. Configuración adicional
Crea un archivo `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## 🏥 Información Médica

### Regla ABCDE para Melanomas
- **A**simetría: La mitad no coincide con la otra
- **B**ordes: Bordes irregulares, desiguales o borrosos
- **C**olor: Variaciones de color en la misma lesión
- **D**iámetro: Mayor a 6mm (tamaño de un borrador)
- **E**volución: Cambios en tamaño, forma o color

### Cuándo Consultar un Médico
- Aparición de lunares nuevos
- Cambios en lunares existentes
- Picazón, sangrado o dolor
- Crecimiento rápido
- Cualquier preocupación sobre una lesión

## 🛠️ Tecnologías Utilizadas

- **Python 3.9+**
- **TensorFlow 2.13** - Deep Learning
- **Streamlit 1.28** - Interfaz web
- **OpenCV** - Procesamiento de imágenes
- **Matplotlib/Seaborn** - Visualización
- **NumPy/Pandas** - Manipulación de datos
- **Scikit-learn** - Métricas de evaluación

## 📈 Arquitectura del Modelo

```python
Sequential([
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(512, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

## 🎨 Características de la Aplicación

- **Interfaz moderna** con CSS personalizado
- **Carga de imágenes** drag-and-drop
- **Análisis en tiempo real** con barra de progreso
- **Resultados visuales** con gráficos de confianza
- **Información educativa** integrada
- **Responsive design** para móviles

## 🔍 Mejoras Futuras

- [ ] Implementar modelos pre-entrenados (ResNet, EfficientNet)
- [ ] Añadir detección de otros tipos de cáncer de piel
- [ ] Integrar con APIs médicas
- [ ] Implementar sistema de historial
- [ ] Añadir funcionalidad de exportación de reportes
- [ ] Mejorar la precisión del modelo
- [ ] Implementar técnicas de explainable AI

## 📝 Contribuir

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🤝 Reconocimientos

- Dataset de imágenes de melanomas
- Comunidad de TensorFlow y Streamlit
- Investigadores en dermatología computacional

## 📞 Contacto

- **Autor**: Diego Cantillano
- **Email**: diego.cantillano@gmail.com
- **GitHub**: https://github.com/diegocantillano (https://github.com/diegocantillano)

## ⚖️ Disclaimer

Esta aplicación ha sido desarrollada con fines educativos y de investigación. Los resultados no deben ser utilizados para diagnóstico médico. Siempre consulte con profesionales médicos calificados para cualquier preocupación de salud.

---

**🔬 Desarrollado con ❤️ usando TensorFlow y Streamlit**
