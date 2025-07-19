import os
import shutil
import random
from pathlib import Path

def create_directory_structure():
    """Crear la estructura de directorios necesaria"""
    directories = [
        'melanoma_cancer_dataset/train/benign',
        'melanoma_cancer_dataset/train/malignant',
        'melanoma_cancer_dataset/test/benign',
        'melanoma_cancer_dataset/test/malignant'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Directorio creado: {directory}")

def organize_images(source_train_dir, source_test_dir):
    """
    Organizar las imágenes en la estructura correcta
    
    Args:
        source_train_dir: Directorio con las imágenes de entrenamiento
        source_test_dir: Directorio con las imágenes de test
    """
    
    print("Organizando imágenes...")
    
    # Estructura esperada en directorio fuente:
    # source_train_dir/
    #   ├── benign/     (3000 imágenes)
    #   └── malignant/  (3000 imágenes)
    #
    # source_test_dir/
    #   ├── benign/     (500 imágenes)
    #   └── malignant/  (500 imágenes)
    
    # Copiar imágenes de entrenamiento
    if os.path.exists(source_train_dir):
        for class_name in ['benign', 'malignant']:
            src_path = os.path.join(source_train_dir, class_name)
            dst_path = f'data/train/{class_name}'
            
            if os.path.exists(src_path):
                images = [f for f in os.listdir(src_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                print(f"Copiando {len(images)} imágenes {class_name} de entrenamiento...")
                
                for img in images:
                    src_file = os.path.join(src_path, img)
                    dst_file = os.path.join(dst_path, img)
                    shutil.copy2(src_file, dst_file)
    
    # Copiar imágenes de test
    if os.path.exists(source_test_dir):
        for class_name in ['benign', 'malignant']:
            src_path = os.path.join(source_test_dir, class_name)
            dst_path = f'data/test/{class_name}'
            
            if os.path.exists(src_path):
                images = [f for f in os.listdir(src_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                print(f"Copiando {len(images)} imágenes {class_name} de test...")
                
                for img in images:
                    src_file = os.path.join(src_path, img)
                    dst_file = os.path.join(dst_path, img)
                    shutil.copy2(src_file, dst_file)

def verify_data_structure():
    """Verificar que la estructura de datos esté correcta"""
    print("\n=== VERIFICACIÓN DE DATOS ===")
    
    required_dirs = [
       'melanoma_cancer_dataset/train/benign',
        'melanoma_cancer_dataset/train/malignant',
        'melanoma_cancer_dataset/test/benign',
        'melanoma_cancer_dataset/test/malignant'
    ]
    
    total_images = 0
    
    for directory in required_dirs:
        if os.path.exists(directory):
            images = [f for f in os.listdir(directory) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(images)
            total_images += count
            print(f"✅ {directory}: {count} imágenes")
        else:
            print(f"❌ {directory}: No existe")
    
    print(f"\nTotal de imágenes: {total_images}")
    
    # Verificar distribución esperada
    expected_counts = {
        'melanoma_cancer_dataset/train/benign': 3000,
        'melanoma_cancer_dataset/train/malignant': 3000,
        'melanoma_cancer_dataset/test/benign': 500,
        'melanoma_cancer_dataset/test/malignant': 500
    }
    
    print("\n=== VERIFICACIÓN DE CANTIDADES ===")
    for directory, expected in expected_counts.items():
        if os.path.exists(directory):
            actual = len([f for f in os.listdir(directory) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            status = "✅" if actual == expected else "⚠️"
            print(f"{status} {directory}: {actual}/{expected} imágenes")

def main():
    """Función principal para configurar los datos"""
    print("=== CONFIGURACIÓN DE DATOS PARA DETECCIÓN DE CÁNCER DE PIEL ===\n")
    
    # Crear estructura de directorios
    create_directory_structure()
    
    # Solicitar rutas de los directorios fuente
    print("\nPor favor, proporciona las rutas de tus directorios de imágenes:")
    print("Estructura esperada:")
    print("  entrenamiento/")
    print("    ├── benign/     (3000 imágenes)")
    print("    └── malignant/  (3000 imágenes)")
    print("  test/")
    print("    ├── benign/     (500 imágenes)")
    print("    └── malignant/  (500 imágenes)")
    
    source_train = input("\nRuta del directorio de entrenamiento: ").strip()
    source_test = input("Ruta del directorio de test: ").strip()
    
    # Organizar imágenes
    if source_train and source_test:
        organize_images(source_train, source_test)
    else:
        print("⚠️ Las rutas no fueron proporcionadas. Creando estructura vacía.")
    
    #