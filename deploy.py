import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Ejecutar un comando y mostrar el resultado"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completado exitosamente")
            if result.stdout:
                print(f"📄 Output: {result.stdout.strip()}")
        else:
            print(f"❌ Error en {description}")
            print(f"🔴 Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"❌ Excepción durante {description}: {str(e)}")
        return False

def check_git_installed():
    """Verificar si Git está instalado"""
    return run_command("git --version", "Verificación de Git")

def check_model_exists():
    """Verificar si el modelo entrenado existe"""
    if os.path.exists('skin_cancer_model.h5'):
        print("✅ Modelo entrenado encontrado")
        return True
    else:
        print("⚠️ Modelo no encontrado. Ejecuta 'python train_model.py' primero")
        return False

def create_gitignore():
    """Crear archivo .gitignore"""
    gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Streamlit
.streamlit/secrets.toml

# Datos de entrenamiento (opcional - comentar si quieres incluir los datos)
# data/train/
# data/test/

# Archivos temporales
*.tmp
*.temp
.DS_Store
Thumbs.db

# Modelos grandes (opcional - comentar si quieres incluir el modelo)
# *.h5
# *.pkl
# *.joblib
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content.strip())
    
    print("✅ Archivo .gitignore creado")

def git_setup():
    """Configurar repositorio Git"""
    print("\n=== CONFIGURACIÓN DE GIT ===")
    
    # Verificar si ya es un repositorio Git
    if os.path.exists('.git'):
        print("✅ Repositorio Git ya existe")
    else:
        if not run_command("git init", "Inicialización del repositorio Git"):
            return False
    
    # Crear .gitignore
    create_gitignore()
    
    # Configurar usuario (opcional)
    print("\n📝 Configuración de usuario Git:")
    user_name = input("Nombre de usuario Git (Enter para omitir): ").strip()
    user_email = input("Email Git (Enter para omitir): ").strip()
    
    if user_name:
        run_command(f'git config user.name "{user_name}"', "Configuración de nombre de usuario")
    if user_email:
        run_command(f'git config user.email "{user_email}"', "Configuración de email")
    
    return True

def github_setup():
    """Configurar para GitHub"""
    print("\n=== CONFIGURACIÓN DE GITHUB ===")
    
    print("📋 Pasos para subir a GitHub:")
    print("1. Crea un nuevo repositorio en GitHub")
    print("2. Copia la URL del repositorio")
    
    repo_url = input("\n🔗 URL del repositorio GitHub: ").strip()
    
    if repo_url:
        # Agregar remote origin
        run_command(f"git remote add origin {repo_url}", "Configuración de remote origin")
        
        # Verificar remote
        run_command("git remote -v", "Verificación de remote")
        
        return True
    else:
        print("⚠️ URL no proporcionada. Configuración manual necesaria.")
        return False

def prepare_deployment():
    """Preparar archivos para deployment"""
    print("\n=== PREPARACIÓN PARA DEPLOYMENT ===")
    
    # Verificar archivos necesarios
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} encontrado")
        else:
            print(f"❌ {file} no encontrado")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ Faltan archivos: {', '.join(missing_files)}")
        return False
    
    # Crear directorio .streamlit si no existe
    Path('.streamlit').mkdir(exist_ok=True)
    
    print("✅ Archivos preparados para deployment")
    return True

def git_commit_and_push():
    """Hacer commit y push a GitHub"""
    print("\n=== COMMIT Y PUSH ===")
    
    # Agregar archivos
    if not run_command("git add .", "Agregando archivos al staging"):
        return False
    
    # Verificar status
    run_command("git status", "Estado del repositorio")
    
    # Commit
    commit_message = input("\n💬 Mensaje del commit (Enter para 'Initial commit'): ").strip()
    if not commit_message:
        commit_message = "Initial commit - Detector de Cancer de Piel"
    
    if not run_command(f'git commit -m "{commit_message}"', "Creando commit"):
        return False
    
    # Push
    branch = input("\n🌿 Rama para push (Enter para 'main'): ").strip()
    if not branch:
        branch = "main"
    
    if not run_command(f"git push -u origin {branch}", f"Push a la rama {branch}"):
        return False
    
    print("🎉 ¡Código subido exitosamente a GitHub!")
    return True

def show_streamlit_deployment_info():
    """Mostrar información sobre deployment en Streamlit Cloud"""
    print("\n=== DEPLOYMENT EN STREAMLIT CLOUD ===")
    print("""
🚀 Pasos para deployment en Streamlit Cloud:

1. Ve a https://share.streamlit.io
2. Conecta tu cuenta de GitHub
3. Selecciona tu repositorio
4. Configura:
   - Main file path: app.py
   - Python version: 3.9
5. Haz clic en "Deploy"

⚙️ Configuración adicional:
- El archivo .streamlit/config.toml ya está configurado
- Asegúrate de que el modelo skin_cancer_model.h5 esté en el repo
- La aplicación estará disponible en pocos minutos

🔗 Enlaces útiles:
- Documentación: https://docs.streamlit.io/streamlit-cloud
- Troubleshooting: https://docs.streamlit.io/streamlit-cloud/troubleshooting
""")

def main():
    """Función principal de deployment"""
    print("🚀 SCRIPT DE DEPLOYMENT PARA DETECTOR DE CÁNCER DE PIEL")
    print("=" * 60)
    
    # Verificar requisitos
    if not check_git_installed():
        print("❌ Git no está instalado. Instálalo desde https://git-scm.com/")
        return
    
    # Verificar modelo (opcional)
    model_exists = check_model_exists()
    if not model_exists:
        continue_anyway = input("¿Continuar sin modelo entrenado? (y/n): ").strip().lower()
        if continue_anyway != 'y':
            print("📋 Ejecuta 'python train_model.py' primero para entrenar el modelo")
            return
    
    # Preparar archivos
    if not prepare_deployment():
        print("❌ Error en la preparación de archivos")
        return
    
    # Configurar Git
    if not git_setup():
        print("❌ Error en la configuración de Git")
        return
    
    # Configurar GitHub
    if not github_setup():
        print("⚠️ Configuración de GitHub incompleta")
        setup_later = input("¿Configurar GitHub después manualmente? (y/n): ").strip().lower()
        if setup_later != 'y':
            return
    
    # Commit y push
    if not git_commit_and_push():
        print("❌ Error en el push a GitHub")
        return
    
    # Mostrar información de deployment
    show_streamlit_deployment_info()
    
    print("\n🎉 ¡DEPLOYMENT COMPLETADO EXITOSAMENTE!")
    print("🔗 Tu aplicación estará disponible en Streamlit Cloud en pocos minutos")

if __name__ == "__main__":
    main()
