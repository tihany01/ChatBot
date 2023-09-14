from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Define el directorio donde están los archivos de texto preprocesados
directorio_preprocesado = 'informacion/'

# Define el directorio donde se guardarán los fragmentos
directorio_fragmentos = 'fragmentos/'

# Crea el directorio si no existe
if not os.path.exists(directorio_fragmentos):
    os.makedirs(directorio_fragmentos)

# Define el tamaño del fragmento y el solapamiento
tamanio_fragmento = 1800  # Puedes ajustar este valor según tus necesidades
solapamiento = 80  # Puedes ajustar este valor según tus necesidades

# Inicializa el divisor de texto
splitter = RecursiveCharacterTextSplitter(chunk_size=tamanio_fragmento, chunk_overlap=solapamiento)

# Recorre cada archivo en el directorio preprocesado
for archivo in os.listdir(directorio_preprocesado):
    if archivo.endswith('.txt'):
        texto = None
        try:
            with open(os.path.join(directorio_preprocesado, archivo), 'r', encoding='utf-8') as f:
                texto = f.read().lower()  # Convierte el texto a minúsculas
        except UnicodeDecodeError:
            try:
                with open(os.path.join(directorio_preprocesado, archivo), 'r', encoding='latin-1') as f:
                    texto = f.read().lower()  # Convierte el texto a minúsculas
            except Exception as e:
                print(f"Error al decodificar el archivo {archivo} con UTF-8 y latin-1. Omitiendo... Error: {e}")
                continue

        if texto:
            # Divide el texto en fragmentos
            fragmentos = splitter.split_text(texto)
            
            # Guarda cada fragmento en un nuevo archivo
            for i, fragmento in enumerate(fragmentos):
                nombre_fragmento = f"{archivo[:-4]}_parte_{i+1}.txt"
                with open(os.path.join(directorio_fragmentos, nombre_fragmento), 'w', encoding='utf-8') as f:
                    f.write(fragmento)
