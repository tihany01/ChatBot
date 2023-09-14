import os
import re
from langchain.document_loaders import WebBaseLoader

# Leer el archivo que contiene los enlaces
with open("lista_de_sitios.txt", "r") as f:
    urls = f.readlines()

# Crear el directorio si no existe
os.makedirs('informacion', exist_ok=True)

# Iterar a través de cada enlace
for i, url in enumerate(urls):
    url = url.strip()  # Eliminar cualquier espacio en blanco o nueva línea al final

    # Cargar la página web
    loader = WebBaseLoader(url)
    docs = loader.load()

    # Limpiar el texto
    cleaned_text = re.sub('<.*?>', '', docs[0].page_content)
    cleaned_text = re.sub('[^A-Za-z0-9áéíóúÁÉÍÓÚñÑ\s]+', '', cleaned_text)
    cleaned_text = re.sub('\s+', ' ', cleaned_text).strip()

    # Contar palabras
    word_count = len(cleaned_text.split())

    # Guardar el contenido en un archivo
    with open(f'informacion/sitioweb{i+1}.txt', 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    print(f"El documento {i+1} tiene {word_count} palabras.")
