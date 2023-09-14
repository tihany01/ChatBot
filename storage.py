import os
import numpy as np
from dotenv import load_dotenv, find_dotenv
import openai
from chromadb.api.types import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings


load_dotenv(find_dotenv())

os.environ["OPENAI_API_KEY"] = os.environ['OPENAI_API_KEY']
openai.api_key = os.environ["OPENAI_API_KEY"]


directorio_vectores = 'vectores/'

# Crea el directorio si no existe
if not os.path.exists(directorio_vectores):
    os.makedirs(directorio_vectores)

# Define el directorio donde están los archivos de texto fragmentados
directorio_fragmentos = 'fragmentos/'

# Inicializa los embeddings de OpenAI
embeddings = OpenAIEmbeddings()

# Recorre cada archivo en el directorio de fragmentos
for archivo in os.listdir(directorio_fragmentos):
    if archivo.endswith('.txt'):
        texto = None
        try:
            with open(os.path.join(directorio_fragmentos, archivo), 'r', encoding='utf-8') as f:
                texto = f.read()
        except UnicodeDecodeError:
            try:
                with open(os.path.join(directorio_fragmentos, archivo), 'r', encoding='latin-1') as f:
                    texto = f.read()
            except Exception as e:
                print(f"Error al decodificar el archivo {archivo} con UTF-8 y latin-1. Omitiendo... Error: {e}")
                continue

        if texto:
            # Obtiene los embeddings para el fragmento de texto
            embedding = embeddings.embed_query(texto)
            
            # Guarda el embedding en un nuevo archivo
            nombre_embedding = f"{archivo[:-4]}_embedding.npy"
            np.save(os.path.join(directorio_vectores, nombre_embedding), embedding)