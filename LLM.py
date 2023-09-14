import os
import glob
import numpy as np
from dotenv import load_dotenv, find_dotenv
import openai
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings.openai import OpenAIEmbeddings  

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]
def generar_respuesta_gpt3(pregunta):
    try:
        modelo = "gpt-3.5-turbo"
        respuesta_gpt3 = openai.ChatCompletion.create(
            model=modelo,
            messages=[
                {"role": "system", "content": "Estás hablando con un asistente de viajes."},
                {"role": "user", "content": pregunta}
            ]
        )
        return respuesta_gpt3['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error al generar respuesta con GPT-3: {e}")
        return None

def cargar_todos_vectores():
    vectores = []
    for npy_file in glob.glob("vectores/*.npy"):
        try:
            vector = np.load(npy_file)
            vectores.append((npy_file, vector))
        except Exception as e:
            print(f"Error al cargar el archivo {npy_file}: {e}")
    return vectores

def encontrar_vector_similar(vector_consulta, vectores):
    vectores_nombre = [x[0] for x in vectores]
    vectores_valores = np.array([x[1] for x in vectores])
    
    similitudes = cosine_similarity([vector_consulta], vectores_valores)[0]
    media_similitud = np.mean(similitudes)
    std_similitud = np.std(similitudes)
    
    # Establecer el umbral de similitud de manera dinámica
    umbral_similitud = media_similitud + (0.6 * std_similitud)
    
    indice_max_similitud = np.argmax(similitudes)
    return vectores_nombre[indice_max_similitud], similitudes[indice_max_similitud], umbral_similitud

def buscar_fragmento(nombre_archivo_vector):
    fragmento_id = os.path.basename(nombre_archivo_vector).replace("_embedding.npy", ".txt")
    fragmento_texto = f"fragmentos/{fragmento_id}"

    try:
        with open(fragmento_texto, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fragmento_texto, 'r', encoding='latin-1') as f:
            return f.read()

def procesar_consulta(vector_consulta, vectores, consulta_usuario):
    if vector_consulta is None or not vectores:
        return "Lo siento, hubo un error al procesar tu consulta."

    nombre_archivo_vector, max_similitud, umbral_similitud = encontrar_vector_similar(vector_consulta, vectores)

    if max_similitud > umbral_similitud:
        fragmento = buscar_fragmento(nombre_archivo_vector)
        if fragmento:
            # Generar un prompt más explícito para GPT-3
            prompt = f"El usuario preguntó: '{consulta_usuario}'. Basándote en la siguiente información que encontré: '{fragmento}', ¿podrías proporcionar una respuesta?"
            respuesta = generar_respuesta_gpt3(prompt)
            return respuesta if respuesta else "Lo siento, no pude encontrar la información relacionada."
        else:
            return "Lo siento, no pude encontrar la información relacionada."
    else:
        respuesta = generar_respuesta_gpt3(consulta_usuario)
        return respuesta if respuesta else "Lo siento, no pude encontrar una coincidencia cercana."


def main(consulta_usuario):
    vectores = cargar_todos_vectores()
    consulta_usuario = consulta_usuario.lower()
    embeddings = OpenAIEmbeddings()
    vector_consulta = embeddings.embed_query(consulta_usuario)
    respuesta = procesar_consulta(vector_consulta, vectores, consulta_usuario)
    return respuesta


if __name__ == "__main__":
    main()



