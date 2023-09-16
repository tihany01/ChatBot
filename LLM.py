import os
import glob
import numpy as np
from dotenv import load_dotenv, find_dotenv
import openai
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings.openai import OpenAIEmbeddings  
from SPARQLWrapper import SPARQLWrapper, JSON

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]
# Function to save data in AnzoGraph
def guardar_en_anzograph(consulta, respuesta):
    endpoint_url = "http://localhost:80/sparql"

    username = "admin"
    password = "Passw0rd1"

    sparql = SPARQLWrapper(endpoint_url)
    sparql.setCredentials(username, password)
    sparql.setMethod("POST")  # Set method to POST
    
    # Format your SPARQL query to save the data here
    consulta_sparql = f"""
        INSERT DATA {{
            <http://example.com/{consulta}> <http://example.com/respuesta> "{respuesta}" .
        }}
    """
    
    sparql.setQuery(consulta_sparql)
    
    try:
        sparql.query()
        print("Datos guardados en AnzoGraph exitosamente.")
    except Exception as e:
        print(f"Error al guardar datos en AnzoGraph: {e}")

# Function to retrieve data from AnzoGraph
def recuperar_de_anzograph():
    endpoint_url = "http://localhost:80/sparql"  
    username = "admin"
    password = "Passw0rd1"

    sparql = SPARQLWrapper(endpoint_url)
    sparql.setCredentials(username, password)
    sparql.setMethod("GET")  # Use GET for SELECT queries

    consulta_sparql = "SELECT * WHERE { ?s ?p ?o } LIMIT 20"  
    
    sparql.setQuery(consulta_sparql)
    sparql.setReturnFormat(JSON)

    try:
        resultados = sparql.query().convert()
        print("Resultado completo:")
        print(resultados) # Print the entire result for debugging

        print("Resultados recuperados de AnzoGraph:")
        for resultado in resultados['results']['bindings']:
            print(resultado)
    except Exception as e:
        print(f"Error al recuperar datos de AnzoGraph: {e}")


# Function to generate a response using the GPT-3 model
def generar_respuesta_gpt3(pregunta):
    try:
        # Specify the model to be used
        modelo = "gpt-3.5-turbo"
        # Generate a response using OpenAI's GPT-3
        respuesta_gpt3 = openai.ChatCompletion.create(
            model=modelo,
            messages=[
                {"role": "system", "content": "Estás hablando con un asistente de viajes."},
                {"role": "user", "content": pregunta}
            ]
        )
        # Return the generated response
        return respuesta_gpt3['choices'][0]['message']['content'].strip()
    except Exception as e:
        # Log any errors that occur
        print(f"Error al generar respuesta con GPT-3: {e}")
        return None

# Function to load all vectors from the 'vectores' directory
def cargar_todos_vectores():
    # Initialize an empty list to store vectors
    vectores = []
    # Loop through all .npy files and load vectors
    for npy_file in glob.glob("vectores/*.npy"):
        try:
            vector = np.load(npy_file)
            vectores.append((npy_file, vector))
        except Exception as e:
            # Log any errors that occur
            print(f"Error al cargar el archivo {npy_file}: {e}")
    # Return the list of vectors
    return vectores

# Function to find the most similar vector from a list
def encontrar_vector_similar(vector_consulta, vectores):
    # Calculate cosine similarity between query vector and all loaded vectors
    vectores_nombre = [x[0] for x in vectores]
    vectores_valores = np.array([x[1] for x in vectores])
    
    similitudes = cosine_similarity([vector_consulta], vectores_valores)[0]
    media_similitud = np.mean(similitudes)
    std_similitud = np.std(similitudes)
    
    # Establecer el umbral de similitud de manera dinámica
    umbral_similitud = media_similitud + (3* std_similitud)
    # Return the most similar vector, its similarity, and the threshold
    indice_max_similitud = np.argmax(similitudes)
    return vectores_nombre[indice_max_similitud], similitudes[indice_max_similitud], umbral_similitud
# Function to fetch the text snippet corresponding to a vector
def buscar_fragmento(nombre_archivo_vector):
    # Extract fragment ID from the file name
    fragmento_id = os.path.basename(nombre_archivo_vector).replace("_embedding.npy", ".txt")
    fragmento_texto = f"fragmentos/{fragmento_id}"
    # Read the corresponding text fragment from the 'fragmentos' directory
    try:
        with open(fragmento_texto, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fragmento_texto, 'r', encoding='latin-1') as f:
            return f.read()
# Function to process user query and generate a response
def procesar_consulta(vector_consulta, vectores, consulta_usuario):
    if vector_consulta is None or not vectores:
        return "Lo siento, hubo un error al procesar tu consulta."
    # Find the most similar vector
    nombre_archivo_vector, max_similitud, umbral_similitud = encontrar_vector_similar(vector_consulta, vectores)

    # Check if the maximum similarity is above the threshold
    if max_similitud < umbral_similitud:
        return "Lo siento, no pude encontrar información relevante sobre tu consulta."
    
    fragmento = buscar_fragmento(nombre_archivo_vector)
    if fragmento:
        # Imprimir el nombre del archivo del fragmento
        print(f"Respondiendo con el fragmento: {nombre_archivo_vector}")
        # Generar un prompt más explícito para GPT-3
        prompt = f"El usuario preguntó: '{consulta_usuario}'. Basándote en la siguiente información que encontré: '{fragmento}', ¿podrías proporcionar una respuesta?"
        respuesta = generar_respuesta_gpt3(prompt)
        return respuesta if respuesta else "Lo siento, no pude encontrar la información relacionada."
    else:
        return "Lo siento, no pude encontrar la información relacionada."


# Main function that orchestrates the above functions
def main(consulta_usuario):
    # Load all vectors
    vectores = cargar_todos_vectores()
    consulta_usuario = consulta_usuario.lower()
    embeddings = OpenAIEmbeddings()
    vector_consulta = embeddings.embed_query(consulta_usuario)
    respuesta = procesar_consulta(vector_consulta, vectores, consulta_usuario)
    return respuesta

