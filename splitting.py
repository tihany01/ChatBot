from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Define the directory where the preprocessed text files are located
directorio_preprocesado = 'informacion/'

# Define the directory where the chunks will be saved
directorio_fragmentos = 'fragmentos/'

# Create the directory if it does not exist
if not os.path.exists(directorio_fragmentos):
    os.makedirs(directorio_fragmentos)

# Define the chunk size and overlap
tamanio_fragmento = 1800  # You can adjust this value according to your needs
solapamiento = 80  # You can adjust this value according to your needs

# Initialize the text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=tamanio_fragmento, chunk_overlap=solapamiento)

# Go through each file in the preprocessed directory
for archivo in os.listdir(directorio_preprocesado):
    if archivo.endswith('.txt'):
        # Check if split files for this text file already exist
        existen_archivos_divididos = any(nombre.startswith(archivo[:-4]) and nombre.endswith('.txt') for nombre in os.listdir(directorio_fragmentos))
        
        # If split files exist, skip this text file
        if existen_archivos_divididos:
            print(f"Los archivos divididos para '{archivo}' ya existen, omitiendo...")
            continue

        texto = None
        try:
            with open(os.path.join(directorio_preprocesado, archivo), 'r', encoding='utf-8') as f:
                texto = f.read().lower()  # Convert the text to lowercase
        except UnicodeDecodeError:
            try:
                with open(os.path.join(directorio_preprocesado, archivo), 'r', encoding='latin-1') as f:
                    texto = f.read().lower()  # Convert the text to
            except Exception as e:
                print(f"Error al decodificar el archivo {archivo} con UTF-8 y latin-1. Omitiendo... Error: {e}")
                continue

        if texto:
            # Split the text into chunks
            fragmentos = splitter.split_text(texto)
            
            # Save each chunk in a new file
            for i, fragmento in enumerate(fragmentos):
                nombre_fragmento = f"{archivo[:-4]}_parte_{i+1}.txt"
                with open(os.path.join(directorio_fragmentos, nombre_fragmento), 'w', encoding='utf-8') as f:
                    f.write(fragmento)
