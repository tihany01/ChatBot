import os
import re
from langchain.document_loaders import WebBaseLoader

# Read the file containing the links
with open("lista_de_sitios.txt", "r") as f:
    urls = [url.strip() for url in f.readlines()]  # Remove any trailing whitespace or newline

# Create the directory if it doesn't exist
os.makedirs('informacion', exist_ok=True)

# Iterate through each link
for i, url in enumerate(urls):
    # Load the web page
    loader = WebBaseLoader(url)
    docs = loader.load()

    # Clean the text
    cleaned_text = re.sub('<.*?>', '', docs[0].page_content)  # Remove HTML tags
    cleaned_text = re.sub('[^A-Za-z0-9áéíóúÁÉÍÓÚñÑ\s]+', '', cleaned_text)  # Remove non-alphanumeric characters except spaces and Spanish accents
    cleaned_text = re.sub('\s+', ' ', cleaned_text).strip()  # Replace multiple spaces with a single space

    # Count words
    word_count = len(cleaned_text.split())

    # Check if the file already exists to avoid repetition
    filename = f'informacion/sitioweb{i+1}.txt'
    if not os.path.exists(filename):
        # Save the content in a file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        print(f"El documento {i+1} tiene {word_count} palabras.")

