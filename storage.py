import os
import numpy as np
from dotenv import load_dotenv, find_dotenv
import openai
from chromadb.api.types import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings


load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = os.environ['OPENAI_API_KEY']
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the directory where the vector files will be saved
vector_directory = 'vectores/'

# Create the directory if it does not exist
if not os.path.exists(vector_directory):
    os.makedirs(vector_directory)

# Define the directory where the chunked text files are located
chunks_directory = 'fragmentos/'

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Go through each file in the chunks directory
for file in os.listdir(chunks_directory):
    if file.endswith('.txt'):
        # Check if vector files for this chunked text file already exist
        vector_files_exist = any(fname.startswith(file[:-4]) and fname.endswith('.npy') for fname in os.listdir(vector_directory))
        
        # If vector files exist, skip this chunked text file
        if vector_files_exist:
            print(f"Vector files for '{file}' already exist, skipping...")
            continue

        text = None
        try:
            with open(os.path.join(chunks_directory, file), 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(os.path.join(chunks_directory, file), 'r', encoding='latin-1') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error decoding file {file} with UTF-8 and latin-1. Skipping... Error: {e}")
                continue

        if text:
            # Get embeddings for the chunk of text
            embedding = embeddings.embed_query(text)
            
            # Save the embedding in a new file
            embedding_name = f"{file[:-4]}_embedding.npy"
            np.save(os.path.join(vector_directory, embedding_name), embedding)
