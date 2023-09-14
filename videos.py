from pytube import YouTube
import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader


load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = os.environ['OPENAI_API_KEY']
openai.api_key = os.environ["OPENAI_API_KEY"]

# Directory to save downloaded audios and texts
saveDir = 'informacion/'

# Read the file containing the video URLs
with open("links_videos.txt", "r") as f:
    video_urls = f.readlines()

# Process each URL
for videoURL in video_urls:
    videoURL = videoURL.strip()  # Remove spaces and line breaks
    print(f"\nProcesando {videoURL}...")
    
    # Get the video title using pytube
    yt = YouTube(videoURL)
    video_title = yt.title
    print(f"Título del video: {video_title}")
    
    # Replace unwanted characters in the title to make it a valid filename
    valid_title = "".join(c for c in video_title if c.isalnum() or c.isspace()).rstrip()
    
    # Check if the file already exists
    if os.path.exists(f"{saveDir}/{valid_title}.txt"):
        print(f"El video '{valid_title}' ya ha sido procesado, pasando al siguiente.")
        continue
    
    # Load and process the audio from the video
    print("Iniciando la carga y procesamiento del audio...")
    loader = GenericLoader(YoutubeAudioLoader([videoURL], saveDir), OpenAIWhisperParser())
    docs = loader.load()
    
    # For debugging purposes: print the document content
    print("Contenido del documento:", docs[0].page_content)
    
    # Save the video content in a .txt file with the video title as filename
    with open(f"{saveDir}/{valid_title}.txt", "w") as f:
        f.write(docs[0].page_content)