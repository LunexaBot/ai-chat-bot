import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import numpy as np

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def scrape_text(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    texts = [p.get_text() for p in soup.find_all('p')]
    return "\n\n".join(texts)

def chunk_text(text, max_words=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    return chunks

def embed_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        resp = openai.embeddings.create(model="text-embedding-ada-002", input=chunk)
        embeddings.append(resp.data[0].embedding)  # <-- Fixed here
    return np.array(embeddings)

def build_index(url):
    text = scrape_text(url)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    return chunks, embeddings
