import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import numpy as np

# Initialize OpenAI client with API key from environment variables
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def scrape_text(url):
    """
    Scrapes text content from a given URL, specifically looking for <p> tags.
    Logs the length of the scraped text and a snippet if it's short.
    """
    print(f"--- SCRAPER LOG: Attempting to scrape URL: {url} ---")
    try:
        resp = requests.get(url)
        resp.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(resp.text, "html.parser")
        texts = [p.get_text() for p in soup.find_all('p')]
        full_text = "\n\n".join(texts)
        
        print(f"--- SCRAPER LOG: Scraped text length: {len(full_text)} characters ---")
        if len(full_text) < 100: # Print a snippet if the text is very short
            print(f"--- SCRAPER LOG: Scraped text snippet (first 200 chars): {full_text[:200]} ---")
        elif len(full_text) > 0: # Print a confirmation if text is found
            print("--- SCRAPER LOG: Successfully scraped substantial text. ---")
        else:
            print("--- SCRAPER LOG: Scraped text is empty. No <p> tags found or content within them. ---")
            
        return full_text
    except requests.exceptions.RequestException as e:
        print(f"--- SCRAPER LOG ERROR: Failed to fetch URL {url}: {e} ---")
        return "" # Return empty string on request failure
    except Exception as e:
        print(f"--- SCRAPER LOG ERROR: An unexpected error occurred during scraping: {e} ---")
        return "" # Return empty string on other errors

def chunk_text(text, max_words=200):
    """
    Splits the given text into smaller chunks of a maximum number of words.
    Logs the number of chunks created.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))
    
    print(f"--- SCRAPER LOG: Number of chunks created: {len(chunks)} ---")
    if len(chunks) == 0 and len(words) > 0:
        print("--- SCRAPER LOG: No chunks were created, but text was present. Check chunking logic. ---")
    elif len(chunks) == 0:
        print("--- SCRAPER LOG: No chunks were created because input text was empty. ---")
        
    return chunks

def embed_chunks(chunks):
    """
    Generates OpenAI embeddings for each text chunk.
    Logs the total number of embeddings generated.
    Handles cases where there are no chunks to embed.
    """
    embeddings = []
    if not chunks: # If the list of chunks is empty, return an empty NumPy array
        print("--- SCRAPER LOG: No chunks to embed. Returning an empty embeddings array. ---")
        return np.array([]) 
    
    print(f"--- SCRAPER LOG: Starting embedding process for {len(chunks)} chunks ---")
    for i, chunk in enumerate(chunks):
        try:
            # Create embedding for the current chunk
            resp = openai.embeddings.create(model="text-embedding-ada-002", input=chunk)
            embeddings.append(resp.data[0].embedding)
            # print(f"--- SCRAPER LOG: Embedded chunk {i+1}/{len(chunks)} successfully. ---") # Uncomment for very verbose logging
        except Exception as e:
            print(f"--- SCRAPER LOG ERROR: Failed to embed chunk {i+1} (length {len(chunk)}): {e} ---")
            # If an embedding fails, you might want to append a list of zeros or skip it
            # For now, we'll just log and continue, which means 'embeddings' might be shorter
            # than 'chunks' if some fail.
            embeddings.append(np.zeros(1536).tolist()) # Append a zero vector as a placeholder
            
    print(f"--- SCRAPER LOG: Total embeddings generated: {len(embeddings)} ---")
    # Ensure the return is a NumPy array, even if empty
    return np.array(embeddings)

def build_index(url):
    """
    Orchestrates the scraping, chunking, and embedding process for a given URL.
    Logs the start and end of the indexing process.
    """
    print(f"--- SCRAPER LOG: Starting build_index for URL: {url} ---")
    text = scrape_text(url)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    print(f"--- SCRAPER LOG: Finished build_index. Chunks found: {len(chunks)}, Embeddings generated: {len(embeddings)} ---")
    return chunks, embeddings
