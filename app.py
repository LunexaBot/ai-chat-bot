from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scraper import build_index
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

app = FastAPI()

# ✅ Allow requests from multiple domains — adjust later for specific clients
origins = [
    "https://pennytoleman.wixsite.com",
    "https://pennytoleman-wixsite-com.filesusr.com",
    "*"  # 🧪 for testing across other client sites (you can restrict this in production)
]

# ✅ Apply CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve static files like your widget.html
app.mount("/static", StaticFiles(directory="."), name="static")

# ✅ OpenAI setup
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Memory: store each client’s data under their site_id
memory = {}

# ✅ Models
class IndexRequest(BaseModel):
    url: str
    site_id: str

class QueryRequest(BaseModel):
    question: str
    site_id: str

# ✅ Indexing endpoint
@app.post("/index")
async def index_website(req: IndexRequest):
    chunks, embeddings = build_index(req.url)
    memory[req.site_id] = {
        "chunks": chunks,
        "embeddings": embeddings
    }
    return {"status": "indexed", "chunks": len(chunks), "site_id": req.site_id}

# ✅ Querying endpoint
@app.post("/query")
async def query_website(req: QueryRequest):
    if req.site_id not in memory:
        return {"answer": "❌ This site has not been indexed yet. Please contact the website owner."}

    chunks = memory[req.site_id]['chunks']
    embeddings = memory[req.site_id]['embeddings']

    # --- AMENDMENT START ---
    # Fix the ValueError: The truth value of an empty array is ambiguous
    # Check if chunks list is empty OR if embeddings numpy array is empty
    if not chunks or embeddings.size == 0:
        return {"answer": "Sorry, the site has not been indexed properly yet, or it contains no relevant text to answer your question."}
    # --- AMENDMENT END ---

    q = req.question
    emb = openai.embeddings.create(model="text-embedding-ada-002", input=q).data[0].embedding

    sims = cosine_similarity([emb], embeddings)[0]
    top_ix = np.argsort(sims)[-3:]
    context = "\n\n".join([chunks[i] for i in reversed(top_ix)])
    prompt = f"You are a helpful assistant. Use the context to answer:\n\nContext:\n{context}\n\nQuestion: {q}"
    resp = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return {"answer": resp.choices[0].message.content}

# ✅ Health check
@app.get("/")
async def root():
    return {"message": "✅ AI Chatbot API is running. POST to /query or /index with site_id to interact."}
