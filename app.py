from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scraper import build_index
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

app = FastAPI()

# ✅ Add the exact origins you want to allow (Wix in this case)
origins = [
    "https://pennytoleman.wixsite.com",
    "https://pennytoleman-wixsite-com.filesusr.com"  # optional if using external widget loader
]

# ✅ Attach CORS middleware with those origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # restrict to only your sites
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (like widget.html) under /static
app.mount("/static", StaticFiles(directory="."), name="static")

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class IndexRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

memory = {}

@app.post("/index")
async def index_website(req: IndexRequest):
    chunks, embeddings = build_index(req.url)
    memory['chunks'] = chunks
    memory['embeddings'] = embeddings
    return {"status": "indexed", "chunks": len(chunks)}

@app.post("/query")
async def query_website(req: QueryRequest):
    if 'chunks' not in memory or 'embeddings' not in memory:
        return {"answer": "The website is not indexed yet. Please send a POST request to /index with the website URL first."}

    q = req.question
    emb = openai.embeddings.create(model="text-embedding-ada-002", input=q).data[0].embedding
    sims = cosine_similarity([emb], memory['embeddings'])[0]
    top_ix = np.argsort(sims)[-3:]
    context = "\n\n".join([memory['chunks'][i] for i in reversed(top_ix)])
    prompt = f"You are a helpful assistant. Use the context to answer:\n\nContext:\n{context}\n\nQuestion: {q}"
    resp = openai.chat.completions.create(model="gpt-4", messages=[{"role":"system","content":prompt}])
    return {"answer": resp.choices[0].message.content}

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Chatbot API! Use POST /query to ask questions."}
