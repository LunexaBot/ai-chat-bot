from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scraper import build_index
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

app = FastAPI()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# CORS settings
origins = ["*"]  # For testing; replace with specific domains for production

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    q = req.question
    emb = openai.embeddings.create(model="text-embedding-ada-002", input=q)['data'][0]['embedding']
    sims = cosine_similarity([emb], memory['embeddings'])[0]
    top_ix = np.argsort(sims)[-3:]
    context = "\n\n".join([memory['chunks'][i] for i in reversed(top_ix)])
    prompt = f"You are a helpful assistant. Use the context to answer:\n\nContext:\n{context}\n\nQuestion: {q}"
    resp = openai.chat.completions.create(model="gpt-4", messages=[{"role":"system","content":prompt}])
    return {"answer": resp['choices'][0]['message']['content']}

@app.get("/")
async def root():
    return {"message": "Welcome to the AI Chatbot API! Use POST /query to ask questions."}
