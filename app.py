from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

# Load model saat startup
@app.on_event("startup")
def load_model():
    app.state.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class TextPair(BaseModel):
    text1: str
    text2: str

@app.post("/similarity")
async def calculate_similarity(pair: TextPair):
    try:
        embeddings = app.state.model.encode([pair.text1, pair.text2])
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        return {"similarity": float(similarity)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
