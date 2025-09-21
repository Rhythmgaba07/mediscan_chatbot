from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests, os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

class ChatRequest(BaseModel):
    question: str

@app.post("/chat/")
def chat(req: ChatRequest):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": req.question}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        return {"answer": answer.strip()}
    except Exception as e:
        return {"error": str(e)}
