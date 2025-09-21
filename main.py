from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# CORS for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

@app.post("/chat/")
def chat(question: str = Form(...)):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",  # Or "mixtral-8x7b-32768" or "gemma-7b-it"
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": question}
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

