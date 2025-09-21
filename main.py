from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv
import logging

load_dotenv()

app = FastAPI()

# Logging
logging.basicConfig(level=logging.INFO)

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

class ChatRequest(BaseModel):
    question: str

@app.post("/chat/")
def chat_endpoint(request: ChatRequest):
    logging.info(f"Received question: {request.question}")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gemma-7b-it",
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": request.question}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Raise error if HTTP error
        data = response.json()
        logging.info(f"GROQ response: {data}")

        # Handle empty choices
        if "choices" in data and len(data["choices"]) > 0:
            answer = data["choices"][0]["message"]["content"]
            return {"answer": answer.strip()}
        else:
            return {"error": "No response from model."}

    except Exception as e:
        logging.error(f"Error calling GROQ API: {e}")
        return {"error": str(e)}

