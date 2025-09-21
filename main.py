from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import google.generativeai as genai
import uvicorn
import logging

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ----------------------------
# App initialization
# ----------------------------
app = FastAPI()
logging.basicConfig(level=logging.INFO)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Initialize model
model = genai.GenerativeModel("gemini-1.5-flash")

# ----------------------------
# Pydantic request model
# ----------------------------
class ChatRequest(BaseModel):
    question: str

# ----------------------------
# Helper function
# ----------------------------
def get_chat_response(question: str) -> str:
    """
    Calls the Generative AI model and returns response text.
    """
    try:
        response = model.generate_content([question])
        return response.text
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"Error: {str(e)}"

# ----------------------------
# Chat endpoint
# ----------------------------
@app.post("/chat/")
async def chat_endpoint(request: ChatRequest):
    try:
        logging.info(f"Received question: {request.question}")

        # Generate response
        answer = get_chat_response(
            f"You are a helpful medical assistant. Answer this question clearly: {request.question}"
        )

        logging.info(f"Answer: {answer}")

        return JSONResponse(content={"answer": answer})

    except Exception as e:
        logging.error(f"Error in /chat/: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


