# fserver.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path where your model is stored in mounted Google Drive
MODEL_PATH = "/content/drive/MyDrive/gemma2b"

print("Loading model and tokenizer from Drive...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()
print("Model loaded!")

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_text(request: PromptRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}
