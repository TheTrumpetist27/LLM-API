from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# command: uvicorn main:app --reload
# Go to http://127.0.0.1:8000/docs to go to swagger UI
app = FastAPI()

model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

class TextRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize_text(request: TextRequest):
    inputs = tokenizer.encode(request.text, return_tensors="pt", max_length=1024, truncation=True)
    #print(f"Aantal tokens: {len(inputs['input_ids'][0])}")
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=1.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)