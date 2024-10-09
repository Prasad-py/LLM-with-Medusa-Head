from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List
import asyncio

from model import load_model, generate_text
from batching import RequestQueue

app = FastAPI()
model = load_model()
request_queue = RequestQueue()

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100

class GenerationResponse(BaseModel):
    generated_text: str

@app.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest, background_tasks: BackgroundTasks):
    # Create a task to add the request to the queue
    task = asyncio.create_task(request_queue.add_request(request))
    # Add the queue processing task to background tasks
    background_tasks.add_task(process_queue)
    # Wait for the generation to complete
    generated_text = await task
    return GenerationResponse(generated_text=generated_text)

async def process_queue():
    while True:
        # Get a batch of requests from the queue
        batch = await request_queue.get_batch()
        if not batch:
            # If no requests, sleep briefly and continue
            await asyncio.sleep(0.1)
            continue
        
        # Extract prompts and max lengths from the batch
        prompts = [req.prompt for req in batch]
        max_lengths = [req.max_length for req in batch]
        
        # Generate text for the entire batch
        results = generate_text(model, prompts, max_lengths)
        
        # Set the results for each request
        for req, result in zip(batch, results):
            req.set_result(result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)