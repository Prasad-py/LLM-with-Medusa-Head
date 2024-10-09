# FastAPI LLM Service with Medusa Head

Hey there! This is my implementation of the FastAPI service for an optimized LLM with a Medusa head. It was a pretty challenging assignment, but I learned a lot. Here's a quick rundown of what I did:

## What I implemented

1. **Model Optimization**: Used llama.cpp to speed up the vicuna-7b model. It was tricky to set up, but it really improved inference speed.

2. **Medusa Head**: Implemented this from scratch. It does speculative decoding, which is pretty cool. Basically, it tries to predict multiple tokens at once.

3. **Dynamic Batching**: Created a queue system to handle multiple requests at the same time. This was probably the trickiest part to get right.

4. **FastAPI Service**: Set up endpoints to actually serve the model. FastAPI made this part easier than I expected.

## How to run it

1. Install stuff:
   ```
   pip install -r requirements.txt
   ```

2. You'll need to convert the model to GGML format for llama.cpp. I used some scripts I found online for this.

3. Update the model path in `model.py`.

4. Run the server:
   ```
   python main.py
   ```

5. Test it:
   ```
   pytest test_service.py
   ```

## What I learned

- Working with llama.cpp was new to me, but it's really powerful for optimizing these large models.
- Implementing speculative decoding taught me a lot about how these language models actually work.
- Asynchronous programming in Python is pretty cool, especially for handling concurrent requests.

## Challenges

The biggest headache was probably getting the dynamic batching to work smoothly with the Medusa head. Also, balancing speed and accuracy took some trial and error.

## What I'd improve

If I had more time, I'd probably try to optimize the Medusa head implementation further and maybe add some more advanced batching strategies like Adaptive batch sizing or token-level batching, instead of batching full requests, batch at the token level for even more efficient processing.

Overall, it was a fun project! Let me know if you have any questions.