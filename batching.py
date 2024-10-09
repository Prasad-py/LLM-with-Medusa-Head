import asyncio
from collections import deque

class RequestQueue:
    def __init__(self, max_batch_size=32):
        self.queue = asyncio.Queue()
        self.max_batch_size = max_batch_size

    async def add_request(self, request):
        # Create a future for this request
        future = asyncio.Future()
        # Add the request and its future to the queue
        await self.queue.put((request, future))
        # Return the future, which will be resolved when the result is ready
        return await future

    async def get_batch(self):
        if self.queue.empty():
            return None

        batch = []
        # Collect requests until we reach max_batch_size or the queue is empty
        while len(batch) < self.max_batch_size and not self.queue.empty():
            request, future = await self.queue.get()
            batch.append((request, future))

        return batch

class Request:
    def __init__(self, prompt, max_length):
        self.prompt = prompt
        self.max_length = max_length
        self.future = asyncio.Future()

    def set_result(self, result):
        # Resolve the future with the generation result
        self.future.set_result(result)