import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_generate_endpoint():
    # Test the basic functionality of the generate endpoint
    response = client.post("/generate", json={"prompt": "Once upon a time", "max_length": 50})
    assert response.status_code == 200
    assert "generated_text" in response.json()

@pytest.mark.parametrize("num_requests", [1, 5, 10])
def test_concurrent_requests(num_requests):
    # Test the service's ability to handle concurrent requests
    import asyncio
    import aiohttp

    async def make_request():
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:8000/generate", json={"prompt": "Test prompt", "max_length": 20}) as response:
                return await response.json()

    async def run_concurrent_requests():
        tasks = [make_request() for _ in range(num_requests)]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_concurrent_requests())
    assert len(results) == num_requests
    for result in results:
        assert "generated_text" in result

def test_model_performance():
    # Test the performance of the model
    from model import load_model, generate_text
    model = load_model()
    
    prompts = ["Once upon a time"] * 10
    max_lengths = [50] * 10
    
    start_time = time.time()
    results = generate_text(model, prompts, max_lengths)
    end_time = time.time()
    
    assert len(results) == 10
    print(f"Average generation time: {(end_time - start_time) / 10:.4f} seconds")