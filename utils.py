import time

def measure_latency(func):
    def wrapper(*args, **kwargs):
        # Record the start time
        start_time = time.time()
        # Call the original function
        result = func(*args, **kwargs)
        # Record the end time
        end_time = time.time()
        # Print the latency
        print(f"Latency: {end_time - start_time:.4f} seconds")
        return result
    return wrapper