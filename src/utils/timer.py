import time
from functools import wraps

def time_it(func):
    """Decorator để đo thời gian chạy của một hàm"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"⏱️ Hàm [{func.__name__}] chạy mất: {end - start:.2f} giây")
        return result
    return wrapper
