import random
import time

def define_seed(seed):
    """
    Define the seed for random number generation.
    If the seed is -1, a new random seed is generated.
    Otherwise, the provided seed is used.
    """
    return random.randint(0, 2**32 - 1) if seed == -1 else seed

def get_actual_time():
    """
    Get the current time in seconds since the epoch.
    """
    return time.time()

def is_data_url(data):
    """
    Check if the provided data is a valid URL
    """
    return data.startswith(("http://", "https://"))

def get_image_bytes_from_url(url):
    """
    Fetch image bytes from a URL.
    """
    import requests
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise ValueError(f"Failed to fetch image from {url}")
    return resp.content

def remove_b64_header(data):
    """
    Remove the base64 header from a data URL.
    """
    if data.startswith("data:image/"):
        img_b64 = data.split(",", 1)[-1]
        img_b64 = "".join(img_b64.split())
        padding = len(img_b64) % 4
        if padding:
            img_b64 += "=" * (4 - padding)
        return img_b64
    return data