import asyncio
import base64
import random
import time
import urllib.request

from comfyui import ComfyUIClient


def define_seed(seed):
    """
    Define the seed for random number generation.
    If the seed is -1, a new random seed is generated.
    Otherwise, the provided seed is used.
    """
    return random.randint(0, 2**32 - 1) if seed == -1 else seed

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

def fetch_image_as_base64(data):
    with urllib.request.urlopen(data["imageUrl"]) as response:
        return base64.b64encode(response.read())

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

class InactivityMonitor:
    """
    A simple inactivity monitor that tracks the last activity time
    """
    def __init__(self, comfyUiClient: ComfyUIClient, timeout=60, check_interval=5):
        """
        Initialize the inactivity monitor with a specified timeout.
        """
        self.timeout = timeout
        self.last_activity = time.time()
        self.check_interval = check_interval
        self.comfyUiClient = comfyUiClient
        self.last_activity_time = time.time()

    def reset(self):
        """
        Reset the last activity timestamp to the current time.
        """
        self.last_activity = time.time()
    
    async def inactivity_monitor(self):
        while True:
            await asyncio.sleep(self.check_interval)
            if (time.time() - self.last_activity_time) > self.timeout and not self.comfyUiClient.get_is_free():
                self.comfyUiClient.send_free()