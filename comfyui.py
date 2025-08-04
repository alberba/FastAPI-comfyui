import json
import urllib.error
import urllib.request
from io import BytesIO

import requests
from fastapi import HTTPException


class ComfyUIClient:

    def __init__(self, comfyui_server: str):
        self.comfyui_server = comfyui_server
        self.free_url = f"{comfyui_server}/free"
        self.prompt_url = f"{comfyui_server}/prompt"
        self.upload_url = f"{comfyui_server}/upload/image"
        self.history_url = f"{comfyui_server}/history/"
        self.is_free = False

    def send_free(self):
        """Sends a POST request to the /free endpoint on the ComfyUI server."""
        try:
            payload = {"unload_models": True, "free_memory": True}
            response = requests.post(self.free_url, json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            print("Sent /free endpoint to ComfyUI successfully.")
            self.is_free = True
        except requests.exceptions.RequestException as e:
            print(f"Error sending /free to ComfyUI: {e}")

    def set_non_free(self):
        """Sets the is_free flag to False."""
        self.is_free = False

    def post_image(self, filename: str, file_bytes: bytes, file_type: str) -> dict:
        files = {"image": (filename, BytesIO(file_bytes), "image/png")}
        data = {"type": "input", "overwrite": "true", "subfolder": ""}
        resp = requests.post(self.upload_url, data=data, files=files)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Error subiendo {file_type}: {resp.status_code} - {resp.text}",
            )
        return resp.json()

    def queue_prompt(self, prompt):
        p = {"prompt": prompt}
        data = json.dumps(p).encode("utf-8")

        req = urllib.request.Request(
            self.prompt_url,
            data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as response:
                response_data = response.read()
                return json.loads(response_data)
        except urllib.error.HTTPError as e:
            print(f"HTTP Error: {e.code} - {e.reason}")
            error_body = e.read().decode()
            print(f"Error response: {error_body}")
            raise HTTPException(
                status_code=500,
                detail=f"Error al comunicarse con ComfyUI: {error_body}",
            )
        except Exception as e:
            print(f"Error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")

    def get_history(self, prompt_id: str) -> dict:
        with urllib.request.urlopen(f"{self.history_url}{prompt_id}") as response:
            return json.loads(response.read())

    def get_is_free(self) -> bool:
        """
        Check if the ComfyUI client is free.
        """
        return self.is_free
