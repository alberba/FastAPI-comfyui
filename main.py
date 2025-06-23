from ast import Str
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import requests
import json
import os
from typing import Optional
from pydantic import BaseModel
import asyncio
import random
import uuid
import urllib.request
import urllib.parse
import urllib.error
import binascii
from workflows.default_workflow import create_default_workflow
from workflows.lora_workflow import create_lora_workflow
import base64
from io import BytesIO
import time
from contextlib import asynccontextmanager
from websockets.exceptions import ConnectionClosedOK
from websockets.client import connect

app = FastAPI(title="ComfyUI Image Generation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Astro app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ComfyUI server configuration
COMFYUI_SERVER = "127.0.0.1:8188"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Almacenar las conexiones WebSocket activas
active_connections = {}

# Global variable to track the last activity time
last_activity_time = time.time()
comfyui_is_free = False # New global variable to track if ComfyUI is free

def send_free_to_comfyui():
    """Sends a POST request to the /free endpoint on the ComfyUI server."""
    try:
        response = requests.post(f"http://{COMFYUI_SERVER}/free")
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        print("Sent /free endpoint to ComfyUI successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error sending /free to ComfyUI: {e}")

INACTIVITY_TIMEOUT = 20  # seconds
CHECK_INTERVAL = 5 # seconds

async def inactivity_monitor():
    global comfyui_is_free
    while True:
        await asyncio.sleep(CHECK_INTERVAL)
        global last_activity_time
        if (time.time() - last_activity_time) > INACTIVITY_TIMEOUT and not comfyui_is_free:
            send_free_to_comfyui()
            comfyui_is_free = True # Set to True after sending the signal

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting inactivity monitor...")
    asyncio.create_task(inactivity_monitor())
    yield
    # Shutdown (if needed, though not for this specific task)
    print("Inactivity monitor shutting down.")

app = FastAPI(title="ComfyUI Image Generation API", lifespan=lifespan)

@app.middleware("http")
async def activity_middleware(request: Request, call_next):
    global last_activity_time
    global comfyui_is_free # Declare global to modify it
    last_activity_time = time.time()
    comfyui_is_free = False # Reset to False on any activity
    response = await call_next(request)
    return response

def queue_prompt(prompt, number):
    p = {"prompt": prompt}
    p["number"] = number
    data = json.dumps(p).encode('utf-8')
    
    req = urllib.request.Request(
        f"http://{COMFYUI_SERVER}/prompt",
        data=data,
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
    )
    try:
        with urllib.request.urlopen(req) as response:
            response_data = response.read()
            return json.loads(response_data)
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        error_body = e.read().decode()
        print(f"Error response: {error_body}")
        raise HTTPException(status_code=500, detail=f"Error al comunicarse con ComfyUI: {error_body}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")
    
def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{COMFYUI_SERVER}/history/{prompt_id}") as response:
        return json.loads(response.read())

@app.websocket("/lorasuib/ws/")
async def websocket_endpoint(websocket: WebSocket):
    try:
        print("Nueva conexión WebSocket intentando conectarse...")
        await websocket.accept()
        print("Conexión WebSocket aceptada")
        
        # Generar un ID único para esta conexión
        client_id = str(uuid.uuid4())
        print(f"Conexión WebSocket establecida con ID: {client_id}")
        
        # Conectar al WebSocket de ComfyUI
        comfyui_ws_url = f"ws://{COMFYUI_SERVER}/ws?clientId={client_id}"
        async with connect(comfyui_ws_url) as comfyui_ws:
            print(f"Conectado al WebSocket de ComfyUI")
            
            # Tarea para reenviar mensajes de ComfyUI al cliente
            async def forward_messages(source_ws, destination_ws):
                try:
                    async for message in source_ws:
                        await destination_ws.send(message)
                except ConnectionClosedOK:
                    pass
                except Exception as e:
                    print(f"Error reenviando mensajes: {str(e)}")

            # Ejecutar ambas tareas concurrentemente
            await asyncio.gather(
                forward_messages(comfyui_ws, websocket),
                forward_messages(websocket, comfyui_ws)
            )
            
    except Exception as e:
        print(f"Error en la conexión WebSocket: {str(e)}")
        try:
            await websocket.close()
        except:
            pass

async def send_progress(client_id: str, message: dict):
    if client_id in active_connections:
        try:
            await active_connections[client_id].send_json(message)
        except Exception as e:
            print(f"Error enviando mensaje a {client_id}: {str(e)}")
            if client_id in active_connections:
                del active_connections[client_id]

def prepare_image_url(prompt_id: str, saveImageNode: str) -> dict:
    history_data = get_history(prompt_id)
    if prompt_id in history_data:
        outputs = history_data[prompt_id]["outputs"]
        if saveImageNode in outputs:  # ID del nodo SaveImage
            images = outputs[saveImageNode]["images"]
            if images:
                image_data = images[0]
                image_url = f"http://{COMFYUI_SERVER}/view?filename={image_data['filename']}&type={image_data['type']}&subfolder={image_data.get('subfolder', '')}"
                print(f"Generated image URL: {image_url}")
                return {"imageUrl": image_url}

    raise HTTPException(status_code=500, detail="No se pudo obtener la imagen generada")

def fetch_image_as_base64(data):
    with urllib.request.urlopen(data["imageUrl"]) as response:
        return base64.b64encode(response.read())

def prepare_response(image_data, seed):
    image = fetch_image_as_base64(image_data)
    image_data["image"] = image
    image_data["seed"] = seed
    return image_data

async def _upload_image_to_comfyui(img_b64: Optional[str], img_type: str) -> dict:
    if not img_b64:
        return {}

    if img_b64.startswith(("http://", "https://")):
        resp = requests.get(img_b64, timeout=10)
        if resp.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Error al descargar la URL para {img_type}: {resp.status_code}"
            )
        img_bytes = resp.content
    else:
        img_b64 = img_b64.split(",", 1)[-1]
        img_b64 = "".join(img_b64.split())
        padding = len(img_b64) % 4
        if padding:
            img_b64 += "=" * (4 - padding)
        try:
            img_bytes = base64.b64decode(img_b64)
        except binascii.Error as e:
            raise HTTPException(status_code=400, detail=f"Base64 inválido para {img_type}: {e}")
    
    upload_url = f"http://{COMFYUI_SERVER}/upload/image"
    files = {
        "image": (f"{img_type}.png", BytesIO(img_bytes), "image/png")
    }
    data = {
        "type": "input",
        "overwrite": "true",
        "subfolder": ""
    }
    resp = requests.post(upload_url, data=data, files=files)
    if resp.status_code != 200:
        raise HTTPException(
            status_code=500,
            detail=f"Error subiendo {img_type}: {resp.status_code} – {resp.text}"
        )
    return resp.json()

async def _queue_and_wait_for_completion(workflow: dict, number: int, save_image_node: str) -> dict:
    try:
        response = queue_prompt(workflow, number)
        prompt_id = response["prompt_id"]
    except Exception as e:
        print(f"Error in queue_prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")

    await wait_for_generation(prompt_id)
    image_data = prepare_image_url(prompt_id, save_image_node)
    return image_data

async def _process_comfyui_generation(workflow: dict, number: int, save_image_node: str) -> dict:
    try:
        return await _queue_and_wait_for_completion(workflow, number, save_image_node)
    except Exception as e:
        print(f"Error during ComfyUI generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")

async def wait_for_generation(prompt_id: str):
    while True:
        try:
            history = get_history(prompt_id)
            if prompt_id in history:
                break
        except Exception as e:
            print(f"Error getting history: {str(e)}")
        await asyncio.sleep(1)

async def get_prompt(request: Request) -> str:
    data = await request.json()
    if not data or "prompt" not in data:
        raise HTTPException(status_code=400, detail="El campo 'prompt' es requerido")
    prompt = data["prompt"]
    return prompt

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    seed: Optional[int] = None
    steps: Optional[int] = 20
    cfg: Optional[float] = 7.0

class SimpleGenerationRequest(BaseModel):
    prompt: str

class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int = -1
    cfg: float = 1.0
    steps: int = 25
    number: int = 1

class ImageWithMaskRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int = -1
    cfg: float = 1.0
    steps: int = 25
    number: int = 1
    mask: str
    image: str
    lora: str

@app.post("/lorasuib/api/generate-simple")
async def generate_simple_image(request: ImageRequest):
    try:
        seed = random.randint(0, 2**32 - 1) if request.seed == -1 else request.seed
        workflow = create_default_workflow(request.prompt, seed, cfg=request.cfg, steps=request.steps, width=request.width, height=request.height)

        # Enviar a ComfyUI
        try:
            image_data = await _process_comfyui_generation(workflow, request.number, "27")
        except Exception as e:
            print(f"Error in _process_comfyui_generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")
        
        return prepare_response(image_data, seed)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lorasuib/api/generate-mask")
async def generate_mask_image(request: ImageWithMaskRequest):
    try:
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)

        workflow = create_lora_workflow(request.prompt, seed, request.width, request.height, request.lora, request.cfg, request.steps)

        # Subir imagen y máscara a ComfyUI antes de enviar el workflow
        files_uploaded = []
        for img_type in ["image", "mask"]:
            img_b64: Optional[str] = getattr(request, img_type, None)
            files_uploaded.append(await _upload_image_to_comfyui(img_b64, img_type))

        # Enviar a ComfyUI
        try:
            image_data = await _process_comfyui_generation(workflow, request.number, "22")
        except Exception as e:
            print(f"Error in _process_comfyui_generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")

        return prepare_response(image_data, seed)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/lorasuib/api/get-loras")
async def get_loras():
    try:
        response = requests.get(f"http://{COMFYUI_SERVER}/models/loras")
        if response.status_code == 200:
            return response.json()
        raise HTTPException(status_code=response.status_code, detail="Error al obtener los modelos LORA")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 