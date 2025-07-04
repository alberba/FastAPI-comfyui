from ast import Str
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import requests
import json
import os
from typing import Optional, Dict, Any, Union
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
from workflows.face_workflow import create_face_workflow
import base64
from io import BytesIO
import time
from contextlib import asynccontextmanager
from websockets.exceptions import ConnectionClosedOK
import websockets
from starlette.datastructures import UploadFile as StarletteUploadFile
from PIL import Image

# ComfyUI server configuration
COMFYUI_SERVER = "127.0.0.1:8188"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

cambioworkflow = False

# Almacenar las conexiones WebSocket activas
active_connections = {}

# Global variable to track the last activity time
last_activity_time = time.time()
comfyui_is_free = False # New global variable to track if ComfyUI is free

def send_free_to_comfyui():
    """Sends a POST request to the /free endpoint on the ComfyUI server."""
    try:
        payload = {
            "unload_models": True,
            "free_memory": True
        }
        response = requests.post(f"http://{COMFYUI_SERVER}/free", json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        print("Sent /free endpoint to ComfyUI successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error sending /free to ComfyUI: {e}")

INACTIVITY_TIMEOUT = 1800  # seconds
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "*"],  # In production, replace with your Astro app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def activity_middleware(request: Request, call_next):
    global last_activity_time
    global comfyui_is_free # Declare global to modify it
    print("Origin", request.headers.get("origin"))
    last_activity_time = time.time()
    comfyui_is_free = False # Reset to False on any activity
    response = await call_next(request)
    return response

def queue_prompt(prompt):
    p = {"prompt": prompt}
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
    
async def forward_client_to_comfyui(client_ws: WebSocket, comfyui_ws):
    try:
        while True:
            data = await client_ws.receive_text()
            await comfyui_ws.send(data)
    except ConnectionClosedOK:
        pass
    except Exception as e:
        print(f"Error forwarding from client to ComfyUI: {str(e)}")

async def forward_comfyui_to_client(comfyui_ws, client_ws: WebSocket):
    try:
        async for data in comfyui_ws:
            await client_ws.send_text(data)
    except ConnectionClosedOK:
        pass
    except Exception as e:
        print(f"Error forwarding from ComfyUI to client: {str(e)}")

@app.websocket("/lorasuib/api/ws/")
async def websocket_endpoint(websocket: WebSocket):
    try:
        print("Nueva conexión WebSocket intentando conectarse...")
        await websocket.accept()
        print("Conexión WebSocket aceptada")
        
        # Generar un ID único para esta conexión
        client_id = websocket.query_params.get("clientId", str(uuid.uuid4()))
        print(f"Conexión WebSocket establecida con ID: {client_id}")
        
        # Conectar al WebSocket de ComfyUI
        comfyui_ws_url = f"ws://{COMFYUI_SERVER}/ws?clientId={client_id}"
        async with websockets.connect(comfyui_ws_url) as comfyui_ws:
            print(f"Conectado al WebSocket de ComfyUI")

            # Ejecutar ambas tareas concurrentemente
            await asyncio.gather(
                forward_client_to_comfyui(websocket, comfyui_ws),
                forward_comfyui_to_client(comfyui_ws, websocket)
            )
            
    except Exception as e:
        print(f"Error en la conexión WebSocket: {str(e)}")
        try:
            await websocket.close()
        except:
            pass

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

async def _upload_image_to_comfyui(img_data: Union[str, UploadFile], img_type: str, img_name: str) -> dict:
    if isinstance(img_data, (UploadFile, StarletteUploadFile)):
        img_bytes = await img_data.read()
    elif isinstance(img_data, str):
        if not img_data:
            return {}
        if img_data.startswith(("http://", "https://")):
            resp = requests.get(img_data, timeout=10)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Error al descargar la URL para {img_type}: {resp.status_code}"
                )
            img_bytes = resp.content
        else:
            img_b64 = img_data.split(",", 1)[-1]
            img_b64 = "".join(img_b64.split())
            padding = len(img_b64) % 4
            if padding:
                img_b64 += "=" * (4 - padding)
            try:
                img_bytes = base64.b64decode(img_b64)
            except binascii.Error as e:
                raise HTTPException(status_code=400, detail=f"Base64 inválido para {img_type}: {e}")
    
    # Resize image to specified width and height
    try:
        img_buffer = BytesIO(img_bytes)
        img = Image.open(img_buffer)

        
        # Ajustar width y height para que sean divisibles entre 2
        # El width será el tamaño actual de la imagen, ajustando solo el height para que sea divisible entre 2
        new_width = img.width - (img.width % 2)
        new_height = img.height - (img.height % 2)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        output_buffer = BytesIO()
        resized_img.save(output_buffer, format="PNG")
        img_bytes = output_buffer.getvalue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen para {img_type}: {e}")

    upload_url = f"http://{COMFYUI_SERVER}/upload/image"
    files = {
        "image": (img_name, BytesIO(img_bytes), "image/png")
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
            detail=f"Error subiendo {img_type}: {resp.status_code} - {resp.text}"
        )
    return resp.json()

async def _queue_and_wait_for_completion(workflow: dict, save_image_node: str) -> dict:
    try:
        response = queue_prompt(workflow)
        prompt_id = response["prompt_id"]
    except Exception as e:
        print(f"Error in queue_prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")

    await wait_for_generation(prompt_id)
    image_data = prepare_image_url(prompt_id, save_image_node)
    return image_data

async def _process_comfyui_generation(workflow: dict, save_image_node: str) -> dict:
    try:
        return await _queue_and_wait_for_completion(workflow, save_image_node)
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
    denoise: float = 1.00
    lora: str = ""

@app.post("/lorasuib/api/generate-simple")
async def generate_simple_image(request: ImageRequest):
    try:
        seed = random.randint(0, 2**32 - 1) if request.seed == -1 else request.seed
        workflow = create_default_workflow(request.prompt, seed, cfg=request.cfg, steps=request.steps, lora=request.lora, width=request.width, height=request.height, denoise=request.denoise)

        # Enviar a ComfyUI
        try:
            image_data = await _process_comfyui_generation(workflow, "27")
        except Exception as e:
            print(f"Error in _process_comfyui_generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")
        
        global last_activity_time
        last_activity_time = time.time()
        return prepare_response(image_data, seed)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lorasuib/api/generate-mask")
async def generate_mask_image(
    prompt: str = Form(...),
    width: int = Form(...),
    height: int = Form(...),
    seed: int = Form(...),
    cfg: float = Form(...),
    steps: int = Form(...),
    denoise: float = Form(0.9),
    lora: str = Form(""),
    image: UploadFile = File(),
    mask: UploadFile = File()
):
    try:
        seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
        global cambioworkflow
        cambioworkflow = not cambioworkflow

        workflow = create_lora_workflow(prompt, seed, width, height, denoise, lora, "imagen1.png" if cambioworkflow else "imagen2.png", "mask1.png" if cambioworkflow else "mask2.png", cfg, steps)

        # Subir imagen y máscara a ComfyUI antes de enviar el workflow
        files_uploaded = []
        files_uploaded.append(await _upload_image_to_comfyui(image, "image", "imagen1.png" if cambioworkflow else "imagen2.png"))
        files_uploaded.append(await _upload_image_to_comfyui(mask, "mask", "mask1.png" if cambioworkflow else "mask2.png"))

        print(f"DEBUG: Uploaded files: {files_uploaded}")

        # Enviar a ComfyUI
        try:
            image_data = await _process_comfyui_generation(workflow, "22")
        except Exception as e:
            print(f"Error in _process_comfyui_generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")

        global last_activity_time
        last_activity_time = time.time()
        return prepare_response(image_data, seed)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/lorasuib/api/face-enhancer")
async def generate_enhancer(
    prompt: str = File(...),
    width: int = File(1024),
    height: int = File(1024),
    seed: int = File(-1),
    cfg: float = File(1.0),
    steps: int = File(25),
    denoise: float = Form(0.9),
    lora: str = File(...),
    maskWidth: str = File(512),
    maskHeight: str = File(512),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
):
    try:
        seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
        global cambioworkflow
        cambioworkflow = not cambioworkflow
        maskmax = max(int(maskWidth), int(maskHeight))
        extend_factor = 3.0
        if maskmax < 1024:
            upscaler = 1024 / (maskmax * extend_factor)
            if upscaler > 4.0:
                upscaler = 4.0
            elif upscaler < 1.0:
                upscaler = 1.0
        

        workflow = create_face_workflow(prompt, seed, width, height, denoise, lora, upscaler, extend_factor, "imagen1.png" if cambioworkflow else "imagen2.png", "mask1.png" if cambioworkflow else "mask2.png", cfg, steps)

        # Subir imagen y máscara a ComfyUI antes de enviar el workflow
        files_uploaded = []
        files_uploaded.append(await _upload_image_to_comfyui(image, "image", "imagen1.png" if cambioworkflow else "imagen2.png"))
        files_uploaded.append(await _upload_image_to_comfyui(mask, "mask", "mask1.png" if cambioworkflow else "mask2.png"))

        # Enviar a ComfyUI
        try:
            image_data = await _process_comfyui_generation(workflow, "60")
        except Exception as e:
            print(f"Error in _process_comfyui_generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")

        global last_activity_time
        last_activity_time = time.time()
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