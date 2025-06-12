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
import websockets
from workflows.default_workflow import create_default_workflow
from workflows.lora_workflow import create_lora_workflow

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

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"http://{COMFYUI_SERVER}/view?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen(f"http://{COMFYUI_SERVER}/history/{prompt_id}") as response:
        return json.loads(response.read())

@app.websocket("/ws")
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
        async with websockets.connect(comfyui_ws_url) as comfyui_ws:
            print(f"Conectado al WebSocket de ComfyUI")
            
            # Tarea para reenviar mensajes de ComfyUI al cliente
            async def forward_comfyui_messages():
                try:
                    while True:
                        message = await comfyui_ws.recv()
                        await websocket.send_text(message)
                except Exception as e:
                    print(f"Error reenviando mensajes de ComfyUI: {str(e)}")
            
            # Tarea para reenviar mensajes del cliente a ComfyUI
            async def forward_client_messages():
                try:
                    while True:
                        message = await websocket.receive_text()
                        await comfyui_ws.send(message)
                except Exception as e:
                    print(f"Error reenviando mensajes del cliente: {str(e)}")
            
            # Ejecutar ambas tareas concurrentemente
            await asyncio.gather(
                forward_comfyui_messages(),
                forward_client_messages()
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
    seed: Optional[int] = None
    cfg: Optional[float] = 1.0
    steps: Optional[int] = 25
    number: Optional[int] = 1

class ImageWithMaskRequest(BaseModel):
    prompt: str
    seed: Optional[int] = None
    cfg: Optional[float] = 1.0
    steps: Optional[int] = 25
    number: Optional[int] = 1
    mask: Optional[str] = None
    image: Optional[str] = None

async def get_prompt(request: Request) -> str:
    data = await request.json()
    if not data or "prompt" not in data:
        raise HTTPException(status_code=400, detail="El campo 'prompt' es requerido")
    prompt = data["prompt"]
    return prompt

async def wait_for_generation(prompt_id: str):
    while True:
        try:
            history = get_history(prompt_id)
            if prompt_id in history:
                break
        except Exception as e:
            print(f"Error getting history: {str(e)}")
        await asyncio.sleep(1)

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

@app.post("/api/generate-simple")
async def generate_simple_image(request: ImageRequest):
    try:
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        workflow = create_default_workflow(request.prompt, seed, cfg=request.cfg, steps=request.steps)

        # Enviar a ComfyUI
        try:
            response = queue_prompt(workflow, request.number)
            prompt_id = response["prompt_id"]
        except Exception as e:
            print(f"Error in queue_prompt: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")

        await wait_for_generation(prompt_id)
        image_data = prepare_image_url(prompt_id, "27")
        image_data["seed"] = seed
        return image_data

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-mask")
async def generate_mask_image(request: ImageWithMaskRequest):
    try:
        seed = request.seed if request.seed is not None else random.randint(0, 2**32 - 1)
        workflow = create_lora_workflow(request.prompt, seed, cfg=request.cfg, steps=request.steps)

        # Subir imagen y máscara a ComfyUI antes de enviar el workflow
        import base64
        from io import BytesIO
        files_uploaded = []
        for img_type in ["image", "mask"]:
            img_b64: str = getattr(request, img_type, None)
            print(img_b64)
            if img_b64:
                if img_b64.startswith(("http://", "https://")):
                    print("hola")
                    # Descarga el recurso remoto como bytes
                    resp = requests.get(img_b64, timeout=10)
                    if resp.status_code != 200:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Error al descargar la URL para {img_type}: {resp.status_code}"
                        )
                    img_bytes = resp.content
                else:
                    # a) Limpia prefijo y padding
                    img_b64 = img_b64.split(",", 1)[-1]  # si viniera con data:image/… se descarta
                    img_b64 = "".join(img_b64.split())
                    padding = len(img_b64) % 4
                    if padding:
                        img_b64 += "=" * (4 - padding)

                    try:
                        img_bytes = base64.b64decode(img_b64)
                    except:
                        raise HTTPException(status_code=400, detail=f"Base64 inválido para {img_type}: {e}")
                
                
                # c) Prepara multipart/form-data
                upload_url = f"http://{COMFYUI_SERVER}/upload/image"
                files = {
                    "image": (f"{img_type}.png", BytesIO(img_bytes), "image/png")
                }
                data = {
                    "type": "input",
                    "overwrite": "true",
                    "subfolder": ""
                }
                # d) Llama a ComfyUI
                resp = requests.post(upload_url, data=data, files=files)
                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error subiendo {img_type}: {resp.status_code} – {resp.text}"
                    )

                files_uploaded.append(resp.json())

        # Enviar a ComfyUI
        try:
            response = queue_prompt(workflow, request.number)            
            prompt_id = response["prompt_id"]
        except Exception as e:
            print(f"Error in queue_prompt: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")

        await wait_for_generation(prompt_id)
        image_data = prepare_image_url(prompt_id, "22")
        image_data["seed"] = seed
        return image_data

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history_endpoint(max_items: int = 64):
    try:
        response = requests.get(f"http://{COMFYUI_SERVER}/api/history")
        if response.status_code == 200:
            history = response.json()
            # Limitar el número de items si es necesario
            if max_items and len(history) > max_items:
                history = dict(list(history.items())[-max_items:])
            return history
        raise HTTPException(status_code=response.status_code, detail="Error al obtener el historial")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 