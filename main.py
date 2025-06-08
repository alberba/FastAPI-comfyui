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

def queue_prompt(prompt):
    p = {"prompt": prompt}
    data = json.dumps(p).encode('utf-8')
    print(f"Sending to ComfyUI: {data.decode()}")
    
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
            print(f"ComfyUI raw response: {response_data.decode()}")
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

@app.post("/api/generate")
async def generate_image(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
            
        # Enviar el prompt a ComfyUI
        response = queue_prompt(prompt)
        return {"prompt_id": response["prompt_id"]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-simple")
async def generate_simple_image(request: Request):
    try:
        # Obtener y validar los datos de la petición
        data = await request.json()
        print(f"Received data: {data}")
        
        if not data or "prompt" not in data:
            raise HTTPException(status_code=400, detail="El campo 'prompt' es requerido")
        
        prompt = data["prompt"]
        print(f"Using prompt: {prompt}")

        # Generar un seed aleatorio
        random_seed = random.randint(0, 2**32 - 1)
        print(f"Generated seed: {random_seed}")

        # Crear workflow simple para ComfyUI
        workflow = {
            "5": {
                "inputs": {
                    "seed": random_seed,
                    "steps": 25,
                    "cfg": 1,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["35", 0],
                    "positive": ["29", 0],
                    "negative": ["20", 0],
                    "latent_image": ["6", 0]
                },
                "class_type": "KSampler"
            },
            "6": {
                "inputs": {
                    "width": 1024,
                    "height": 1024,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "7": {
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["9", 0]
                },
                "class_type": "VAEDecode"
            },
            "9": {
                "inputs": {
                    "vae_name": "ae.safetensors"
                },
                "class_type": "VAELoader"
            },
            "11": {
                "inputs": {
                    "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
                    "clip_name2": "clip_l.safetensors",
                    "type": "flux",
                    "device": "default"
                },
                "class_type": "DualCLIPLoader"
            },
            "14": {
                "inputs": {
                    "text": prompt,
                    "clip": ["11", 0]
                },
                "class_type": "CLIPTextEncode"
            },
            "15": {
                "inputs": {
                    "unet_name": "flux1-dev.safetensors",
                    "weight_dtype": "fp8_e4m3fn"
                },
                "class_type": "UNETLoader"
            },
            "18": {
                "inputs": {
                    "lora_name": "mascaroPerfeccionadoFlux.safetensors",
                    "strength_model": 1.0000000000000002,
                    "model": ["15", 0]
                },
                "class_type": "LoraLoaderModelOnly"
            },
            "20": {
                "inputs": {
                    "text": "",
                    "clip": ["11", 0]
                },
                "class_type": "CLIPTextEncode"
            },
            "27": {
                "inputs": {
                    "filename_prefix": "LoRA_Flux_no_inpaint",
                    "images": ["7", 0]
                },
                "class_type": "SaveImage"
            },
            "29": {
                "inputs": {
                    "guidance": 3.5,
                    "conditioning": ["14", 0]
                },
                "class_type": "FluxGuidance"
            },
            "35": {
                "inputs": {
                    "lora_name": "aidmaImageUprader-FLUX-v0.3.safetensors",
                    "strength_model": 1.0000000000000002,
                    "model": ["18", 0]
                },
                "class_type": "LoraLoaderModelOnly"
            }
        }
        print(f"Sending workflow to ComfyUI: {json.dumps(workflow, indent=2)}")

        # Enviar a ComfyUI
        try:
            response = queue_prompt(workflow)
            print(f"ComfyUI response: {response}")
            prompt_id = response["prompt_id"]
            print(f"Got prompt_id: {prompt_id}")
        except Exception as e:
            print(f"Error in queue_prompt: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")

        # Esperar a que la generación termine
        print("Waiting for generation to complete...")
        while True:
            try:
                history = get_history(prompt_id)
                print(f"History response: {history}")
                if prompt_id in history:
                    break
            except Exception as e:
                print(f"Error getting history: {str(e)}")
            await asyncio.sleep(1)

        # Obtener la imagen generada
        print("Getting generated image...")
        history_data = get_history(prompt_id)
        print(f"Final history data: {history_data}")
        
        if prompt_id in history_data:
            outputs = history_data[prompt_id]["outputs"]
            if "27" in outputs:  # ID del nodo SaveImage
                images = outputs["27"]["images"]
                if images:
                    image_data = images[0]
                    image_url = f"http://{COMFYUI_SERVER}/view?filename={image_data['filename']}&type={image_data['type']}&subfolder={image_data.get('subfolder', '')}"
                    print(f"Generated image URL: {image_url}")
                    return {"imageUrl": image_url}

        raise HTTPException(status_code=500, detail="No se pudo obtener la imagen generada")

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/progress/{prompt_id}")
async def get_progress(prompt_id: str):
    try:
        # Obtener el progreso actual
        progress = requests.get(f"http://{COMFYUI_SERVER}/api/progress")
        if progress.status_code == 200:
            progress_data = progress.json()
            return {
                "status": "in_progress",
                "progress": progress_data.get("value", 0),
                "eta": progress_data.get("eta", 0)
            }
        
        # Si no hay progreso, verificar si la generación terminó
        history = requests.get(f"http://{COMFYUI_SERVER}/api/history/{prompt_id}")
        if history.status_code == 200 and history.json():
            return {"status": "completed"}
        
        return {"status": "pending"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{prompt_id}")
async def get_status(prompt_id: str):
    try:
        response = requests.get(f"http://{COMFYUI_SERVER}/api/history/{prompt_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
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

@app.get("/api/queue")
async def get_queue():
    try:
        response = requests.get(f"http://{COMFYUI_SERVER}/api/queue")
        if response.status_code == 200:
            return response.json()
        raise HTTPException(status_code=response.status_code, detail="Error al obtener la cola")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test")
async def test():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 