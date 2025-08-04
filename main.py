import asyncio
import base64
import binascii
import uuid
from contextlib import asynccontextmanager
from io import BytesIO

import requests
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from comfyui import ComfyUIClient
from utils import (InactivityMonitor, define_seed, fetch_image_as_base64,
                   get_image_bytes_from_url, is_data_url, remove_b64_header)
from workflows.default_workflow import create_default_workflow
from workflows.face_workflow import create_face_workflow
from workflows.lora_workflow import create_lora_workflow

# ComfyUI server configuration
COMFYUI_SERVER = "127.0.0.1:8188"
INACTIVITY_TIMEOUT = 1800  # seconds
CHECK_INTERVAL = 5 # seconds

cambioworkflow = False
comfyUiClient = ComfyUIClient(f"http://{COMFYUI_SERVER}")

inactivity_monitor = InactivityMonitor(timeout=INACTIVITY_TIMEOUT, check_interval=CHECK_INTERVAL, comfyUiClient=comfyUiClient)

@asynccontextmanager
async def lifespan(app: FastAPI):

    asyncio.create_task(inactivity_monitor.inactivity_monitor())
    yield

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
    inactivity_monitor.reset()
    comfyUiClient.set_non_free()
    response = await call_next(request)
    return response

@app.websocket("/lorasuib/api/ws/")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        
        # Generar un ID único para esta conexión
        client_id = websocket.query_params.get("clientId", str(uuid.uuid4()))
        
        comfyui_ws_url = f"ws://{COMFYUI_SERVER}/ws?clientId={client_id}"
        async with websockets.connect(comfyui_ws_url) as comfyui_ws:
            try:
                async for data in comfyui_ws:
                    await websocket.send_text(data) # type: ignore
            except Exception as e:
                print(f"Error forwarding from ComfyUI to client: {str(e)}")
            
    except Exception as e:
        print(f"Error en la conexión WebSocket: {str(e)}")
        try:
            await websocket.close()
        except:
            pass

class ImageRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    seed: int = -1
    cfg: float = 1.0
    steps: int = 25
    denoise: float = 1.00
    lora: str = ""

class MaskRequest(ImageRequest):
    image_b64: str
    mask_b64: str

class MaskEnhancerRequest(MaskRequest):
    maskWidth: int
    maskHeight: int

def toggle_cambioworkflow():
    global cambioworkflow
    cambioworkflow = not cambioworkflow

def fetch_image_as_base64(data):
    with urllib.request.urlopen(data["imageUrl"]) as response:
        return base64.b64encode(response.read())
    
def get_upscaler_and_extend_factor(maskSize: tuple[int, int]) -> tuple:
    maskWidth, maskHeight = maskSize
    maskmax = max(int(maskWidth), int(maskHeight))
    extend_factor = 2.0
    if maskmax < 1024:
        upscaler_factor = 1024 / (maskmax * extend_factor)
        if upscaler_factor > 4.0:
            upscaler_factor = 4.0
        elif upscaler_factor < 1.0:
            upscaler_factor = 1.0
    return upscaler_factor, extend_factor
    
def optimize_image_for_processing(img_bytes: bytes, img_type: str):
    # Resize image to specified width and height
    try:
        img = Image.open(BytesIO(img_bytes))
        
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
    
    return img_bytes

async def prepare_img_bytes(img_data: str, img_type: str) -> bytes:
    if not img_data:
        return b""
    if is_data_url(img_data):
        img_bytes = get_image_bytes_from_url(img_data)
    else:
        img_b64 = remove_b64_header(img_data)
        
        try:
            img_bytes = base64.b64decode(img_b64)
        except binascii.Error as e:
            raise HTTPException(status_code=400, detail=f"Base64 inválido para {img_type}: {e}")
    
    img_bytes = optimize_image_for_processing(img_bytes, img_type)
    
    return img_bytes

async def get_history_generation(prompt_id: str) -> dict:
    while True:
        try:
            history = comfyUiClient.get_history(prompt_id)
            if prompt_id in history:
                return history
            else:
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error getting history: {str(e)}")
            return {}

def prepare_image_url(prompt_id: str, history_data: dict, saveImageNode: str) -> dict:
    outputs = history_data[prompt_id]["outputs"]
    if saveImageNode in outputs:  # ID del nodo SaveImage
        images = outputs[saveImageNode]["images"]
        if images:
            image_data = images[0]
            image_url = f"http://{COMFYUI_SERVER}/view?filename={image_data['filename']}&type={image_data['type']}&subfolder={image_data.get('subfolder', '')}"
            print(f"Generated image URL: {image_url}")
            return {"imageUrl": image_url}

    raise HTTPException(status_code=500, detail="No se pudo obtener la imagen generada")

async def _queue_and_wait_for_completion(workflow: dict, save_image_node: str) -> dict:
    try:
        response = comfyUiClient.queue_prompt(workflow)
        prompt_id = response["prompt_id"]
    except Exception as e:
        print(f"Error in queue_prompt: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")

    history_data = await get_history_generation(prompt_id)
    image_data = prepare_image_url(prompt_id, history_data, save_image_node)
    return image_data

async def _process_comfyui_generation(workflow: dict, save_image_node: str, seed: int) -> dict:
    try:
        image_data = await _queue_and_wait_for_completion(workflow, save_image_node)
        return prepare_api_response(image_data, seed)
    except Exception as e:
        print(f"Error during ComfyUI generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}")
    
async def _upload_image_to_comfyui(img_data: str, img_type: str, img_name: str) -> dict:
    img_bytes = await prepare_img_bytes(img_data, img_type)

    return comfyUiClient.post_image(img_name, img_bytes, img_type)

async def send_files_to_comfyui(image: str, mask: str, cambioworkflow: bool) -> list:
    results = []
    results.append(await _upload_image_to_comfyui(image, "image", "imagen1.png" if cambioworkflow else "imagen2.png"))
    results.append(await _upload_image_to_comfyui(mask, "mask", "mask1.png" if cambioworkflow else "mask2.png"))
    return results

def prepare_api_response(image_data, seed):
    image = fetch_image_as_base64(image_data)
    image_data["image"] = image
    image_data["seed"] = seed
    return image_data

@app.post("/lorasuib/api/generate-simple")
async def generate_simple_image(req: ImageRequest):
    try:
        seed = define_seed(req.seed)
        workflow = create_default_workflow(req.prompt, seed, cfg=req.cfg, steps=req.steps, lora=req.lora, width=req.width, height=req.height, denoise=req.denoise)

        return await _process_comfyui_generation(workflow, "27", seed)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/lorasuib/api/generate-mask")
async def generate_mask_image(req: MaskRequest):
    try:
        seed = define_seed(req.seed)
        toggle_cambioworkflow()

        workflow = create_lora_workflow(req.prompt, seed, req.width, req.height, req.denoise, req.lora, "imagen1.png" if cambioworkflow else "imagen2.png", "mask1.png" if cambioworkflow else "mask2.png", req.cfg, req.steps)

        await send_files_to_comfyui(req.image_b64, req.mask_b64, cambioworkflow)

        return await _process_comfyui_generation(workflow, "22", seed)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/lorasuib/api/face-enhancer")
async def generate_enhancer(req: MaskEnhancerRequest):
    try:
        seed = define_seed(req.seed)
        toggle_cambioworkflow()

        upscaler_factor, extend_factor = get_upscaler_and_extend_factor((req.maskWidth, req.maskHeight))
        

        workflow = create_face_workflow(req.prompt, seed, req.width, req.height, req.denoise, req.lora, upscaler_factor, extend_factor, "imagen1.png" if cambioworkflow else "imagen2.png", "mask1.png" if cambioworkflow else "mask2.png", req.cfg, req.steps)

        await send_files_to_comfyui(req.image_b64, req.mask_b64, cambioworkflow)

        return await _process_comfyui_generation(workflow, "60", seed)

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