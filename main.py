from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import requests
import json
import os
from typing import Optional
from pydantic import BaseModel
import asyncio
import random

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
COMFYUI_SERVER = "http://127.0.0.1:8188"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Almacenar las conexiones WebSocket activas
active_connections = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, clientId: str):
    await websocket.accept()
    active_connections[clientId] = websocket
    try:
        while True:
            # Mantener la conexión viva
            await websocket.receive_text()
    except:
        if clientId in active_connections:
            del active_connections[clientId]

async def send_progress(clientId: str, progress_data: dict):
    if clientId in active_connections:
        await active_connections[clientId].send_json(progress_data)

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    seed: Optional[int] = None
    steps: Optional[int] = 20
    cfg: Optional[float] = 7.0

@app.post("/api/generate")
async def generate_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    seed: Optional[int] = Form(None),
    steps: Optional[int] = Form(20),
    cfg: Optional[float] = Form(7.0)
):
    try:
        # Guardar imagen y máscara
        image_path = os.path.join(UPLOAD_DIR, image.filename)
        mask_path = os.path.join(UPLOAD_DIR, mask.filename)
        
        with open(image_path, "wb") as f:
            f.write(await image.read())
        with open(mask_path, "wb") as f:
            f.write(await mask.read())

        # Crear workflow para ComfyUI
        workflow = """ 
{
  "1": {
    "inputs": {
      "unet_name": "flux1-dev.safetensors",
      "weight_dtype": "fp8_e4m3fn"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "3": {
    "inputs": {
      "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
      "clip_name2": "clip_l.safetensors",
      "type": "flux",
      "device": "default"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "4": {
    "inputs": {
      "text": "bibiloni",
      "clip": [
        "3",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "5": {
    "inputs": {
      "vae_name": "ae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "7": {
    "inputs": {
      "text": "",
      "clip": [
        "3",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Prompt)"
    }
  },
  "8": {
    "inputs": {
      "model": [
        "20",
        0
      ]
    },
    "class_type": "DifferentialDiffusion",
    "_meta": {
      "title": "Differential Diffusion"
    }
  },
  "9": {
    "inputs": {
      "guidance": 3.5,
      "conditioning": [
        "4",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "FluxGuidance"
    }
  },
  "10": {
    "inputs": {
      "noise_mask": true,
      "positive": [
        "9",
        0
      ],
      "negative": [
        "7",
        0
      ],
      "vae": [
        "5",
        0
      ],
      "pixels": [
        "17",
        0
      ],
      "mask": [
        "19",
        0
      ]
    },
    "class_type": "InpaintModelConditioning",
    "_meta": {
      "title": "InpaintModelConditioning"
    }
  },
  "11": {
    "inputs": {
      "model": [
        "8",
        0
      ],
      "conditioning": [
        "10",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "BasicGuider"
    }
  },
  "12": {
    "inputs": {
      "noise": [
        "13",
        0
      ],
      "guider": [
        "11",
        0
      ],
      "sampler": [
        "14",
        0
      ],
      "sigmas": [
        "15",
        0
      ],
      "latent_image": [
        "10",
        2
      ]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {
      "title": "SamplerCustomAdvanced"
    }
  },
  "13": {
    "inputs": {
      "noise_seed": 1109338547990423
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "RandomNoise"
    }
  },
  "14": {
    "inputs": {
      "sampler_name": "euler"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "15": {
    "inputs": {
      "scheduler": "simple",
      "steps": 20,
      "denoise": 0.9000000000000001,
      "model": [
        "20",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "17": {
    "inputs": {
      "image": "clipspace/clipspace-mask-5649382.600000143.png [input]",
      "resize": false,
      "width": 2048,
      "height": 2048,
      "repeat": 1,
      "keep_proportion": true,
      "divisible_by": 2,
      "mask_channel": "alpha",
      "background_color": ""
    },
    "class_type": "LoadAndResizeImage",
    "_meta": {
      "title": "Load & Resize Image"
    }
  },
  "19": {
    "inputs": {
      "kernel_size": 10,
      "sigma": 10,
      "mask": [
        "17",
        1
      ]
    },
    "class_type": "ImpactGaussianBlurMask",
    "_meta": {
      "title": "Gaussian Blur Mask"
    }
  },
  "20": {
    "inputs": {
      "lora_name": "Bibiloni1024.safetensors",
      "strength_model": 0.9500000000000002,
      "model": [
        "1",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  },
  "21": {
    "inputs": {
      "samples": [
        "12",
        0
      ],
      "vae": [
        "5",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "22": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "21",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "32": {
    "inputs": {
      "mask": [
        "17",
        1
      ]
    },
    "class_type": "MaskToImage",
    "_meta": {
      "title": "Convert Mask to Image"
    }
  },
  "33": {
    "inputs": {
      "images": [
        "32",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  }
}
"""
        workflow = json.loads(workflow)

        # Enviar a ComfyUI
        response = requests.post(
            f"{COMFYUI_SERVER}/api/prompt",
            json=workflow
        )
        response.raise_for_status()
        
        return {"message": "Imagen en proceso de generación", "prompt_id": response.json()["prompt_id"]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/generate-simple")
async def generate_simple_image(prompt: str, client_id: Optional[str] = None):
    try:
        # Generar un seed aleatorio
        random_seed = random.randint(0, 2**32 - 1)

        # Crear workflow simple para ComfyUI
        workflow = {
            "prompt": {
  "5": {
    "inputs": {
      "seed": random_seed,
      "steps": 25,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1,
      "model": [
        "35",
        0
      ],
      "positive": [
        "29",
        0
      ],
      "negative": [
        "20",
        0
      ],
      "latent_image": [
        "6",
        0
      ]
    },
    "class_type": "KSampler",
    "_meta": {
      "title": "KSampler"
    }
  },
  "6": {
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "class_type": "EmptyLatentImage",
    "_meta": {
      "title": "Empty Latent Image"
    }
  },
  "7": {
    "inputs": {
      "samples": [
        "5",
        0
      ],
      "vae": [
        "9",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "vae_name": "ae.safetensors"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "11": {
    "inputs": {
      "clip_name1": "t5xxl_fp8_e4m3fn.safetensors",
      "clip_name2": "clip_l.safetensors",
      "type": "flux",
      "device": "default"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "14": {
    "inputs": {
      "text": "A photorealistic portrait of a man named Mqlmscr smiling in the park",
      "clip": [
        "11",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "Positive Prompt"
    }
  },
  "15": {
    "inputs": {
      "unet_name": "flux1-dev.safetensors",
      "weight_dtype": "fp8_e4m3fn"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "18": {
    "inputs": {
      "lora_name": "mascaroPerfeccionadoFlux.safetensors",
      "strength_model": 1.0000000000000002,
      "model": [
        "15",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  },
  "20": {
    "inputs": {
      "text": "",
      "clip": [
        "11",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "Negative Prompt"
    }
  },
  "27": {
    "inputs": {
      "filename_prefix": "LoRA_Flux_no_inpaint",
      "images": [
        "7",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "29": {
    "inputs": {
      "guidance": 3.5,
      "conditioning": [
        "14",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "FluxGuidance"
    }
  },
  "35": {
    "inputs": {
      "lora_name": "aidmaImageUprader-FLUX-v0.3.safetensors",
      "strength_model": 1.0000000000000002,
      "model": [
        "18",
        0
      ]
    },
    "class_type": "LoraLoaderModelOnly",
    "_meta": {
      "title": "LoraLoaderModelOnly"
    }
  }
}
        }

        # Enviar a ComfyUI
        response = requests.post(
            f"{COMFYUI_SERVER}/api/prompt",
            json=workflow
        )
        response.raise_for_status()
        
        prompt_id = response.json()["prompt_id"]
        
        # Si hay un client_id, monitorear el progreso
        if client_id:
            while True:
                progress = requests.get(f"{COMFYUI_SERVER}/api/progress")
                if progress.status_code == 200:
                    progress_data = progress.json()
                    await send_progress(client_id, {
                        "status": "in_progress",
                        "progress": progress_data.get("value", 0),
                        "eta": progress_data.get("eta", 0)
                    })

                # Verificar si la generación terminó
                history = requests.get(f"{COMFYUI_SERVER}/api/history/{prompt_id}")
                if history.status_code == 200 and history.json():
                    break
                
                await asyncio.sleep(1)

        # Obtener la imagen generada
        history_data = history.json()
        if prompt_id in history_data:
            outputs = history_data[prompt_id]["outputs"]
            if "27" in outputs:  # ID del nodo SaveImage
                images = outputs["27"]["images"]
                if images:
                    image_data = images[0]
                    result = {
                        "status": "completed",
                        "image": {
                            "filename": image_data["filename"],
                            "type": image_data["type"],
                            "subfolder": image_data.get("subfolder", ""),
                            "url": f"{COMFYUI_SERVER}/view?filename={image_data['filename']}&type={image_data['type']}&subfolder={image_data.get('subfolder', '')}"
                        }
                    }
                    if client_id:
                        await send_progress(client_id, result)
                    return result
        
        error_result = {"status": "error", "message": "No se pudo obtener la imagen generada"}
        if client_id:
            await send_progress(client_id, error_result)
        return error_result

    except Exception as e:
        error_result = {"status": "error", "message": str(e)}
        if client_id:
            await send_progress(client_id, error_result)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/progress/{prompt_id}")
async def get_progress(prompt_id: str):
    try:
        # Obtener el progreso actual
        progress = requests.get(f"{COMFYUI_SERVER}/api/progress")
        if progress.status_code == 200:
            progress_data = progress.json()
            return {
                "status": "in_progress",
                "progress": progress_data.get("value", 0),
                "eta": progress_data.get("eta", 0)
            }
        
        # Si no hay progreso, verificar si la generación terminó
        history = requests.get(f"{COMFYUI_SERVER}/api/history/{prompt_id}")
        if history.status_code == 200 and history.json():
            return {"status": "completed"}
        
        return {"status": "pending"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{prompt_id}")
async def get_status(prompt_id: str):
    try:
        response = requests.get(f"{COMFYUI_SERVER}/api/history/{prompt_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/test")
async def test():
    return {"message": "Hello, World!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 