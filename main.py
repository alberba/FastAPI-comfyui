from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import requests
import json
import os
from typing import Optional
from pydantic import BaseModel

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

@app.post("/api/generate-simple")
async def generate_simple_image(
    prompt: str = Form(...),
):
    try:
        # Crear workflow simple para ComfyUI
        workflow = {
            "prompt": {
  "5": {
    "inputs": {
      "seed": 121019983053432,
      "steps": 25,
      "cfg": 1,
      "sampler_name": "euler",
      "scheduler": "beta",
      "denoise": 1,
      "model": [
        "18",
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
      "text": prompt,
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
      "lora_name": "Mascaro\\mascaroPerfeccionadoFlux.safetensors",
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
  }
}
        }

        # Enviar a ComfyUI
        response = requests.post(
            f"{COMFYUI_SERVER}/api/prompt",
            json=workflow
        )
        response.raise_for_status()
        
        return {"message": "Imagen en proceso de generación", "prompt_id": response.json()["prompt_id"]}

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