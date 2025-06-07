import requests
import json

# URL de nuestra API
API_URL = "http://localhost:8000/api/workflow"

# Ejemplo de workflow de ComfyUI
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

def queue_prompt(prompt):
    try:
        response = requests.post(
            API_URL,
            json={"workflow": prompt}
        )
        response.raise_for_status()
        print("Workflow enviado correctamente:", response.json())
    except requests.exceptions.RequestException as e:
        print("Error al enviar el workflow:", str(e))

prompt = json.loads(workflow)
#set the text prompt for our positive CLIPTextEncode
prompt["6"]["inputs"]["text"] = "masterpiece best quality man"

#set the seed for our KSampler node
prompt["3"]["inputs"]["seed"] = 5

# Enviar el workflow
queue_prompt(prompt)