def create_lora_workflow(prompt: str, random_seed: int, width: int, height: int, lora: str, cfg: float = 1.0, steps: int = 20) -> dict:
    """
    Crea un workflow simple para ComfyUI con la configuración por defecto.
    
    Args:
        prompt (str): El prompt de texto para la generación
        random_seed (int): Semilla aleatoria para la generación
        cfg (float): Valor de CFG para la generación (default: 1.0)
        steps (int): Número de pasos para la generación (default: 20)
        
    Returns:
        dict: Workflow configurado para ComfyUI
    """
    return {
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
        "text": prompt,
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
        "noise_mask": True,
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
            "42",
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
        "noise_seed": random_seed,
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
        "steps": steps,
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
    "19": {
        "inputs": {
        "kernel_size": 10,
        "sigma": 10,
        "mask": [
            "43",
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
        "lora_name": "aidmaImageUprader-FLUX-v0.3.safetensors" if lora == "" else lora,
        "strength_model": 0.20 if lora == "" else 0.9500000000000002,
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
    "42": {
        "inputs": {
        "image": "image.png",
        "resize": True,
        "width": width,
        "height": height,
        "repeat": 1,
        "keep_proportion": False,
        "divisible_by": 2,
        "mask_channel": "alpha",
        "background_color": ""
        },
        "class_type": "LoadAndResizeImage",
        "_meta": {
        "title": "Load & Resize Image"
        }
    },
    "43": {
        "inputs": {
        "image": "mask.png",
        "resize": True,
        "width": width,
        "height": height,
        "repeat": 1,
        "keep_proportion": False,
        "divisible_by": 2,
        "mask_channel": "alpha",
        "background_color": "#fff"
        },
        "class_type": "LoadAndResizeImage",
        "_meta": {
        "title": "Load & Resize Image"
        }
    }
    }