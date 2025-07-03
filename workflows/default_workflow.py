def create_default_workflow(prompt: str, random_seed: int, width: int, height: int, lora: str, cfg: float = 1.0, steps: int = 25, ) -> dict:
    """
    Crea un workflow simple para ComfyUI con la configuración por defecto.
    
    Args:
        prompt (str): El prompt de texto para la generación
        random_seed (int): Semilla aleatoria para la generación
        cfg (float): Valor de CFG para la generación (default: 1.0)
        steps (int): Número de pasos para la generación (default: 25)
        
    Returns:
        dict: Workflow configurado para ComfyUI
    """
    return {
        "5": {
            "inputs": {
                "seed": random_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "beta",
                "denoise": 1.00,
                "model": ["36", 0],
                "positive": ["29", 0],
                "negative": ["20", 0],
                "latent_image": ["6", 0]
            },
            "class_type": "KSampler"
        },
        "6": {
            "inputs": {
                "width": width,
                "height": height,
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
                "lora_name": "aidmaImageUprader-FLUX-v0.3.safetensors" if lora == "" else lora,
                "strength_model": 0 if lora == "" else 1.0,
                "model": ["15", 0]
            },
            "class_type": "LoraLoaderModelOnly"
        },
        "36": {
            "inputs": {
                "lora_name": "aidmaImageUprader-FLUX-v0.3.safetensors",
                "strength_model": 0.15 if lora == "" else 0.2,
                "model": ["35", 0]
            },
            "class_type": "LoraLoaderModelOnly"
        },
    } 