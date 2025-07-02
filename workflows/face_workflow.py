def create_face_workflow(prompt: str, random_seed: int, width: int, height: int, lora: str, img_name: str, mask_name: str, cfg: float = 1.0, steps: int = 20) -> dict:
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
        "51",
        0
      ],
      "mask": [
        "56",
        2
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
      "noise_seed": random_seed
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
  "17": {
    "inputs": {
      "image": img_name,
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
  "18": {
    "inputs": {
      "image": mask_name,
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
  },
  "19": {
    "inputs": {
      "kernel_size": 10,
      "sigma": 10,
      "mask": [
        "18",
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
  "32": {
    "inputs": {
      "mask": [
        "19",
        0
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
  },
  "50": {
    "inputs": {
      "model_name": "remacri_original.pth"
    },
    "class_type": "UpscaleModelLoader",
    "_meta": {
      "title": "Load Upscale Model"
    }
  },
  "51": {
    "inputs": {
      "upscale_by": 2,
      "seed": random_seed,
      "steps": 1,
      "cfg": 2,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 0.20000000000000004,
      "mode_type": "Linear",
      "tile_width": 3072,
      "tile_height": 2048,
      "mask_blur": 8,
      "tile_padding": 32,
      "seam_fix_mode": "None",
      "seam_fix_denoise": 1,
      "seam_fix_width": 64,
      "seam_fix_mask_blur": 8,
      "seam_fix_padding": 16,
      "force_uniform_tiles": True,
      "tiled_decode": False,
      "image": [
        "56",
        1
      ],
      "model": [
        "20",
        0
      ],
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
      "upscale_model": [
        "50",
        0
      ]
    },
    "class_type": "UltimateSDUpscale",
    "_meta": {
      "title": "Ultimate SD Upscale"
    }
  },
  "54": {
    "inputs": {
      "image": [
        "56",
        1
      ]
    },
    "class_type": "GetImageSizeAndCount",
    "_meta": {
      "title": "Get Image Size & Count"
    }
  },
  "55": {
    "inputs": {
      "width": [
        "54",
        1
      ],
      "height": [
        "54",
        2
      ],
      "upscale_method": "nearest-exact",
      "keep_proportion": "stretch",
      "pad_color": "0, 0, 0",
      "crop_position": "center",
      "divisible_by": 2,
      "device": "cpu",
      "image": [
        "21",
        0
      ]
    },
    "class_type": "ImageResizeKJv2",
    "_meta": {
      "title": "Resize Image v2"
    }
  },
  "56": {
    "inputs": {
      "downscale_algorithm": "bilinear",
      "upscale_algorithm": "bicubic",
      "preresize": False,
      "preresize_mode": "ensure minimum resolution",
      "preresize_min_width": 1024,
      "preresize_min_height": 1024,
      "preresize_max_width": 16384,
      "preresize_max_height": 16384,
      "mask_fill_holes": True,
      "mask_expand_pixels": 0,
      "mask_invert": False,
      "mask_blend_pixels": 32,
      "mask_hipass_filter": 0.1,
      "extend_for_outpainting": False,
      "extend_up_factor": 1,
      "extend_down_factor": 1,
      "extend_left_factor": 1,
      "extend_right_factor": 1,
      "context_from_mask_extend_factor": 2.0000000000000004,
      "output_resize_to_target_size": False,
      "output_target_width": 512,
      "output_target_height": 512,
      "output_padding": "32",
      "image": [
        "17",
        0
      ],
      "mask": [
        "19",
        0
      ]
    },
    "class_type": "InpaintCropImproved",
    "_meta": {
      "title": "✂️ Inpaint Crop (Improved)"
    }
  },
  "58": {
    "inputs": {
      "stitcher": [
        "56",
        0
      ],
      "inpainted_image": [
        "55",
        0
      ]
    },
    "class_type": "InpaintStitchImproved",
    "_meta": {
      "title": "✂️ Inpaint Stitch (Improved)"
    }
  },
  "59": {
    "inputs": {
      "images": [
        "21",
        0
      ]
    },
    "class_type": "PreviewImage",
    "_meta": {
      "title": "Preview Image"
    }
  },
  "60": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "58",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  }
}