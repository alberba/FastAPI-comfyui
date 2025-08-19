"""
Modelos de datos y validación

Define los esquemas Pydantic utilizados para:
- Validar los datos de entrada en los endpoints
- Documentar automáticamente la API con OpenAPI
"""

from pydantic import BaseModel


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
