"""
Servicios de procesamiento de imágenes

Proporciona funcionalidad especializada en manipular, transformar 
y preparar imágenes antes y después de la interacción con ComfyUI.

Características:
- Conversión de diferentes formatos a bytes
- Redimensionamiento y transformación de imágenes a un formato aceptado por ComfyUI
- Subida de imágenes a ComfyUI

"""

import base64
import binascii
from io import BytesIO

from fastapi import HTTPException
from PIL import Image

from app.config import COMFYUI_SERVER
from app.deps import comfyUiClient
from app.utils import get_image_bytes_from_url, is_data_url, remove_b64_header


def optimize_image_for_processing(img_bytes: bytes, img_type: str) -> bytes:
    # Resize image to specified width and height
    try:
        img = Image.open(BytesIO(img_bytes))

        # Ajustar width y height para que sean divisibles entre 2
        # El width será el tamaño actual de la imagen, ajustando solo el height para
        # que sea divisible entre 2
        new_width = img.width - (img.width % 2)
        new_height = img.height - (img.height % 2)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        output_buffer = BytesIO()
        resized_img.save(output_buffer, format="PNG")
        img_bytes = output_buffer.getvalue()
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error al procesar la imagen para {img_type}: {e}"
        ) from e

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
            raise HTTPException(
                status_code=400, detail=f"Base64 inválido para {img_type}: {e}"
            ) from e

    img_bytes = optimize_image_for_processing(img_bytes, img_type)

    return img_bytes


def prepare_image_url_endpoint(
    prompt_id: str, history_data: dict, saveImageNode: str
) -> dict:
    outputs = history_data[prompt_id]["outputs"]
    if saveImageNode in outputs:  # ID del nodo SaveImage
        images = outputs[saveImageNode]["images"]
        if images:
            image_data = images[0]
            image_url = (
                f"http://{COMFYUI_SERVER}/view?"
                f"filename={image_data['filename']}"
                f"&type={image_data['type']}"
                f"&subfolder={image_data.get('subfolder', '')}"
            )
            print(f"Generated image URL: {image_url}")
            return {"imageUrl": image_url}

    raise HTTPException(status_code=500, detail="No se pudo obtener la imagen generada")


async def _upload_image_to_comfyui(img_data: str, img_type: str, img_name: str) -> dict:
    img_bytes = await prepare_img_bytes(img_data, img_type)

    return comfyUiClient.post_image(img_name, img_bytes, img_type)


def get_upscaler_and_extend_factor(mask_size: tuple[int, int]) -> tuple:
    mask_width, mask_height = mask_size
    maskmax = max(int(mask_width), int(mask_height))
    extend_factor = 2.0
    if maskmax < 1024:
        upscaler_factor = 1024 / (maskmax * extend_factor)
        if upscaler_factor > 4.0:
            upscaler_factor = 4.0
        elif upscaler_factor < 1.0:
            upscaler_factor = 1.0
    return upscaler_factor, extend_factor
