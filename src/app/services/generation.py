"""
Servicios de generación de imágenes

Implementa la lógica de la aplicación relacionada con la generación de imágenes
a través de ComfyUI, actuando como capa intermedia entre los endpoints
de la API y el cliente ComfyUI.

Responsabilidades:
- Interpretar y transformar los resultados obtenidos de ComfyUI
- Devolver respuestas adecuadas a los endpoints de la API
- Manejar la lógica de subida de imágenes y máscaras
"""

import asyncio

from fastapi import HTTPException

from app.deps import comfyUiClient
from app.utils import fetch_image_as_base64
from app.services.images import prepare_image_url_endpoint, _upload_image_to_comfyui


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


async def _queue_and_wait_for_completion(workflow: dict, save_image_node: str) -> dict:
    try:
        response = comfyUiClient.queue_prompt(workflow)
        prompt_id = response["prompt_id"]
    except Exception as e:
        print(f"Error in queue_prompt: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}"
        ) from e

    history_data = await get_history_generation(prompt_id)
    image_data = prepare_image_url_endpoint(prompt_id, history_data, save_image_node)
    return image_data


def prepare_api_response(image_data, seed):
    image = fetch_image_as_base64(image_data)
    image_data["image"] = image
    image_data["seed"] = seed
    return image_data


async def _process_comfyui_generation(
    workflow: dict, save_image_node: str, seed: int
) -> dict:
    try:
        image_data = await _queue_and_wait_for_completion(workflow, save_image_node)
        return prepare_api_response(image_data, seed)
    except Exception as e:
        print(f"Error during ComfyUI generation: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error al enviar a ComfyUI: {str(e)}"
        ) from e


async def send_files_to_comfyui(image: str, mask: str, workflow_flag: bool) -> list:
    results = []
    results.append(
        await _upload_image_to_comfyui(
            image, "image", "imagen1.png" if workflow_flag else "imagen2.png"
        )
    )
    results.append(
        await _upload_image_to_comfyui(
            mask, "mask", "mask1.png" if workflow_flag else "mask2.png"
        )
    )
    return results
