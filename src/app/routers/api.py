import requests
from fastapi import APIRouter, HTTPException

from app.config import COMFYUI_SERVER
from app.schemas import ImageRequest, MaskEnhancerRequest, MaskRequest
from app.services.generation import (_process_comfyui_generation,
                                     send_files_to_comfyui)
from app.services.images import get_upscaler_and_extend_factor
from app.utils import define_seed
from app.workflows.default_workflow import create_default_workflow
from app.workflows.face_workflow import create_face_workflow
from app.workflows.lora_workflow import create_lora_workflow

cambioworkflow = False

router = APIRouter()


def toggle_cambioworkflow():
    global cambioworkflow
    cambioworkflow = not cambioworkflow


@router.post("/lorasuib/api/generate-simple")
async def generate_simple_image(req: ImageRequest):
    try:
        seed = define_seed(req.seed)
        workflow = create_default_workflow(
            req.prompt,
            seed,
            cfg=req.cfg,
            steps=req.steps,
            lora=req.lora,
            width=req.width,
            height=req.height,
            denoise=req.denoise,
        )

        return await _process_comfyui_generation(workflow, "27", seed)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/generate-mask")
async def generate_mask_image(req: MaskRequest):
    """Generate an image with a mask using ComfyUI."""
    try:
        seed = define_seed(req.seed)
        toggle_cambioworkflow()

        workflow = create_lora_workflow(
            req.prompt,
            seed,
            req.width,
            req.height,
            req.denoise,
            req.lora,
            "imagen1.png" if cambioworkflow else "imagen2.png",
            "mask1.png" if cambioworkflow else "mask2.png",
            req.cfg,
            req.steps,
        )

        await send_files_to_comfyui(req.image_b64, req.mask_b64, cambioworkflow)

        return await _process_comfyui_generation(workflow, "22", seed)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/face-enhancer")
async def generate_enhancer(req: MaskEnhancerRequest):
    """Generate an image with face enhancement using ComfyUI."""
    try:
        seed = define_seed(req.seed)
        toggle_cambioworkflow()

        upscaler_factor, extend_factor = get_upscaler_and_extend_factor(
            (req.maskWidth, req.maskHeight)
        )

        workflow = create_face_workflow(
            req.prompt,
            seed,
            req.width,
            req.height,
            req.denoise,
            req.lora,
            upscaler_factor,
            extend_factor,
            "imagen1.png" if cambioworkflow else "imagen2.png",
            "mask1.png" if cambioworkflow else "mask2.png",
            req.cfg,
            req.steps,
        )

        await send_files_to_comfyui(req.image_b64, req.mask_b64, cambioworkflow)

        return await _process_comfyui_generation(workflow, "60", seed)

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/get-loras")
async def get_loras():
    """Fetches the list of LORA models from the ComfyUI server."""
    try:
        response = requests.get(f"http://{COMFYUI_SERVER}/models/loras", timeout=10)
        if response.status_code == 200:
            return response.json()
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Error al obtener los modelos LORA: {response.text}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def get_router():
    return router
