# ComfyUI API

This API serves as a bridge between your ComfyUI workflow and a web interface. An Example web interface is [Comfyui-web](https://github.com/alberba/comfyui-web)

## Prerequisites

- Python 3.8 or higher
- ComfyUI running locally (default port: 8188)
- Node.js and npm (for Astro frontend)

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Start the API server:

```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /`: Health check endpoint
- `POST /api/workflow`: Execute a ComfyUI workflow with a provided JSON workflow.

### General Image Generation (with LoRA support)

- `POST /lorasuib/api/generate-simple`: Generate an image with basic parameters, including optional LoRA application.
  **Parameters (JSON Body):**

  - `prompt` (string, required): The text prompt for image generation.
  - `width` (integer, optional, default: 1024): The desired width of the generated image.
  - `height` (integer, optional, default: 1024): The desired height of the generated image.
  - `seed` (integer, optional, default: -1): A seed for reproducible random generation. Use -1 for a random seed.
  - `cfg` (float, optional, default: 1.0): Classifier Free Guidance value.
  - `steps` (integer, optional, default: 25): Number of sampling steps.
  - `lora` (string, optional, default: ""): The name of the LoRA model to use. If an empty string, no LoRA is applied.

- `POST /lorasuib/api/generate-mask`: Generate an image with an input image and mask, supporting LoRA application.
  **Parameters (Form Data - `multipart/form-data`):**
  - `prompt` (string, required): The text prompt for image generation.
  - `width` (integer, required): The desired width of the generated image.
  - `height` (integer, required): The desired height of the generated image.
  - `seed` (integer, required): A seed for reproducible random generation.
  - `cfg` (float, required): Classifier Free Guidance value.
  - `steps` (integer, required): Number of sampling steps.
  - `lora` (string, required): The name of the LoRA model to use.
  - `image` (file, required): The input image file.
  - `mask` (file, required): The input mask file.

### Face Enhancer Workflow

- `POST /lorasuib/api/face-enhancer`: Enhance a face in an image using a pre-defined ComfyUI workflow.
  **Parameters (Form Data - `multipart/form-data`):**
  - `prompt` (string, required): The text prompt for face enhancement.
  - `width` (integer, optional, default: 1024): The desired width of the output image.
  - `height` (integer, optional, default: 1024): The desired height of the output image.
  - `seed` (integer, optional, default: -1): A seed for reproducible random generation.
  - `cfg` (float, optional, default: 1.0): Classifier Free Guidance value.
  - `steps` (integer, optional, default: 25): Number of sampling steps.
  - `lora` (string, required): The name of the LoRA model to use for enhancement.
  - `image` (file, required): The input image file containing the face to enhance.
  - `mask` (file, required): The input mask file for the face region.

### LoRA Model Information

- `GET /lorasuib/api/get-loras`: Get a list of available LoRA models.
