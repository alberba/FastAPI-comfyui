# ComfyUI API

This API serves as a bridge between your ComfyUI workflow and a web interface built with Astro.

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
- `POST /api/workflow`: Execute a ComfyUI workflow
- `GET /api/history`: Get ComfyUI execution history

## Example Usage

To execute a workflow:

```python
import requests

workflow = {
    # Your ComfyUI workflow JSON here
}

response = requests.post(
    "http://localhost:8000/api/workflow",
    json={"workflow": workflow}
)
print(response.json())
```

## CORS Configuration

The API is configured to accept requests from any origin (`*`). In production, you should update the `allow_origins` list in `main.py` to include only your Astro application's domain.

## Security Notes

- This is a development setup. For production:
  - Implement proper authentication
  - Restrict CORS origins
  - Use environment variables for configuration
  - Add rate limiting
  - Implement proper error handling
