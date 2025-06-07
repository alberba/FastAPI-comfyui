from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
from typing import Dict, Any
from pydantic import BaseModel

app = FastAPI(title="ComfyUI API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Astro app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ComfyUI server configuration
COMFYUI_SERVER = "http://127.0.0.1:8188"

class WorkflowRequest(BaseModel):
    workflow: Dict[str, Any]

@app.get("/")
async def read_root():
    return {"message": "ComfyUI API is running"}

@app.post("/api/workflow")
async def execute_workflow(workflow_request: WorkflowRequest):
    try:
        # Send workflow to ComfyUI
        response = requests.post(
            f"{COMFYUI_SERVER}/api/prompt",
            json=workflow_request.workflow
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history():
    try:
        response = requests.get(f"{COMFYUI_SERVER}/api/history")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 