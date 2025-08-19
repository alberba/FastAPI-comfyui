from fastapi import FastAPI, Request

import src.app.routers.api as images_router
import app.routers.websocket as websocket_router
from app.deps import comfyUiClient, inactivity_monitor, lifespan


def create_app() -> FastAPI:

    app = FastAPI(title="ComfyUI Image Generation API", lifespan=lifespan)

    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        # In production, replace with your Astro app's domain
        allow_origins=["http://localhost:4321", "*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(images_router.get_router(), prefix="/lorasuib/api")
    app.include_router(websocket_router.get_router(), prefix="/lorasuib/api")
    return app


app = create_app()


@app.middleware("http")
async def activity_middleware(request: Request, call_next):
    inactivity_monitor.reset()
    comfyUiClient.set_non_free()
    response = await call_next(request)
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
