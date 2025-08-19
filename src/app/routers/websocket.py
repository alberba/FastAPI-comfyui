import uuid

from fastapi import APIRouter, WebSocket

from app.deps import comfyUiClient

router = APIRouter()


@router.websocket("/ws/")
async def websocket_endpoint(websocket_browser: WebSocket):
    try:
        await websocket_browser.accept()

        # Generar un ID único para esta conexión
        client_id = websocket_browser.query_params.get("clientId", str(uuid.uuid4()))

        async for data in comfyUiClient.stream_websocket_messages(client_id):
            await websocket_browser.send_text(str(data))

    except Exception as e:
        print(f"Error en la conexión WebSocket: {str(e)}")
        try:
            await websocket_browser.close()
        except:
            pass


def get_router():
    return router
