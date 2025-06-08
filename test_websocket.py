import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print("Conectado al WebSocket!")
            
            # Enviar un mensaje de prueba
            test_message = "Hola, esto es una prueba"
            print(f"Enviando mensaje: {test_message}")
            await websocket.send(test_message)
            
            # Esperar respuesta
            response = await websocket.recv()
            print(f"Respuesta recibida: {response}")
            
            # Mantener la conexión abierta por un momento
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_websocket()) 