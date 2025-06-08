import asyncio
import websockets
import json
import uuid

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    try:
        async with websockets.connect(uri) as websocket:
            print("Conectado al WebSocket!")
            
            # Crear el prompt para ComfyUI
            prompt = {
                "3": {
                    "class_type": "KSampler",
                    "inputs": {
                        "cfg": 8,
                        "denoise": 1,
                        "latent_image": ["5", 0],
                        "model": ["4", 0],
                        "negative": ["7", 0],
                        "positive": ["6", 0],
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "seed": 8566257,
                        "steps": 20
                    }
                },
                "4": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {
                        "ckpt_name": "v1-5-pruned-emaonly.safetensors"
                    }
                },
                "5": {
                    "class_type": "EmptyLatentImage",
                    "inputs": {
                        "batch_size": 1,
                        "height": 512,
                        "width": 512
                    }
                },
                "6": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "clip": ["4", 1],
                        "text": "masterpiece best quality girl"
                    }
                },
                "7": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "clip": ["4", 1],
                        "text": "bad hands"
                    }
                },
                "8": {
                    "class_type": "VAEDecode",
                    "inputs": {
                        "samples": ["3", 0],
                        "vae": ["4", 2]
                    }
                },
                "save_image_websocket_node": {
                    "class_type": "SaveImageWebsocket",
                    "inputs": {
                        "images": ["8", 0]
                    }
                }
            }
            
            # Enviar el prompt
            print("Enviando prompt...")
            await websocket.send(json.dumps({"prompt": prompt}))
            
            # Esperar respuestas
            while True:
                response = await websocket.recv()
                data = json.loads(response)
                print(f"Respuesta recibida: {data}")
                
                if data.get("type") == "executing":
                    print(f"Ejecutando nodo: {data.get('data', {}).get('node')}")
                elif data.get("type") == "progress":
                    print(f"Progreso: {data.get('data', {}).get('value')}%")
                elif data.get("type") == "executed":
                    print("Nodo ejecutado")
                elif data.get("type") == "execution_complete":
                    print("Generación completada")
                    break
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_websocket()) 