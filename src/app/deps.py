"""
Proporciona instancias compartidas de servicios y clientes que pueden ser inyectados en cualquier punto de la aplicación

Gestiona:
- Cliente ComfyUI
- Control de inactividad del sistema

Este módulo es fundamental para mantener un solo punto para las dependencias compartidas y evitar la inicialización repetida
"""

import asyncio
import time
from contextlib import asynccontextmanager

from app.comfyui import ComfyUIClient
from app.config import CHECK_INTERVAL, COMFYUI_SERVER, INACTIVITY_TIMEOUT


class InactivityMonitor:
    """
    A simple inactivity monitor that tracks the last activity time
    """

    def __init__(self, comfyUiClient: ComfyUIClient, timeout=60, check_interval=5):
        """
        Initialize the inactivity monitor with a specified timeout.
        """
        self.timeout = timeout
        self.check_interval = check_interval
        self.comfyUiClient = comfyUiClient
        self.last_activity_time = time.time()

    def reset(self):
        """
        Reset the last activity timestamp to the current time.
        """
        self.last_activity_time = time.time()

    async def inactivity_monitor(self):
        while True:
            await asyncio.sleep(self.check_interval)
            if (
                time.time() - self.last_activity_time
            ) > self.timeout and not self.comfyUiClient.get_is_free():
                self.comfyUiClient.send_free()


comfyUiClient = ComfyUIClient(f"{COMFYUI_SERVER}")

inactivity_monitor = InactivityMonitor(
    timeout=INACTIVITY_TIMEOUT,
    check_interval=CHECK_INTERVAL,
    comfyUiClient=comfyUiClient,
)


@asynccontextmanager
async def lifespan(app):

    asyncio.create_task(inactivity_monitor.inactivity_monitor())
    yield

    print("Inactivity monitor shutting down.")
