import atexit
import logging
import asyncio
import threading
from concurrent.futures import Future
from typing import Coroutine, Any

from multi_agent_centre.client.agent import AgentClient

from core.env_helper import EnvHelper
from core.server import xyz_server


class RpcManager:
    @classmethod
    async def get_agent_client(
        cls,
        agent_id: int | str,
    ) -> AgentClient:
        model_info = await xyz_server.get_model_info(agent_id)
        model_config = {
            "model_name": model_info["model_name"],
            "api_key": model_info["api_key"],
        }

        return AgentClient(
            agent_id=agent_id,
            model_config=model_config,
            host=EnvHelper.get_multi_agent_host(),
            port=EnvHelper.get_multi_agent_port(),
        )


class RpcExecutor:
    def __init__(self):
        self._loop = None
        self._thread = None
        self._running = False

        self.start()

    def start(self):
        if self._running:
            return

        atexit.register(self.stop)
        self._loop = asyncio.new_event_loop()
        loop_started_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop, args=(loop_started_event,), daemon=True
        )
        self._running = True
        self._thread.start()
        loop_started_event.wait()
        logging.info("gRPC 执行器已启动")

    def _run_loop(self, started_event: threading.Event):
        asyncio.set_event_loop(self._loop)
        started_event.set()

        try:
            self._loop.run_forever()
        finally:
            if hasattr(self._loop, "shutdown_asyncgens"):  # Python 3.6+
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    def stop(self):
        if not self._running or not self._loop or not self._loop.is_running():
            return

        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._running = False
        logging.info("gRPC 执行器已停止")

    def submit_coroutine(self, coro: Coroutine[Any, Any, Any]) -> Future:
        if not self._running or not self._loop:
            raise RuntimeError("GrpcAioExecutor is not running.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
