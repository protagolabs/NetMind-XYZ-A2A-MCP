import typing
import logging
from typing import Any

import aiohttp
import asyncio_atexit

from . import aio_retry
from core.env_helper import EnvHelper

logging.basicConfig(level=logging.INFO)


Object = typing.TypeVar("Object", bound=dict[str, Any])


class XyzPlatformServer:
    def __init__(self):
        self.base_url = EnvHelper.get_xyz_platform_url()

    @aio_retry
    async def get_model_info(self, agent_id: int):
        path = "/agents/model/info"
        logging.info(f"请求: {path}")

        async with aiohttp.ClientSession(
            base_url=self.base_url, timeout=aiohttp.ClientTimeout(30)
        ) as client:
            async with client.get(url=path, params={"agent_id": agent_id}) as response:
                data = await response.json()

                if response.status == 200:
                    if not data.get("isFailed"):
                        logging.info(f"请求成功, 返回 data: {data}")
                        return data["result"]
                    else:
                        err = f"请求错误, 错误信息: {data['message']}"
                        logging.error(err)
                        raise Exception(err)

                err = f"请求失败, 状态码: {response.status}"
                logging.error(err)
                raise Exception(err)
