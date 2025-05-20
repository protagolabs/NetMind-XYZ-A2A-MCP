import typing
import logging
from typing import Any

import aiohttp

from core.env_helper import EnvHelper

logging.basicConfig(level=logging.INFO)


Object = typing.TypeVar("Object", bound=dict[str, Any])


class XyzPlatformServer:
    def __init__(self):
        self.base_url = EnvHelper.get_xyz_platform_url()

    async def parser_reponse(
        self, path, response: aiohttp.ClientResponse
    ) -> Any | typing.NoReturn:
        data = await response.json()

        if response.status == 200:
            if not data.get("isFailed"):
                logging.info(f"请求 {path} 成功, 返回数据: {data}")
                return data["result"]
            else:
                err = f"请求 {path} 错误, 错误信息: {data['message']}"
                logging.error(err)
                raise Exception(err)

        err = f"请求 {path} 失败, 状态码: {response.status}"
        logging.error(err)
        raise Exception(err)

    async def get_model_info(self, agent_id: int):
        path = "/agents/model/info"
        async with aiohttp.ClientSession(
            base_url=self.base_url,
            timeout=aiohttp.ClientTimeout(EnvHelper.get_http_timeout()),
        ) as client:
            async with client.get(url=path, params={"agent_id": agent_id}) as response:
                return await self.parser_reponse(path, response)

    async def conversation_read(self, user_id: str, agent_id: int):
        path = "/conversation/read"

        async with aiohttp.ClientSession(
            base_url=self.base_url,
            timeout=aiohttp.ClientTimeout(EnvHelper.get_http_timeout()),
        ) as client:
            async with client.get(
                url=path, params={"agent_id": agent_id, "user_id": user_id}
            ) as response:
                return await self.parser_reponse(path, response)

    async def conversation_write(
        self, user_id: str, agent_id: int, message: str, role: str
    ):
        path = "/conversation/write"

        async with aiohttp.ClientSession(
            base_url=self.base_url,
            timeout=aiohttp.ClientTimeout(EnvHelper.get_http_timeout()),
        ) as client:
            async with client.post(
                url=path,
                json={
                    "userId": str(user_id),
                    "agentId": int(agent_id),
                    "content": message,
                    "role": role,
                },
            ) as response:
                return await self.parser_reponse(path, response)
