import weakref
from typing import Optional

from multi_agent_centre.client.agent import AgentClient
from core.env_helper import EnvHelper
from core.server.xyz_platform import XyzPlatformServer

xyz_server = XyzPlatformServer()


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
