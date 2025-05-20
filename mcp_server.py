import re
import os
import httpx
import time
import logging
from typing import Optional
from fastmcp import FastMCP
from urllib.parse import urlparse

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from core.a2a.client import A2AClient, MakeResponseModel
from core.server import xyz_server
from core.env_helper import EnvHelper


A2A_SERVER_URL = os.getenv("A2A_SERVER_URL", "http://127.0.0.1:5000")

mcp = FastMCP("A2A-Client", host="0.0.0.0", port=EnvHelper.get_mcp_server_port())


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


@mcp.tool()
async def get_agent_card_by_url(url: str):
    """
    Get AgentCard via BASE_URL.

    Args:
        url (str): Agent Server base url.

    Returns:
        dict: A dictionary containing the agent's metadata and capabilities.

    Example Response:

        {
          "name": "XyzAgent: 1036",
          "version": "1.0.0",
          "url": "http://127.0.0.1:5000/1036",
          "capabilities": {
            "streaming": true
          },
          "defaultInputModes": [
            "text/plain"
          ],
          "defaultOutputModes": [
            "text/plain"
          ],
          "skills": [],
          "description": "Agent description..."
        }

    """
    async with httpx.AsyncClient(timeout=EnvHelper.get_http_timeout()) as client:
        agent_server_url = f"{url}/.well-known.json"
        response = await client.get(agent_server_url)
        return response.json()


@mcp.tool()
async def get_agent_card_by_agent_id(agent_id: int):
    """
    Retrieves the IdCard (AgentCard) for a specified agent.

    Args:
        agent_id (int): The unique identifier of the agent whose IdCard is being requested.

    Returns:
        dict: A dictionary containing the agent's metadata and capabilities.

    Example Response:

        {
          "name": "XyzAgent: 1036",
          "version": "1.0.0",
          "url": "http://127.0.0.1:5000/1036",
          "capabilities": {
            "streaming": true
          },
          "defaultInputModes": [
            "text/plain"
          ],
          "defaultOutputModes": [
            "text/plain"
          ],
          "skills": [],
          "description": "Agent description..."
        }

    """
    async with httpx.AsyncClient(timeout=EnvHelper.get_http_timeout()) as client:
        agent_server_url = f"{A2A_SERVER_URL}/.well-known.json"
        response = await client.get(agent_server_url, params={"agent_id": agent_id})
        return response.json()


@mcp.tool()
async def call_agent(
    url: str, from_agent_id: int, message: str, to_agent_id: Optional[int]
):
    """
    Sends a message to a specific Agent Server.

    Args:
        url (str): The service URL of the target Agent Server.
        from_agent_id (int): The ID of the calling agent (typically the sender).
        to_agent_id (int): The ID of the target agent to which the message is being sent.

    Returns:
        str: The response content from the target agent.
    """
    client = A2AClient(url=url)

    logging.info(f"""
          调用 A2AClient call_agent 工具, 参数;
              - urL: {url}
              - form_agent_id: {from_agent_id}
              - to_agent_id: {to_agent_id}
          """)

    path = urlparse(url).path  # /1036

    message_list = await xyz_server.conversation_read(
        user_id=from_agent_id, agent_id=to_agent_id
    )

    rest_message = {
        "role": {"id": from_agent_id, "type": "user", "name": "Default User"},
        "content": message,
        "metadata": {"time": time.time()},
    }

    message_list.append(rest_message)

    if re.match(r"^/\d+$", path):
        # 内部 Agent-Server 发送方式
        send_message = MakeResponseModel(
            messages_list=message_list,
            user_id=str(from_agent_id),
            mcp_info_list=[],
            other_data=None,
        )
        # XYZ 平台使用 stream 发送信息
        response = await client.send_stream_message(send_message.model_dump_json())
    else:
        # 外部服务使用常规方式发送信息
        response = await client.send_message(send_message.model_dump_json())

    # 缓存历史记录
    await xyz_server.conversation_write(
        user_id=from_agent_id,
        agent_id=to_agent_id,
        message=message,
        role="user",
    )
    await xyz_server.conversation_write(
        user_id=from_agent_id,
        agent_id=to_agent_id,
        message=response,
        role="assistant",
    )

    return f"Agent {to_agent_id} said: {response}"


if __name__ == "__main__":
    # 你好, 我需要获取 1039 的 Card 信息, 超时时间为 1min
    # 我的 AgentID 是 2, 我希望向 1039 发送一条 "你好" 的消息并等待回复. 超时时间为 2min
    mcp.run(transport="sse")
