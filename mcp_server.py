import re
import os
import zlib
import time
import logging
from typing import Optional
from fastmcp import FastMCP
from urllib.parse import urlparse

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from core.a2a.client import A2AClient
from core.models import MakeResponseModel
from core.server import xyz_server
from core.env_helper import EnvHelper


def get_consistent_int_from_url(url_string):
    """
    将 URL 转换为 INT 值.
    """
    # 1. 确保编码一致 (例如，UTF-8)
    url_bytes = url_string.encode("utf-8")
    # 2. 使用相同的哈希算法
    crc_value = zlib.crc32(url_bytes)
    # 3. 确保结果的处理一致 (例如，& 0xffffffff 确保是无符号32位正整数)
    return crc_value & 0xFFFFFFFF


A2A_SERVER_URL = os.getenv("A2A_SERVER_URL", "http://127.0.0.1:5000")

mcp = FastMCP("A2A-Client", host="0.0.0.0", port=EnvHelper.get_mcp_server_port())


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


@mcp.tool()
async def get_agent_card_by_url(url: str):
    """
    Get AgentCard via url.

    Args:
        url (str): get Agent Server agentCard Url.

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

    try:
        async with httpx.AsyncClient(timeout=EnvHelper.get_http_timeout()) as client:
            response = await client.get(url)
            return response.json()
    except Exception as exc:
        return {
            "err": str(exc),
            "message": "Failed to request AgentCard, possibly due to incorrect URL or Agent Serve service problem. Please confirm",
        }


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
        to_agent_id (int | None): Optional, The ID of the target agent to which the message is being sent.

    Returns:
        str: The response content from the target agent.

    Important:
        - If the user only provides the target AgentID but not the URL, the get_agent_card_by_agent_id tool should be called first to obtain the URL of the Call Agent.
        - If the user only provides the URL but not the to_agent_id, it indicates that this is a third-party Agent Server
        - If the user does not provide from_agent_id, the tool cannot be called. Please prompt the user 'Since I don't know who I am, I can't access other agents.'
    """
    client = A2AClient(url=url)

    logging.info(f"""
          调用 A2AClient call_agent 工具, 参数;
              - urL: {url}
              - form_agent_id: {from_agent_id}
              - to_agent_id: {to_agent_id}
          """)

    path = urlparse(url).path  # /1036

    message_list = []
    if to_agent_id:
        # 内部 Agent 需要带上上下文
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

    if to_agent_id:
        # 内部 Agent 需要缓存历史记录
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
