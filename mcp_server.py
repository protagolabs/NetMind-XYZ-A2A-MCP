import re
import os
import zlib
import time
import logging
from typing import Optional
from fastmcp import FastMCP
from urllib.parse import urlparse

import httpx
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from core.a2a.client import A2AClient, Message
from core.models import MakeResponseModel
from core.server import xyz_server
from core.env_helper import EnvHelper


A2A_SERVER_URL = os.getenv("A2A_SERVER_URL", "http://127.0.0.1:5000")

mcp = FastMCP("A2A-Client", host="0.0.0.0", port=EnvHelper.get_mcp_server_port())


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


@mcp.tool()
async def get_agent_card_by_url(url: str = Field(description="Agent Server URL")):
    """
    Get AgentCard by server base url.

    Args:
        url (str): Agent-serve url.

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

    Important:
        - URL only supports BASE_URL, such as http://127.0.0.1:8000 or http://127.0.0.1:8000/
        - If the URL has a path part, you should call `get_agent_card_by_agent_id` to get the AgentCard instead of this tool
    """

    try:
        async with httpx.AsyncClient(timeout=EnvHelper.get_http_timeout()) as client:
            if not url.endswith("/"):
                agent_server_url = f"{url}/.well-known.json"
            else:
                agent_server_url = f"{url}.well-known.json"

            logging.info(f"根据 URL 获取 Card 信息: {agent_server_url}")
            response = await client.get(agent_server_url)
            return response.json()
    except Exception as exc:
        return {
            "err": str(exc),
            "message": "Failed to request AgentCard, possibly due to incorrect URL or Agent Serve service problem. Please confirm",
        }


@mcp.tool()
async def get_agent_card_by_agent_id(
    agent_id: int = Field(
        description="AgentCard Unique identifier of the person to whom it belongs"
    ),
):
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
    try:
        logging.info(f"根据 AgentID 获取 Card 信息: {agent_id}")
        async with httpx.AsyncClient(timeout=EnvHelper.get_http_timeout()) as client:
            agent_server_url = f"{A2A_SERVER_URL}/.well-known.json"
            response = await client.get(agent_server_url, params={"agent_id": agent_id})
            return response.json()
    except Exception as exc:
        return {
            "err": str(exc),
            "message": "Failed to request AgentCard, Check agentID is valid",
        }


@mcp.tool()
async def call_agent_by_agent_url(
    url: str = Field(
        description="The complete HTTP or HTTPS URL of the target external Agent-server. This URL is used to directly send a message to an agent operating outside the internal XYZ system."
    ),
    message: str = Field(
        description="The message content to be sent to the target Agent-server. This should be a clear, concise piece of information or a question intended for the external agent."
    ),
):
    """Sends a message to an external Agent-server using its specific URL via the A2A (Agent-to-Agent) protocol.

    This tool is intended for communication with Agent-servers that are addressable via a direct URL
    and are considered external to the XYZ internal agent network. It performs a straightforward
    message transmission without including conversational context from the XYZ system.

    Args:
        url: The fully qualified URL of the external Agent-server (e.g., 'https://example-agent.com').
        message: The textual message to be delivered to the Agent-server at the specified URL.

    Returns:
        A string containing the response from the target Agent-server.
    """

    logging.info(f"""
          调用 A2AClient call_agent 工具, 参数;
              - urL: {url}
              - message: {message}
          """)

    client = A2AClient(url=url)

    # 外部服务使用常规方式发送信息
    response = client.send_message(message)
    logging.info(f"{url} Response: {response}")
    return response.content.text


@mcp.tool()
async def call_agent_by_agent_id(
    from_agent_id: int = Field(
        description="The unique identifier of the XYZ Agent initiating this call. This ID is used to fetch the relevant conversation history and identify the sender in the message."
    ),
    message: str = Field(
        description="The new message content to be sent to the target internal XYZ Agent. This message will be appended to the existing conversation history before sending."
    ),
    to_agent_id: int = Field(
        description="The unique identifier of the target XYZ Agent within the internal network to which the message and conversation history will be sent.",
    ),
):
    """Sends a message to an internal XYZ Agent using its Agent ID, including conversation context, via the A2A protocol.

    This tool facilitates communication between Agents within the XYZ internal network. It automatically
    retrieves the existing conversation history between the `from_agent_id` (acting as user) and
    the `to_agent_id`. The new `message` is appended to this history, and the entire context is sent
    to the target agent. Both the sent message and the received response are then saved back into the
    conversation history.

    Args:
        from_agent_id: The ID of the agent making the request (e.g., an agent acting on behalf of a user).
        message: The current message to send to the `to_agent_id`.
        to_agent_id: The ID of the internal XYZ Agent that should receive the message.

    Returns:
        A string containing the response from the target Agent-server.
    """
    agent_server_url = f"{A2A_SERVER_URL}/{to_agent_id}"

    client = A2AClient(url=agent_server_url)

    logging.info(f"""
          调用 A2AClient call_agent 工具, 参数;
              - urL: {agent_server_url}
              - form_agent_id: {from_agent_id}
              - to_agent_id: {to_agent_id}
              - message: {message}
          """)

    message_list = []

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

    # 内部 Agent-Server 发送方式
    send_message = MakeResponseModel(
        messages_list=message_list,
        user_id=str(from_agent_id),
        mcp_info_list=[],
        other_data=None,
    )

    response: Message = await client.send_stream_message(send_message.model_dump_json())

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
        message=response.content.text,
        role="assistant",
    )

    logging.info(f"{to_agent_id} Response: {response}")
    return response.content.text


if __name__ == "__main__":
    mcp.run(transport="sse")
