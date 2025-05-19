import re
import os
import httpx
import time
from mcp.server.fastmcp import FastMCP
from urllib.parse import urlparse

from typing import Optional
from core.a2a.client import A2AClient, MakeResponseModel
from tester.memory import RedisMemoryManager

short_term_memory_manager = RedisMemoryManager()


A2A_SERVER_URL = os.getenv("A2A_SERVER_URL", "http://127.0.0.1:5000")

mcp = FastMCP()


@mcp.tool()
async def get_agent_card(agent_id: int):
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
    async with httpx.AsyncClient() as client:
        a2a_agent_url = f"{A2A_SERVER_URL}/.well-known.json"
        response = await client.get(a2a_agent_url, params={"agent_id": agent_id})
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

    print(f"""
          调用 A2AClient call_agent 工具, 参数;
              - urL: {url}
              - form_agent_id: {from_agent_id}
              - to_agent_id: {to_agent_id}
          """)

    path = urlparse(url).path  # /1036

    message_list = await short_term_memory_manager.get_memory(
        user_id=str(from_agent_id), agent_id=str(to_agent_id)
    )

    rest_message = {
        "role": {"id": from_agent_id, "type": "user", "name": "Default User"},
        "content": message,
        "metadata": {"time": time.time()},
    }

    message_list.append(rest_message)

    print(f"发送消息: {rest_message}")

    if re.match(r"^/\d+$", path):
        # 内部 Agent-Server 发送方式
        model = MakeResponseModel(
            messages_list=message_list,
            user_id=str(from_agent_id),
            mcp_info_list=[],
            other_data=None,
        )
        print("已经成功通过内部方式发送")
        response = await client.send_stream_message(model)
    else:
        print("已经成功通过外部方式发送")
        response = await client.send_message(message)

    resp_message = {
        "role": {
            "type": "assistant",
            "id": to_agent_id,
            "name": "Default Agent",
        },
        "content": response,
        "metadata": {"time": time.time()},
    }

    print(f"收到消息: {resp_message}")

    # 缓存历史记录
    await short_term_memory_manager.add_to_memory(
        from_agent_id, to_agent_id, rest_message
    )
    await short_term_memory_manager.add_to_memory(
        from_agent_id, to_agent_id, resp_message
    )

    return f"Agent {to_agent_id} said: {response}"


if __name__ == "__main__":
    # 你好, 我需要获取 1039 的 Card 信息, 超时时间为 1min
    # 我的 AgentID 是 2, 我希望向 1039 发送一条 "你好" 的消息并等待回复. 超时时间为 2min
    mcp.run(transport="sse")
