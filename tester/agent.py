import asyncio

from core.server.xyz_platform import XyzPlatformServer
from core.libs.rpc import RpcManager, AgentClient
from google.protobuf.json_format import MessageToDict

server = XyzPlatformServer()
agent_id = 941


async def chat(agent: AgentClient):
    messages_list = [
        {
            "role": {
                "type": "user",  # 角色类型
                "id": "1234567890",  # 用户 ID
                "name": "Bin Liang",  # 用户名
            },
            "content": "你好，请你自我介绍一下",
            "metadata": {"time": "2025-05-08 15:00:00"},  # 时间戳
        }
    ]

    async for response in agent.run_message_streaming(
        messages_list=messages_list,
        user_id=str(agent_id),
        mcp_info_list=[],
        other_data=None,
    ):
        msg = MessageToDict(response)
        print(msg)
        if msg["type"] == "message_content":
            print(msg["data"]["content"])


async def info(agent: AgentClient):
    info = await agent.get_agent_info()
    print(info)


async def main():
    agent = await RpcManager.get_agent_client(
        agent_id,
    )

    await info(agent)
    await chat(agent)


if __name__ == "__main__":
    asyncio.run(main())
