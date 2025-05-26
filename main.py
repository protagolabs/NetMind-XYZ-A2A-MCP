import time
import asyncio
import traceback
import logging
import queue

from google.protobuf.json_format import MessageToDict

from core.a2a.server import (
    BaseXyzA2AServer,
    AgentCard,
    AgentSkill,
    Message,
    AsyncGenerator,
)
from core.libs.rpc import RpcManager, RpcExecutor
from core.models import MakeResponseModel

rpc_executor = RpcExecutor()


class XyzA2AServer(BaseXyzA2AServer):
    """
    适用于 XYZ 的内部 Agent 的 Server 实现.
    由于 gRPC/IO 对多线程下的工作支持并不友好(这是一个已知的 BUG).
    所以我们使用 gRPC 执行器来通过 gRPC/IO 来发送任务. 避免多线程竞争问题.
    """

    def xyz_get_agent_card(self, agent_id: int):
        q = queue.Queue()

        async def _xyz_get_agent_card():
            async with await RpcManager.get_agent_client(agent_id=agent_id) as agent:
                agent_info = await agent.get_agent_info()
                agent_description = agent_info.get("agent_description", "")
                agent_skills = agent_info.get("skills", {})

                skills = []
                for name, info in agent_skills.items():
                    skills.append(
                        AgentSkill(name=name, description=info["description"])
                    )

                q.put(
                    AgentCard(
                        url=f"{self.url}/{agent_id}",
                        name=f"XyzAgent: {agent_id}",
                        description=f"From XYZ platform. Agent: {agent_id}: {agent_description}",
                        capabilities={"streaming": True},
                        skills=[],
                    )
                )

        future = rpc_executor.submit_coroutine(_xyz_get_agent_card())

        while True:
            try:
                item = q.get(timeout=30)

                if item is not None:
                    return item

            except queue.Empty:
                if future.done():
                    exc = future.exception()
                    if exc:
                        logging.error(f"运行 rpc 执行器 err: {exc}")
                        return f"运行 rpc 执行器 err: {exc}"
                else:
                    pass

            time.sleep(0.1)

    async def xyz_stream_response(
        self, agent_id: int, message: Message
    ) -> AsyncGenerator[str, None]:
        q = queue.Queue()

        async def _xyz_stream_response():
            model = MakeResponseModel.model_validate_json(
                json_data=message.content.text
            )
            logging.info(f"收到来自 {model.user_id} 发送给 {agent_id} 的信息 {message}")

            async with await RpcManager.get_agent_client(agent_id=agent_id) as agent:
                try:
                    async for msg in agent.run_message_streaming(
                        messages_list=model.messages_list,
                        user_id=str(model.user_id),
                        mcp_info_list=model.mcp_info_list,
                        other_data=model.other_data,
                    ):
                        msg = MessageToDict(msg)

                        if msg["type"] == "stream_content":
                            content = msg["data"]["content"]
                            q.put(content)
                        else:
                            logging.info(msg)

                except Exception as exc:
                    q.put(traceback.format_exc())
                    logging.error(f"生成回复时错误: {exc}")

            q.put(None)

        future = rpc_executor.submit_coroutine(_xyz_stream_response())

        while True:
            try:
                item = q.get()

                if item is not None:
                    yield item
                else:
                    break

            except queue.Empty:
                if future.done():
                    exc = future.exception()
                    if exc:
                        logging.error(f"运行 rpc 执行器 err: {exc}")
                        break
                else:
                    pass

            await asyncio.sleep(0)

    async def xyz_handle_message(self, agent_id: int, message: Message) -> Message:
        pass


app = XyzA2AServer().load_app()
