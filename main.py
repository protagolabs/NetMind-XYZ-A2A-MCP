import logging
import asyncio
import traceback

from google.protobuf.json_format import MessageToDict

from core.a2a.server import (
    BaseXyzA2AServer,
    AgentSkill,
    AgentCard,
    Message,
    AsyncGenerator,
)
from core.libs.rpc import RpcManager
from core.models import MakeResponseModel


class XyzA2AServer(BaseXyzA2AServer):
    async def xyz_get_agent_card(self, agent_id: int):
        logging.info("开始获取 AgentClient")
        agent = await RpcManager.get_agent_client(agent_id=agent_id)
        logging.info("获取 AgentClient成功")

        try:
            logging.info(
                f"开始获取 Agent Info信息, Agent: {agent}, Id: {agent_id}, Type: {type(agent_id)}"
            )
            agent_info = await agent.get_agent_info()
            agent_description = agent_info.get("agent_description", "")

            agent_skills: dict = agent_info.get("skills", {})
            logging.info(f"AgentInfo 获取成功: {agent_info}")

            skills = []
            for name, info in agent_skills.items():
                skills.append(AgentSkill(name=name, description=info["description"]))

            return AgentCard(
                url=f"{self.url}/{agent_id}",
                name=f"XyzAgent: {agent_id}",
                description=agent_description,
                capabilities={"streaming": True},
                skills=skills,
            )
        except Exception as exc:
            logging.error(f"在获取 AgentCard 时发生错误: {traceback.format_exc()}")
            return AgentCard(
                url=f"{self.url}/{agent_id}",
                name=f"XyzAgent: {agent_id}",
                description="From XYZ platform",
                capabilities={"streaming": True},
                skills=[],
            )

    async def xyz_stream_response(
        self, agent_id: int, message: Message
    ) -> AsyncGenerator[str, None]:
        model = MakeResponseModel.model_validate_json(json_data=message.content.text)
        logging.info(f"收到来自 {model.user_id} 发送给 {agent_id} 的信息 {message}")

        agent = await RpcManager.get_agent_client(agent_id=agent_id)

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
                    yield content
                    await asyncio.sleep(0.1)

        except Exception as exc:
            logging.error(f"生成回复时错误: {exc}")
            raise exc

    async def xyz_handle_message(self, agent_id: int, message: Message) -> Message:
        pass


app = XyzA2AServer().load_app()
