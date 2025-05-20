import logging
from python_a2a import (
    A2AClient as StandardClient,
    StreamingClient,
    Message,
    MessageRole,
    TextContent,
)

from core.models import MakeResponseModel


class A2AClient:
    def __init__(self, url: str):
        self.standard_client = StandardClient(url)
        self.streaming_client = StreamingClient(url)

    def send_message(self, message: str) -> str:
        return self.standard_client.send_message(message)

    async def send_stream_message(
        self, model_or_string: MakeResponseModel | str
    ) -> str:
        streaming_text = ""

        if isinstance(model_or_string, MakeResponseModel):
            message = Message(
                content=TextContent(text=model_or_string.model_dump_json()),
                role=MessageRole.USER,
            )
        else:
            message = Message(
                content=TextContent(text=model_or_string), role=MessageRole.USER
            )

        try:
            async for chunk in self.streaming_client.stream_response(message):
                if isinstance(chunk, dict):
                    if "content" in chunk:
                        streaming_text += chunk["content"]
                    elif "text" in chunk:
                        streaming_text += chunk["text"]
                    else:
                        streaming_text += str(chunk)
                else:
                    streaming_text += str(chunk)

        except Exception as exc:
            logging.error(f"收集流式信息错误: {str(exc)}")
            raise exc

        return streaming_text
