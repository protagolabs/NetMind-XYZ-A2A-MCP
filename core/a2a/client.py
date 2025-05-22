import logging
from python_a2a import (
    A2AClient as StandardClient,
    StreamingClient,
    Message,
    MessageRole,
    TextContent,
)


class A2AClient:
    def __init__(self, url: str):
        self.standard_client = StandardClient(url)
        self.streaming_client = StreamingClient(url)

    def send_message(self, message: str) -> str:
        message = Message(
            content=TextContent(text=message),
            role=MessageRole.USER,
        )

        return self.standard_client.send_message(message)

    async def send_stream_message(self, message: str) -> Message:
        streaming_text = ""

        message = Message(
            content=TextContent(text=message),
            role=MessageRole.USER,
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

        return Message(content=TextContent(text=streaming_text), role=MessageRole.AGENT)
