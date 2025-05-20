""" 
Stream decoder for OpenAI Agents SDK events.
This module provides functionality to decode streaming events from OpenAI Agents SDK
into structured formats for downstream processing.
"""

import traceback
from pydantic import BaseModel 
from enum import Enum 
from typing import Dict, Optional, AsyncIterator, Any, Union
import json
import re

from src.model_price import model_pricing

""" 
后端同事使用的时候，请先看 "StreamDecoder" 的定义，然后根据需要选择合适的 decoder 类型。

是否要 stream_show, 请判断这个字段是不是 true

PS：
- 每次调用的时候都会先给一个 AGENT_CALL 的 decoder，这个是 debug 用的，可以获取到当前 agent 的配置信息。
    - 信息储存在 new_agent 字段，每次交互 只会显示一次

- 其他的 EVENT 会根据 EVENT 的类型，选择合适的 data 字段的类型，然后根据需要获取。

- 暂时请忽略掉 AGENT_CALL 类型，因为只是 debug，不需要给用户展示。
"""

class StreamingStateType(Enum):
    START = "start" # In each time of start streaming, we will give a start event
    END = "end" # In each time of end streaming, we will give a end event
    STREAMING = "streaming" # In each time of streaming, we will give a streaming event
    NO_STREAMING = "no_streaming" # In each time of no streaming, we will give a no streaming event
    
class XYZAgentOutputDecoderType(Enum):
    STREAM_CONTENT = "stream_content"
    MESSAGE_CONTENT = "message_content"
    REASONING_CONTENT= "reasoning_content"
    TOOL_USE = "tool_use"
    MCP_SERVER_USE = "mcp_server_use"
    AGENT_CALL = "agent_call"
    XYZ_AGENT_POWER = "xyz_agent_power"
    USAGE_REPORT = "usage_report"
    SPECIAL_SIGNAL = "special_signal"

class UsageReport(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the UsageReport to a dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost
        }

class BasicXYZOutput(BaseModel):
    
    content: Optional[str] = None 
    
    mcp_server_name: Optional[str] = None
    mcp_tool_name: Optional[str] = None
    mcp_tool_arguments: Optional[Dict] = None
    mcp_tool_response: Optional[Union[str, Dict]] = None  # Accept either string or dict 
    
    agent_name: Optional[str] = None
    agent_arguments: Optional[Dict] = None
    interaction_chain: Optional[list[str]] = None
    
    power_name: Optional[str] = None
    power_arguments: Optional[Dict] = None
    power_response: Optional[Union[str, Dict]] = None  # Accept either string or dict 
    
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict] = None
    tool_response: Optional[Union[str, Dict]] = None  # Accept either string or dict 
    
    usage_report: Optional[UsageReport] = None
   
    signal_type: Optional[Enum] = None 
    signal_data: Optional[Enum] = None
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the output to a dictionary, filtering out None values."""
        result = {}
        for key, value in self.dict().items():
            if value is not None:
                if key == "usage_report" and self.usage_report is not None:
                    result[key] = self.usage_report.to_dict()
                else:
                    result[key] = value
        return result

class StreamOutput(BasicXYZOutput):
    content: str 

class MessageOutput(BasicXYZOutput):
    content: str 
    
class ReasoningOutput(BasicXYZOutput):
    content: str 
    
class ToolUseOutput(BasicXYZOutput):
    tool_name: str 
    tool_arguments: Dict 
    tool_response: str
    
class MCPServerUseOutput(BasicXYZOutput):
    mcp_server_name: str 
    mcp_tool_name: str 
    mcp_tool_arguments: Dict 
    mcp_tool_response: str 
    
class AgentCallOutput(BasicXYZOutput):
    agent_name: str 
    agent_arguments: Dict 
    interaction_chain: list[str]
    
class XYZAgentPowerOutput(BasicXYZOutput):
    power_name: str 
    power_arguments: Dict 
    power_response: str 
    
class UsageReportOutput(BasicXYZOutput):
    usage_report: UsageReport
    
class ShowButtonType(Enum):
    TOOLS = "tools"
    ROLE_TEMPLATES = "role_templates"
    LLM_MODELS = "llm_models"
    
class SpecialSignalType(Enum):
    SHOW_BUTTON = "show_button"

class SpecialSignalOutput(BasicXYZOutput):
    signal_type: SpecialSignalType
    signal_data: ShowButtonType

class StreamDecoder(BaseModel):
    type: XYZAgentOutputDecoderType
    data: BasicXYZOutput  # Pydantic will automatically accept subclasses
    stream_show: bool = False
    new_agent: Optional[Any] = None
    streaming_state: StreamingStateType = StreamingStateType.NO_STREAMING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the StreamDecoder to a clean dictionary format."""
        return {
            "type": self.type.value,
            "data": self.data.to_dict()
        }
    
    def to_json(self, indent: int = 4) -> str:
        """Convert to a formatted JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

class StreamDecoderFactory:
    @staticmethod
    def create_for_output_type(
        output_type: XYZAgentOutputDecoderType, 
        stream_show: bool = False,
        new_agent: Optional[Any] = None,
        streaming_state: StreamingStateType = StreamingStateType.NO_STREAMING,
        **kwargs
    ) -> StreamDecoder:
        """
        Create appropriate output object based on the output type and wrap it in a decoder.
        
        Args:
            output_type: Type of the output to create
            stream_show: Whether to show streaming content
            new_agent: Optional agent object
            **kwargs: Additional arguments for the output object
            
        Returns:
            StreamDecoder: A decoder containing the created output object
        """
        # Create appropriate output object based on type
        stream_show_override = False
        if output_type == XYZAgentOutputDecoderType.STREAM_CONTENT:
            data = StreamOutput(**kwargs)
            stream_show_override = True
        elif output_type == XYZAgentOutputDecoderType.MESSAGE_CONTENT:
            data = MessageOutput(**kwargs)
        elif output_type == XYZAgentOutputDecoderType.REASONING_CONTENT:
            data = ReasoningOutput(**kwargs)
        elif output_type == XYZAgentOutputDecoderType.TOOL_USE:
            data = ToolUseOutput(**kwargs)
        elif output_type == XYZAgentOutputDecoderType.MCP_SERVER_USE:
            data = MCPServerUseOutput(**kwargs)
        elif output_type == XYZAgentOutputDecoderType.AGENT_CALL:
            data = AgentCallOutput(**kwargs)
        elif output_type == XYZAgentOutputDecoderType.XYZ_AGENT_POWER:
            data = XYZAgentPowerOutput(**kwargs)
        elif output_type == XYZAgentOutputDecoderType.USAGE_REPORT:
            data = UsageReportOutput(**kwargs)
        elif output_type == XYZAgentOutputDecoderType.SPECIAL_SIGNAL:
            data = SpecialSignalOutput(**kwargs)
        else:
            data = BasicXYZOutput(**kwargs)
            
        return StreamDecoder(
            type=output_type, 
            data=data, 
            stream_show=stream_show_override if stream_show_override else stream_show,
            new_agent=new_agent,
            streaming_state=streaming_state
        )

def compute_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Compute the cost of an LLM call based on the model name and token counts.
    
    Args:
        model_name: The name of the model used for the LLM call
        prompt_tokens: The number of prompt tokens used in the call
        completion_tokens: The number of completion tokens used in the call
        
    Returns:
        float: The cost of the LLM call in USD
    """
    # Current OpenAI pricing as of May 2024 (USD per 1M tokens)
    if model_name.startswith("gpt-o4"):
        pricing = model_pricing["o4-mini"]
    elif model_name.startswith("gpt-o3"):
        pricing = model_pricing["o3-mini"]
    elif model_name.startswith("gpt-o1"):
        pricing = model_pricing["o1-mini"]
    elif model_name.startswith("gpt-4o-mini"):
        pricing = model_pricing["gpt-4o-mini"]
    elif model_name.startswith("gpt-4o"):
        pricing = model_pricing["gpt-4o"]
    else:
        return 0.0
    
    # Calculate cost in USD
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    return total_cost

async def extract_mcp_and_tools_from_event(event: Any) -> Dict[str, list]:
    """
    Extract all MCPs and their tools, as well as normal tools from an agent object.
    
    Args:
        event: The event object containing the agent data
        
    Returns:
        Dict[str, list]: A dictionary with MCPs and their tools, and normal tools
    """
    mcp_tools_dict = {
        "normal_tools": [],
        "show_button": []
    }
    
    if not hasattr(event, "new_agent"):
        return mcp_tools_dict
        
    agent_data = event.new_agent
    
    # Extract normal tools
    if hasattr(agent_data, "tools") and agent_data.tools:
        for tool in agent_data.tools:
            if tool.name in ['show_xyz_platform_tools_button', 'show_agent_configing_templates_button', 'show_llm_selection_button']:
                mcp_tools_dict["show_button"].append(tool.name)
            else:
                mcp_tools_dict["normal_tools"].append(tool.name)
    
    # Extract MCPs and their tools
    if hasattr(agent_data, "mcp_servers") and agent_data.mcp_servers:
        for mcp_server in agent_data.mcp_servers:
            local_tools = await mcp_server.list_tools()
            tools_name = [tool.name for tool in local_tools]
            mcp_tools_dict[mcp_server.name] = tools_name
    
    return mcp_tools_dict

def find_tool_mcp(mcp_tools_dict: Dict[str, list], tool_name: str) -> Optional[str]:
    """
    Find which MCP a tool belongs to.
    
    Args:
        mcp_tools_dict: Dictionary returned by extract_mcp_and_tools function
        tool_name: Name of the tool to look for
        
    Returns:
        Optional[str]: Name of the MCP the tool belongs to, or "normal_tools" if it's a normal tool,
                      or None if the tool is not found
    """
    # Check in each MCP and normal_tools
    for mcp_name, tools in mcp_tools_dict.items():
        if tool_name in tools:
            return mcp_name
    
    return None

async def process_agent_updated_event(event: Any) -> StreamDecoder:
    """
    Process the AgentUpdatedStreamEvent and return a StreamDecoder.
    
    Args:
        event: The AgentUpdatedStreamEvent to process
        
    Returns:
        StreamDecoder: A decoder with agent information
    """
    agent_name = ""
    agent_instructions = ""
    agent_model = ""
    agent_tools = []
    
    # Extract agent information
    if hasattr(event, "new_agent"):
        agent = event.new_agent
        if hasattr(agent, "name"):
            agent_name = agent.name
        if hasattr(agent, "instructions"):
            agent_instructions = agent.instructions
        if hasattr(agent, "model"):
            agent_model = agent.model
        if hasattr(agent, "tools"):
            agent_tools = [tool.name for tool in agent.tools if hasattr(tool, "name")]
    
    # Create decoder with agent information            
    return StreamDecoderFactory.create_for_output_type(
        XYZAgentOutputDecoderType.AGENT_CALL,
        new_agent=event.new_agent,
        agent_name=agent_name,
        agent_arguments={
            "instructions": agent_instructions,
            "model": agent_model,
            "available_tools": agent_tools
        },
        interaction_chain=[]
    )

def process_response_completed_event(event: Any) -> Optional[StreamDecoder]:
    """
    Process the ResponseCompletedEvent and extract usage information.
    
    Args:
        event: The ResponseCompletedEvent to process
        
    Returns:
        Optional[StreamDecoder]: A decoder with usage information or None if usage data not available
    """
    usage_data = None
    model_name = ""
    
    # Extract usage data and model name
    if hasattr(event, "data") and hasattr(event.data, "response"):
        response_obj = event.data.response
        if hasattr(response_obj, "model"):
            model_name = response_obj.model
        if hasattr(response_obj, "usage") and response_obj.usage is not None:
            usage = response_obj.usage
            prompt_tokens = getattr(usage, "input_tokens", 0)
            completion_tokens = getattr(usage, "output_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
            cost = compute_cost(model_name, prompt_tokens, completion_tokens)
            usage_data = UsageReport(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost
            )
    
    if usage_data:
        return StreamDecoderFactory.create_for_output_type(
            XYZAgentOutputDecoderType.USAGE_REPORT,
            usage_report=usage_data
        )
    
    return None

def process_text_done_event(event: Any) -> Optional[StreamDecoder]:
    """
    Process the ResponseTextDoneEvent or TextDoneEvent and extract the full text message.
    
    Args:
        event: The event to process
        
    Returns:
        Optional[StreamDecoder]: A decoder with the full message content or None if no text found
    """
    full_text = ""
    
    # Extract text content from ResponseTextDoneEvent
    if hasattr(event, "data") and hasattr(event.data, "text"):
        full_text = event.data.text
    # Extract text content from TextDoneEvent
    elif hasattr(event, "text"):
        full_text = event.text
    else:
        # Try to extract text from string representation
        text_match = re.search(r"text='([^']*)'", str(event))
        if text_match:
            full_text = text_match.group(1)
    
    if full_text:
        return StreamDecoderFactory.create_for_output_type(
            XYZAgentOutputDecoderType.MESSAGE_CONTENT,
            content=full_text
        )
    
    return None

def process_text_delta_event(event: Any) -> Optional[StreamDecoder]:
    """
    Process the ResponseTextDeltaEvent or TextDelta event and extract the delta content.
    
    Args:
        event: The event to process
        
    Returns:
        Optional[StreamDecoder]: A decoder with the delta content or None if no content found
    """
    content = ""
    
    # Extract delta content
    if hasattr(event, "data") and hasattr(event.data, "delta"):
        content = event.data.delta
    elif hasattr(event, "delta"):
        content = event.delta
    
    if content:
        return StreamDecoderFactory.create_for_output_type(
            XYZAgentOutputDecoderType.STREAM_CONTENT,
            streaming_state=StreamingStateType.STREAMING,
            content=content
        )
    
    return None

def process_tool_call_event(event: Any) -> Optional[StreamDecoder]:
    """
    Process the tool call event and extract the tool name and arguments.
    
    Args:
        event: The tool call event to process
        
    Returns:
        Optional[StreamDecoder]: A decoder with tool call information or None if no tool name found
    """
    tool_name = ""
    tool_args = {}
    
    # Extract tool name and arguments
    if hasattr(event, "item") and hasattr(event.item, "name"):
        tool_name = event.item.name
        if hasattr(event.item, "input"):
            tool_args = event.item.input
    
    if tool_name:
        return StreamDecoderFactory.create_for_output_type(
            XYZAgentOutputDecoderType.TOOL_USE,
            tool_name=tool_name,
            tool_arguments=tool_args,
            tool_response=""
        )
    
    return None

def process_raw_tool_call_event(event: Any) -> tuple[str, str, Dict]:
    """
    Process a raw tool call event and extract call ID, tool name and arguments.
    
    Args:
        event: The raw tool call event to process
        
    Returns:
        tuple: A tuple containing (call_id, tool_name, tool_arguments)
    """
    tool_name = ""
    tool_args = {}
    call_id = ""
    
    # Extract tool name, arguments and call ID
    if hasattr(event, "item") and hasattr(event.item, "raw_item"):
        raw_item = event.item.raw_item
        if hasattr(raw_item, "name"):
            tool_name = raw_item.name
        if hasattr(raw_item, "arguments"):
            try:
                tool_args = json.loads(raw_item.arguments)
            except (json.JSONDecodeError, TypeError):
                tool_args = {"raw_arguments": str(raw_item.arguments)}
        if hasattr(raw_item, "call_id"):
            call_id = raw_item.call_id
    
    return call_id, tool_name, tool_args

def process_tool_output_event(event: Any) -> Optional[StreamDecoder]:
    """
    Process the tool output event and extract the tool name and response.
    
    Args:
        event: The tool output event to process
        
    Returns:
        Optional[StreamDecoder]: A decoder with tool output information or None if no relevant info found
    """
    tool_name = ""
    tool_response = ""
    
    # Extract tool name and response
    if hasattr(event, "item"):
        if hasattr(event.item, "tool_name"):
            tool_name = event.item.tool_name
        if hasattr(event.item, "output"):
            tool_response = event.item.output
    
    if tool_name or tool_response:
        return StreamDecoderFactory.create_for_output_type(
            XYZAgentOutputDecoderType.TOOL_USE,
            tool_name=tool_name,
            tool_arguments={},
            tool_response=str(tool_response)
        )
    
    return None

def classify_tool_output_event(tools_data: Dict[str, Any], mcp_tools_dict: Dict[str, list]) -> Optional[StreamDecoder]:
    """
    Classify the tool output event and return a StreamDecoder.
    
    Args:
        tools_data: Dictionary containing tool data (name, arguments, response)
        mcp_tools_dict: Dictionary mapping MCPs to their tools
        
    Returns:
        Optional[StreamDecoder]: A decoder for the classified tool event or None if classification fails
    """
    if not tools_data.get("tool_name"):
        return None
        
    mcp_name = find_tool_mcp(mcp_tools_dict, tools_data["tool_name"])
    
    if mcp_name == "normal_tools":
        return StreamDecoderFactory.create_for_output_type(
            XYZAgentOutputDecoderType.XYZ_AGENT_POWER,
            power_name=tools_data["tool_name"],
            power_arguments=tools_data["tool_arguments"],
            power_response=tools_data["tool_response"]
        )
    elif mcp_name == "show_button":
        signal_data = None
        if tools_data["tool_name"] == "show_xyz_platform_tools_button":
            signal_data = ShowButtonType.TOOLS
        elif tools_data["tool_name"] == "show_agent_configing_templates_button": 
            signal_data = ShowButtonType.ROLE_TEMPLATES 
        elif tools_data["tool_name"] == "show_llm_selection_button": 
            signal_data = ShowButtonType.LLM_MODELS
            
        if signal_data:
            return StreamDecoderFactory.create_for_output_type(
                XYZAgentOutputDecoderType.SPECIAL_SIGNAL,
                signal_type=SpecialSignalType.SHOW_BUTTON,
                signal_data=signal_data
            )
    elif mcp_name:
        return StreamDecoderFactory.create_for_output_type(
            XYZAgentOutputDecoderType.MCP_SERVER_USE,
            mcp_server_name=mcp_name,
            mcp_tool_name=tools_data["tool_name"],
            mcp_tool_arguments=tools_data["tool_arguments"],
            mcp_tool_response=tools_data["tool_response"]
        )
    
    return None

def process_raw_tool_output_event(event: Any) -> tuple[str, str]:
    """
    Process a raw tool output event and extract call ID and output.
    
    Args:
        event: The raw tool output event to process
        
    Returns:
        tuple: A tuple containing (call_id, output)
    """
    call_id = ""
    output = ""
    
    # Extract output and call ID
    if hasattr(event, "item") and hasattr(event.item, "raw_item"):
        raw_item = event.item.raw_item
        if isinstance(raw_item, dict):
            call_id = raw_item.get("call_id", "")
            output = raw_item.get("output", "")
        else:
            if hasattr(raw_item, "call_id"):
                call_id = raw_item.call_id
            if hasattr(raw_item, "output"):
                output = raw_item.output
    
    return call_id, output

def process_usage_info_event(event: Any) -> Optional[StreamDecoder]:
    """
    Process an event with usage information and create a usage report.
    
    Args:
        event: The event with usage information
        
    Returns:
        Optional[StreamDecoder]: A decoder with usage report or None if insufficient data
    """
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    model_name = ""
    
    # Try to extract model name
    if hasattr(event, "data") and hasattr(event.data, "response") and hasattr(event.data.response, "model"):
        model_name = event.data.response.model
    
    # Extract usage information - handle different object structures
    if hasattr(event, "usage"):
        usage = event.usage
        if hasattr(usage, "input_tokens"):
            prompt_tokens = usage.input_tokens
        if hasattr(usage, "output_tokens"):
            completion_tokens = usage.output_tokens
        if hasattr(usage, "total_tokens"):
            total_tokens = usage.total_tokens
    # Extract from nested response structure
    elif hasattr(event, "data") and hasattr(event.data, "response"):
        response = event.data.response
        if hasattr(response, "usage") and response.usage is not None:
            usage = response.usage
            prompt_tokens = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
    # Try to extract from raw item structure
    elif hasattr(event, "item") and hasattr(event.item, "raw_item"):
        raw_item = event.item.raw_item
        if isinstance(raw_item, dict) and "usage" in raw_item:
            usage = raw_item["usage"]
            prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
        elif hasattr(raw_item, "usage") and raw_item.usage is not None:
            usage = raw_item.usage
            prompt_tokens = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
    
    # Only proceed if we have token data
    if prompt_tokens > 0 or completion_tokens > 0:
        cost = compute_cost(model_name, prompt_tokens, completion_tokens)
        
        return StreamDecoderFactory.create_for_output_type(
            XYZAgentOutputDecoderType.USAGE_REPORT,
            usage_report=UsageReport(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost
            )
        )
    
    return None

async def process_event_stream(
    stream_events: AsyncIterator, 
) -> AsyncIterator[StreamDecoder]:
    """
    Process and decode events from OpenAI Agents SDK event stream.
    
    This function processes the event stream, decodes each event, and yields decoded events.
    It handles deduplication of messages and usage events.
    
    Args:
        stream_events: Async iterator of events from OpenAI Agents SDK
    
    Yields:
        StreamDecoder: Decoded stream events
    
    Example:
    ```python
    from mxyz_agent_core.factory.mcp_factory import MCPFactory
    from mxyz_agent_core.a2b_modules.stream_decoder import process_event_stream, XYZAgentOutputDecoderType
    
    async def main():
        db_mcp = MCPFactory.get_admin_mcp_server()
        await db_mcp.connect()
        agent = Agent(
            name="HR",
            instructions="You are a helpful assistant.",
            model="gpt-4o-2024-11-20",
            mcp_servers=[db_mcp],
            tools=[get_user_info],
        )
    
        result = Runner.run_streamed(agent, input="你好啊？")
        
        async for event in process_event_stream(result):
            if event.type == XYZAgentOutputDecoderType.MESSAGE_CONTENT:
                print(f"Final message: {event.data.content}")
                
        await db_mcp.cleanup()
    ```
    """
    
    mcp_tools_dict = {}
    # Store pending tool calls
    pending_tool_calls = {}
    # Track processed usage events to avoid duplicates
    processed_usage_events = set()
    # Track processed message content to avoid duplicates
    processed_messages = set()
    streaming = False
    
    try:
        async for response in stream_events.stream_events():
            # Extract string representation for analysis
            response_str = str(response)
            
            # Compute hash for deduplication
            response_hash = hash(response_str)
            
            # Handle AgentUpdatedStreamEvent
            if "AgentUpdatedStreamEvent" in response_str:
                try:
                    mcp_tools_dict = await extract_mcp_and_tools_from_event(response)
                    decoder = await process_agent_updated_event(response)
                    yield decoder
                except Exception as e:
                    # Log error but continue processing other events
                    print(f"Error processing agent updated event: {e}")
                    continue
                    
            # Handle ResponseCompletedEvent with usage info
            elif "ResponseCompletedEvent" in response_str and "usage" in response_str:
                if response_hash not in processed_usage_events:
                    try:
                        decoder = process_response_completed_event(response)
                        if decoder:
                            processed_usage_events.add(response_hash)
                            yield decoder
                    except Exception as e:
                        print(f"Error processing usage event: {e}")
        
            # Handle text completion events
            elif ("ResponseTextDoneEvent" in response_str or "TextDoneEvent" in response_str and "text=" in response_str):
                try:
                    decoder = process_text_done_event(response)
                    if decoder and decoder.type == XYZAgentOutputDecoderType.MESSAGE_CONTENT:
                        content = decoder.data.content
                        message_hash = hash(content) if content else 0
                        if message_hash and message_hash not in processed_messages:
                            processed_messages.add(message_hash)
                            decoder = StreamDecoderFactory.create_for_output_type(
                                XYZAgentOutputDecoderType.MESSAGE_CONTENT,
                                content=content
                            )
                            yield decoder
                    elif decoder:
                        yield decoder
                except Exception as e:
                    print(f"Error processing text done event: {e}")
            
            # Handle tool call events
            elif "RunItemStreamEvent" in response_str and "ToolCallItem" in response_str:
                try:
                    call_id, tool_name, tool_args = process_raw_tool_call_event(response)
                    
                    if call_id and tool_name:
                        pending_tool_calls[call_id] = {
                            "tool_name": tool_name,
                            "tool_arguments": tool_args,
                            "tool_response": None
                        }
                    else:
                        decoder = process_tool_call_event(response)
                        if decoder:
                            yield decoder
                except Exception as e:
                    print(f"Error processing tool call event: {e}")
            
            # Handle tool output events
            elif "RunItemStreamEvent" in response_str and "ToolCallOutputItem" in response_str:
                try:
                    call_id, output = process_raw_tool_output_event(response)
                    
                    if call_id and call_id in pending_tool_calls:
                        tool_data = pending_tool_calls[call_id]
                        tool_data["tool_response"] = output
                        
                        decoder = classify_tool_output_event(tool_data, mcp_tools_dict)
                        if decoder:
                            # Remove processed call
                            del pending_tool_calls[call_id]
                            yield decoder
                        else:
                            # Create generic tool use decoder if classification failed
                            combined_decoder = StreamDecoderFactory.create_for_output_type(
                                XYZAgentOutputDecoderType.TOOL_USE,
                                tool_name=tool_data["tool_name"],
                                tool_arguments=tool_data["tool_arguments"],
                                tool_response=output
                            )
                            
                            # Remove processed call
                            del pending_tool_calls[call_id]
                            yield combined_decoder
                    else:
                        # Process standalone tool output
                        decoder = process_tool_output_event(response)
                        if decoder:
                            yield decoder
                except Exception as e:
                    print(f"Error processing tool output event: {e}")
            
            # Handle message created events
            elif "RunItemStreamEvent" in response_str and "message_output_created" in response_str:
                try:
                    if hasattr(response, "item") and hasattr(response.item, "raw_item"):
                        raw_item = response.item.raw_item
                        if hasattr(raw_item, "content"):
                            content_list = raw_item.content
                            for content_item in content_list:
                                if hasattr(content_item, "text") and content_item.text:
                                    message_hash = hash(content_item.text)
                                    if message_hash not in processed_messages:
                                        processed_messages.add(message_hash)
                                        decoder = StreamDecoderFactory.create_for_output_type(
                                            XYZAgentOutputDecoderType.MESSAGE_CONTENT,
                                            content=content_item.text
                                        )
                                        yield decoder
                except Exception as e:
                    error_message = traceback.format_exc()
                    raise Exception(f"Error processing message created event: {error_message}")
                        # Handle text streaming events
            if "delta" in response_str and ("ResponseTextDeltaEvent" in response_str or "TextDelta" in response_str):
                try:
                    if not streaming:
                        streaming = True
                        yield StreamDecoderFactory.create_for_output_type(
                            XYZAgentOutputDecoderType.STREAM_CONTENT,
                            stream_show=True,
                            streaming_state=StreamingStateType.START,
                            content="",
                        )
                    decoder = process_text_delta_event(response)
                    if decoder:
                        yield decoder
                except Exception as e:
                    print(f"Error processing text delta event: {e}")
            else:
                if streaming:
                    streaming = False
                    yield StreamDecoderFactory.create_for_output_type(
                        XYZAgentOutputDecoderType.STREAM_CONTENT,
                        stream_show=True,
                        streaming_state=StreamingStateType.END,
                        content="",
                    )
    except Exception as e:
        error_message = traceback.format_exc()
        # Re-raise to allow caller to handle the error
        raise Exception(f"Unexpected error in stream processing: {error_message}")

