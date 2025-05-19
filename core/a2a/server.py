# 这是一个用于扩展标准 A2A 协议的方法
# 适用于动态的根据 AgentID 来运行 A2AServer
# 他避免了为每一个单独的 Agent 启动一个 Server 的繁琐步骤
# 当然, 这必须依赖项目本身已经具有了 Target Agent Search 的功能 ...

import abc
import sys
import selectors
import time
import json
import logging
import typing
import asyncio
import threading
from queue import Queue, Empty
from typing import Any, AsyncGenerator, Union

from python_a2a.server import A2AServer
from python_a2a.server.ui_templates import JSON_HTML_TEMPLATE
from python_a2a.models import AgentCard
from python_a2a.models import TaskState, TaskStatus
from python_a2a.models.message import Message
from python_a2a.models.conversation import Conversation
from flask import (
    Flask,
    Response,
    jsonify,
    request,
    make_response,
    render_template_string,
    g,
)


def run_coroutine_thread(coro: typing.Coroutine):
    """
    Run a coroutine in a separate thread with its own event loop.
    Returns the result of the coroutine or raises an exception if it fails.

    Args:
        coro: The coroutine to run

    Returns:
        The result of the coroutine

    Raises:
        TimeoutError: If the coroutine doesn't complete within the timeout
        Exception: If the coroutine raises an exception
    """
    queue = Queue()
    done_event = threading.Event()

    def run_thread():
        """Run the coroutine in a dedicated thread with its own event loop."""
        # Create a new event loop for this thread
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the coroutine and put the result in the queue
            result = loop.run_until_complete(coro)
            queue.put({"result": result})
        except Exception as exc:
            # Put the exception in the queue
            logging.error(
                f"Exception in run_thread, coro is {coro.__qualname__}: {exc}",
                exc_info=True,
            )
            queue.put({"error": str(exc)})
        finally:
            # Make sure to close all running tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()

            # Run the event loop until all tasks are cancelled
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )

            # Clean up the loop and signal we're done
            loop.close()
            done_event.set()

    # Start the thread
    thread = threading.Thread(target=run_thread)
    thread.daemon = True
    thread.start()

    run_result = queue.get(block=True)

    err = run_result.get("error")
    if not err:
        return run_result["result"]

    logging.error(err)
    raise Exception(err)


class BaseXyzA2AServer(A2AServer):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5000,
        debug=False,
        google_a2a_compatible: bool = True,
    ):
        self.host = host
        self.port = port
        self.debug = debug

        self.url = f"http://{self.host}:{self.port}"

        # Initialize task storage
        self.tasks = {}

        # Initialize streaming subscriptions
        self.streaming_subscriptions = {}

        # Set Google A2A compatibility mode
        self._use_google_a2a = google_a2a_compatible

    def get_agent_card(self, agent_id: int) -> AgentCard:
        return run_coroutine_thread(self.xyz_get_agent_card(agent_id))

    @abc.abstractmethod
    async def xyz_get_agent_card(self, agent_id: int) -> AgentCard:
        pass

    @abc.abstractmethod
    async def xyz_stream_response(
        self, agent_id: int, message: Message
    ) -> AsyncGenerator[str, None]:
        pass

    @abc.abstractmethod
    def xyz_handle_message(self, agent_id: int, message: Message) -> Message:
        pass

    def handle_message(self, message):
        return run_coroutine_thread(
            self.xyz_handle_message(agent_id=g.agent_id, message=message)
        )

    def get_metadata(self, agent_id) -> dict[str, Any]:
        agent_card: AgentCard = self.get_agent_card(agent_id)

        return {
            "agent_type": "A2AServer",
            "capabilities": ["text"],
            "has_agent_card": True,
            "agent_name": agent_card.name,
            "agent_version": agent_card.version,
            "google_a2a_compatible": self._use_google_a2a,
        }

    def load_app(self):
        app = Flask(__name__)

        @app.after_request
        def add_cors_headers(response):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization"
            )
            return response

        @app.route("/", methods=["OPTIONS"])
        @app.route("/<path:path>", methods=["OPTIONS"])
        def options_handler(path=None):
            return "", 200

        # http://127.0.0.1:5000/.well-known/agent.json
        # http://127.0.0.1:5000/.well-known.json?agent_id=200
        @app.route("/<int:agent_id>/", methods=["GET"])
        def a2a_root_get(agent_id: int):
            """Root endpoint for A2A (GET), redirects to agent card"""
            agent_card: AgentCard = self.get_agent_card(agent_id)

            return jsonify(
                {
                    "name": agent_card.name,
                    "description": agent_card.description,
                    "agent_card_url": f"{self.url}/.well-known.json?agent_id={agent_id}",
                    "protocol": "a2a",
                    "capabilities": agent_card.capabilities,
                }
            )

        @app.route("/<int:agent_id>/agent.json", methods=["GET"])
        def agent_card(agent_id: int):
            agent_card: AgentCard = self.get_agent_card(agent_id)
            return jsonify(agent_card.to_dict())

        @app.route("/.well-known.json", methods=["GET"])
        def get_agent_card():
            agent_id = request.args.get("agent_id", type=int)
            return agent_card(agent_id)

        @app.route("/<int:agent_id>/a2a/tasks/send", methods=["POST"])
        def a2a_tasks_send(agent_id: int):
            g.agent_id = agent_id
            try:
                # Parse JSON data
                request_data = request.json

                # Handle as JSON-RPC if it follows that format
                if "jsonrpc" in request_data:
                    rpc_id = request_data.get("id", 1)
                    params = request_data.get("params", {})

                    # Detect format from params
                    is_google_format = False
                    if isinstance(params, dict) and "message" in params:
                        message_data = params.get("message", {})
                        if (
                            isinstance(message_data, dict)
                            and "parts" in message_data
                            and "role" in message_data
                        ):
                            is_google_format = True

                    # Process the task
                    result = self._handle_task_request(params, is_google_format)

                    # Get the data from the response
                    result_data = (
                        result.get_json()
                        if hasattr(result, "get_json")
                        else result.json
                    )

                    # Return JSON-RPC response
                    return jsonify(
                        {"jsonrpc": "2.0", "id": rpc_id, "result": result_data}
                    )
                else:
                    # Direct task submission - detect format
                    is_google_format = False
                    if "message" in request_data:
                        message_data = request_data.get("message", {})
                        if (
                            isinstance(message_data, dict)
                            and "parts" in message_data
                            and "role" in message_data
                        ):
                            is_google_format = True

                    # Handle the task request
                    return self._handle_task_request(request_data, is_google_format)

            except Exception as e:
                # Handle error based on request format
                if "jsonrpc" in request_data:
                    return jsonify(
                        {
                            "jsonrpc": "2.0",
                            "id": request_data.get("id", 1),
                            "error": {
                                "code": -32603,
                                "message": f"Internal error: {str(e)}",
                            },
                        }
                    ), 500
                else:
                    if self._use_google_a2a:
                        return jsonify(
                            {
                                "role": "agent",
                                "parts": [
                                    {
                                        "type": "data",
                                        "data": {"error": f"Error: {str(e)}"},
                                    }
                                ],
                            }
                        ), 500
                    else:
                        return jsonify(
                            {
                                "content": {
                                    "type": "error",
                                    "message": f"Error: {str(e)}",
                                },
                                "role": "system",
                            }
                        ), 500

        @app.route("/<int:agent_id>/tasks/send", methods=["POST"])
        def tasks_send(agent_id: int):
            """Forward to the A2A tasks/send endpoint"""
            return a2a_tasks_send(agent_id)

        @app.route("/<int:agent_id>/a2a/tasks/get", methods=["POST"])
        def a2a_tasks_get(agent_id):
            try:
                # Parse JSON data
                request_data = request.json

                # Handle as JSON-RPC if it follows that format
                if "jsonrpc" in request_data:
                    rpc_id = request_data.get("id", 1)
                    params = request_data.get("params", {})

                    # Extract task ID
                    task_id = params.get("id")
                    history_length = params.get("historyLength", 0)

                    # Get the task
                    task = self.tasks.get(task_id)
                    if not task:
                        return jsonify(
                            {
                                "jsonrpc": "2.0",
                                "id": rpc_id,
                                "error": {
                                    "code": -32000,
                                    "message": f"Task not found: {task_id}",
                                },
                            }
                        ), 404

                    # Convert task to dict in appropriate format
                    if self._use_google_a2a:
                        task_dict = task.to_google_a2a()
                    else:
                        task_dict = task.to_dict()

                    # Return the task
                    return jsonify(
                        {"jsonrpc": "2.0", "id": rpc_id, "result": task_dict}
                    )
                else:
                    # Handle as direct task request
                    task_id = request_data.get("id")

                    # Get the task
                    task = self.tasks.get(task_id)
                    if not task:
                        return jsonify({"error": f"Task not found: {task_id}"}), 404

                    # Convert task to dict in appropriate format
                    if self._use_google_a2a:
                        task_dict = task.to_google_a2a()
                    else:
                        task_dict = task.to_dict()

                    # Return the task
                    return jsonify(task_dict)

            except Exception as e:
                # Handle error
                return jsonify(
                    {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id", 1)
                        if "request_data" in locals()
                        else 1,
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}",
                        },
                    }
                ), 500

        # Also support the standard /tasks/get at the root
        @app.route("/<int:agent_id>/tasks/get", methods=["POST"])
        def tasks_get(agent_id: int):
            """Forward to the A2A tasks/get endpoint"""
            return a2a_tasks_get(agent_id)

        @app.route("/<int:agent_id>/a2a/tasks/cancel", methods=["POST"])
        def a2a_tasks_cancel(agent_id: int):
            """Handle POST request to cancel a task"""
            try:
                # Parse JSON data
                request_data = request.json

                # Handle as JSON-RPC if it follows that format
                if "jsonrpc" in request_data:
                    rpc_id = request_data.get("id", 1)
                    params = request_data.get("params", {})

                    # Extract task ID
                    task_id = params.get("id")

                    # Get the task
                    task = self.tasks.get(task_id)
                    if not task:
                        return jsonify(
                            {
                                "jsonrpc": "2.0",
                                "id": rpc_id,
                                "error": {
                                    "code": -32000,
                                    "message": f"Task not found: {task_id}",
                                },
                            }
                        ), 404

                    # Cancel the task
                    task.status = TaskStatus(state=TaskState.CANCELED)

                    # Convert task to dict in appropriate format
                    if self._use_google_a2a:
                        task_dict = task.to_google_a2a()
                    else:
                        task_dict = task.to_dict()

                    # Return the task
                    return jsonify(
                        {"jsonrpc": "2.0", "id": rpc_id, "result": task_dict}
                    )
                else:
                    # Handle as direct task request
                    task_id = request_data.get("id")

                    # Get the task
                    task = self.tasks.get(task_id)
                    if not task:
                        return jsonify({"error": f"Task not found: {task_id}"}), 404

                    # Cancel the task
                    task.status = TaskStatus(state=TaskState.CANCELED)

                    # Convert task to dict in appropriate format
                    if self._use_google_a2a:
                        task_dict = task.to_google_a2a()
                    else:
                        task_dict = task.to_dict()

                    # Return the task
                    return jsonify(task_dict)

            except Exception as e:
                # Handle error
                return jsonify(
                    {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id", 1)
                        if "request_data" in locals()
                        else 1,
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}",
                        },
                    }
                ), 500

        @app.route("/<int:agent_id>/tasks/cancel", methods=["POST"])
        def tasks_cancel(agent_id: int):
            return a2a_tasks_cancel(agent_id)

        @app.route("/<int:agent_id>/a2a/tasks/stream", methods=["POST"])
        def a2a_tasks_stream(agent_id: int):
            g.agent_id = agent_id

            try:
                # Parse JSON data
                request_data = request.json

                # Check if this is a JSON-RPC request
                if "jsonrpc" in request_data:
                    method = request_data.get("method", "")
                    params = request_data.get("params", {})
                    rpc_id = request_data.get("id", 1)

                    # Handle different streaming methods
                    if method == "tasks/sendSubscribe":
                        # Process tasks/sendSubscribe
                        return self._handle_tasks_send_subscribe(params, rpc_id)
                    elif method == "tasks/resubscribe":
                        # Process tasks/resubscribe
                        return self._handle_tasks_resubscribe(params, rpc_id)
                    else:
                        # Unknown method
                        return jsonify(
                            {
                                "jsonrpc": "2.0",
                                "id": rpc_id,
                                "error": {
                                    "code": -32601,
                                    "message": f"Method '{method}' not found",
                                },
                            }
                        ), 404
                else:
                    # Not a JSON-RPC request
                    return jsonify(
                        {"error": "Expected JSON-RPC format for streaming requests"}
                    ), 400

            except Exception as e:
                # Handle error
                return jsonify(
                    {
                        "jsonrpc": "2.0",
                        "id": request_data.get("id", 1)
                        if "request_data" in locals()
                        else 1,
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}",
                        },
                    }
                ), 500

        @app.route("/<int:agent_id>/tasks/stream", methods=["POST"])
        def tasks_stream(agent_id: int):
            return a2a_tasks_stream(agent_id)

        def get_agent_data(agent_id: int):
            agent_card: AgentCard = self.get_agent_card(agent_id)

            return {
                "name": agent_card.name,
                "description": agent_card.description,
                "version": agent_card.version,
                "skills": agent_card.skills,
            }

        @app.route("/<int:agent_id>/a2a/agent.json", methods=["GET"])
        def enhanced_a2a_agent_json(agent_id: int):
            agent_data = get_agent_data(agent_id)

            if hasattr(self, "_use_google_a2a"):
                if "capabilities" not in agent_data:
                    agent_data["capabilities"] = {}
                agent_data["capabilities"]["google_a2a_compatible"] = getattr(
                    self, "_use_google_a2a", False
                )
                agent_data["capabilities"]["parts_array_format"] = getattr(
                    self, "_use_google_a2a", False
                )

            user_agent = request.headers.get("User-Agent", "")
            accept_header = request.headers.get("Accept", "")
            format_param = request.args.get("format", "")

            if format_param == "json" or (
                "application/json" in accept_header
                and not any(
                    browser in user_agent.lower()
                    for browser in ["mozilla", "chrome", "safari", "edge"]
                )
            ):
                return jsonify(agent_data)

            # Otherwise serve HTML with pretty JSON visualization
            formatted_json = json.dumps(agent_data, indent=2)
            response = make_response(
                render_template_string(
                    JSON_HTML_TEMPLATE,
                    title=agent_data.get("name", "A2A Agent"),
                    description="Agent Card JSON Data",
                    json_data=formatted_json,
                )
            )
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response

        @app.route("/<int:agent_id>/agent.json", methods=["GET"])
        def enhanced_root_agent_json(agent_id: int):
            return enhanced_a2a_agent_json(agent_id)

        @app.route("/<int:agent_id>/stream", methods=["POST"])
        def handle_streaming_request(agent_id):
            g.agent_id = agent_id

            try:
                # CORS for streaming - important for browser compatibility
                if request.method == "OPTIONS":
                    response = Response()
                    response.headers["Access-Control-Allow-Origin"] = "*"
                    response.headers["Access-Control-Allow-Methods"] = "POST"
                    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
                    return response

                # Check accept header for streaming support

                # Extract the message from the request
                data = request.json

                # Check if this is a direct message or wrapped
                if "message" in data and isinstance(data["message"], dict):
                    message = Message.from_dict(data["message"])
                else:
                    # Try parsing the entire request as a message
                    message = Message.from_dict(data)

                # NOTE: 使用 xyz_stream_response 来进行处理, 由于这会开启一个新的线程
                # 故在这里无法直接使用 g 对象 ...
                if not hasattr(self, "xyz_stream_response"):
                    error_msg = "This agent does not support streaming"
                    return jsonify({"error": error_msg}), 405

                # Check if stream_response is implemented (not just inherited)
                if not self.xyz_stream_response:
                    error_msg = (
                        "This agent inherits but does not implement xyz_stream_response"
                    )
                    return jsonify({"error": error_msg}), 501

                # Set up SSE streaming response
                def generate():
                    """Generator for streaming server-sent events."""
                    # Create a thread and asyncio event loop for streaming
                    queue = Queue()
                    done_event = threading.Event()

                    def run_async_stream(agent_id):
                        """Run the async stream in a dedicated thread with its own event loop."""

                        async def process_stream():
                            """Process the streaming response."""
                            try:
                                # Get the stream generator from the agent
                                # Note: stream_response returns an async generator, not an awaitable
                                stream_gen = self.xyz_stream_response(agent_id, message)

                                # First heartbeat is sent from outside this function

                                # Process each chunk
                                index = 0
                                async for chunk in stream_gen:
                                    # Create chunk object with metadata
                                    chunk_data = {
                                        "content": chunk,
                                        "index": index,
                                        "append": True,
                                    }

                                    # Put in queue
                                    queue.put(chunk_data)
                                    index += 1

                                # Signal completion
                                queue.put(
                                    {
                                        "content": "",
                                        "index": index,
                                        "append": True,
                                        "lastChunk": True,
                                    }
                                )

                            except Exception as e:
                                # Put error in queue
                                queue.put({"error": str(e)})

                            finally:
                                # Signal we're done
                                done_event.set()

                        # Create a new event loop for this thread
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                        # Run the streaming process
                        try:
                            loop.run_until_complete(process_stream())
                        except Exception as exc:
                            logging.error(exc)
                        finally:
                            loop.close()

                    # Start the streaming thread
                    thread = threading.Thread(target=run_async_stream, args=(agent_id,))
                    thread.daemon = True
                    thread.start()

                    # Yield initial SSE comment to establish connection
                    yield ": SSE stream established\n\n"

                    # Process queue items until done
                    timeout = time.time() + 60  # 60-second timeout

                    total_chunks = 0

                    while not done_event.is_set() and time.time() < timeout:
                        try:
                            # Check if we have a chunk in the queue
                            if not queue.empty():
                                chunk = queue.get(block=False)
                                total_chunks += 1

                                # Check if it's an error
                                if "error" in chunk:
                                    error_event = (
                                        f"event: error\ndata: {json.dumps(chunk)}\n\n"
                                    )
                                    yield error_event
                                    break

                                # Format as SSE event with proper newlines
                                data_event = f"data: {json.dumps(chunk)}\n\n"
                                yield data_event

                                # Check if it's the last chunk
                                if chunk.get("lastChunk", False):
                                    break
                            else:
                                # No data yet, sleep briefly
                                time.sleep(0.01)
                        except Empty:
                            # Queue was empty
                            time.sleep(0.01)
                        except Exception as e:
                            # Other error
                            error_event = f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                            yield error_event
                            break

                    # If timed out, send timeout error
                    if time.time() >= timeout and not done_event.is_set():
                        error_event = f"event: error\ndata: {json.dumps({'error': 'Streaming timed out'})}\n\n"
                        yield error_event

                # Create the streaming response
                response = Response(generate(), mimetype="text/event-stream")
                response.headers["Cache-Control"] = "no-cache"
                response.headers["Connection"] = "keep-alive"
                response.headers["X-Accel-Buffering"] = "no"  # Important for Nginx
                return response

            except Exception as e:
                # Return error response for any other exception
                return jsonify({"error": str(e)}), 500

        @app.route("/<int:agent_id>/a2a", methods=["POST"])
        def handle_a2a_request(agent_id: int) -> Union[Response, tuple]:
            """Handle A2A protocol requests"""
            g.agent_id = agent_id

            try:
                data = request.json

                # Detect if this is Google A2A format
                is_google_format = False
                if "parts" in data and "role" in data and "content" not in data:
                    is_google_format = True
                elif (
                    "messages" in data
                    and data["messages"]
                    and "parts" in data["messages"][0]
                    and "role" in data["messages"][0]
                ):
                    is_google_format = True

                # Check if this is a single message or a conversation
                if "messages" in data:
                    # This is a conversation
                    if is_google_format:
                        conversation = Conversation.from_google_a2a(data)
                    else:
                        conversation = Conversation.from_dict(data)

                    response = self.handle_conversation(conversation)

                    # Format response based on request format or agent preference
                    use_google_format = is_google_format
                    if hasattr(self, "_use_google_a2a"):
                        use_google_format = use_google_format or self._use_google_a2a

                    if use_google_format:
                        return jsonify(response.to_google_a2a())
                    else:
                        return jsonify(response.to_dict())
                else:
                    # This is a single message
                    if is_google_format:
                        message = Message.from_google_a2a(data)
                    else:
                        message = Message.from_dict(data)

                    response = self.handle_message(message)

                    # Format response based on request format or agent preference
                    use_google_format = is_google_format
                    if hasattr(self, "_use_google_a2a"):
                        use_google_format = use_google_format or self._use_google_a2a

                    if use_google_format:
                        return jsonify(response.to_google_a2a())
                    else:
                        return jsonify(response.to_dict())

            except Exception as e:
                # Determine response format based on request
                is_google_format = False
                if "data" in locals():
                    if isinstance(data, dict):
                        if "parts" in data and "role" in data and "content" not in data:
                            is_google_format = True
                        elif (
                            "messages" in data
                            and data["messages"]
                            and "parts" in data["messages"][0]
                            and "role" in data["messages"][0]
                        ):
                            is_google_format = True

                # Also consider agent preference
                if hasattr(self, "_use_google_a2a"):
                    is_google_format = is_google_format or self._use_google_a2a

                # Return error in appropriate format
                error_msg = f"Error processing request: {str(e)}"
                if is_google_format:
                    # Google A2A format
                    return jsonify(
                        {
                            "role": "agent",
                            "parts": [{"type": "data", "data": {"error": error_msg}}],
                        }
                    ), 500
                else:
                    # python_a2a format
                    return jsonify(
                        {
                            "content": {"type": "error", "message": error_msg},
                            "role": "system",
                        }
                    ), 500

        @app.route("/<int:agent_id>/a2a/metadata", methods=["GET"])
        def get_agent_metadata(agent_id: int) -> Response:
            """Return metadata about the agent"""
            metadata = self.get_metadata(agent_id)

            # Add Google A2A compatibility flag if available
            if hasattr(self, "_use_google_a2a"):
                metadata["google_a2a_compatible"] = getattr(
                    self, "_use_google_a2a", False
                )
                metadata["parts_array_format"] = getattr(self, "_use_google_a2a", False)

            return jsonify(metadata)

        @app.route("/a2a/health", methods=["GET"])
        @app.route("/<int:agent_id>/a2a/health", methods=["GET"])
        def health_check() -> Response:
            """Health check endpoint"""
            return jsonify({"status": "ok"})

        return app

    def start(self):
        app = self.load_app()
        app.run(host=self.host, port=self.port, debug=self.debug)


