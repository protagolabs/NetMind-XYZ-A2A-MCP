import os
import dotenv

# 环境
env = os.getenv("ENV", "local")

dirname = os.path.dirname(os.path.dirname(__file__))
dotenv.load_dotenv(os.path.join(dirname, ".env", env + ".env"))


class EnvHelper:
    @classmethod
    def get_env_mark(cls) -> str:
        return env

    @classmethod
    def is_prod(cls) -> bool:
        # 是否是线上
        return env == "prod"

    @classmethod
    def is_test(cls) -> bool:
        # 是否是 test
        return env == "test"

    @classmethod
    def is_local(cls) -> bool:
        # 是否是 local
        return env == "local"

    @classmethod
    def get_multi_agent_host(cls) -> str:
        host = os.getenv("MULTI_AGENT_HOST", "localhost")
        return host

    @classmethod
    def get_multi_agent_port(cls) -> str:
        port = os.getenv("MULTI_AGENT_POST", 50051)

        if isinstance(port, str):
            port = int(port)

        return port

    @classmethod
    def get_xyz_platform_url(cls) -> str:
        url = os.getenv("XYZ_PLATFORM_URL", "http://127.0.0.1:8100")
        return url

    @classmethod
    def get_mcp_server_port(cls) -> str:
        port = os.getenv("MCP_SERVER_PORT", 10254)

        if isinstance(port, str):
            port = int(port)

        return port

    @classmethod
    def get_http_timeout(cls) -> int:
        timeout = os.getenv("HTTP_TIMEOUT", 30)

        if isinstance(timeout, str):
            timeout = int(timeout)

        return timeout
