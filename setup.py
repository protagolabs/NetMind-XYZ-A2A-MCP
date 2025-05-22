import threading
from main import app
from mcp_server import mcp

if __name__ == "__main__":
    worker = threading.Thread(target=mcp.run, args=("sse",))
    worker.start()
    print("---")
    app.run()
