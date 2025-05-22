#!/bin/bash
# 使用 set -e，如果任何命令失败，脚本会立即退出
set -e

echo "Starting Gunicorn server..."
# 在后台运行 Gunicorn
gunicorn -w 1 --threads 8 -b 0.0.0.0:5000 main:app &
gunicorn_pid=$!
echo "Gunicorn started with PID $gunicorn_pid"

echo "Starting fastmcp server..."
# 在后台运行 fastmcp
fastmcp run mcp_server.py --transport sse --port 10254 &
fastmcp_pid=$!
echo "fastmcp server started with PID $fastmcp_pid"

cleanup() {
	echo "Caught SIGTERM/SIGINT signal! Shutting down..."
	# 检查 PID 是否存在且非空
	if [ -n "$gunicorn_pid" ] && kill -0 "$gunicorn_pid" 2>/dev/null; then
		echo "Sending SIGTERM to Gunicorn (PID $gunicorn_pid)..."
		kill -TERM "$gunicorn_pid"
	fi
	if [ -n "$fastmcp_pid" ] && kill -0 "$fastmcp_pid" 2>/dev/null; then
		echo "Sending SIGTERM to fastmcp (PID $fastmcp_pid)..."
		kill -TERM "$fastmcp_pid"
	fi

	echo "Waiting for Gunicorn to shut down..."
	wait "$gunicorn_pid" 2>/dev/null || true # 忽略 wait 可能因进程已死引发的错误
	echo "Gunicorn shut down."

	echo "Waiting for fastmcp to shut down..."
	wait "$fastmcp_pid" 2>/dev/null || true # 忽略 wait 可能因进程已死引发的错误
	echo "fastmcp shut down."

	echo "All processes terminated. Exiting."
}
trap cleanup TERM INT

echo "Waiting for any process to exit (using wait -n)..."
# 这一行现在应该可以正常工作了，因为使用的是 bash
wait -n
EXIT_CODE=$?

echo "A process exited with code $EXIT_CODE. Initiating shutdown of other processes..."
# trap 会处理优雅关闭，但为了确保，如果脚本不是因为信号而是因为子进程退出到这里，
# 我们也可以尝试再次触发清理，或者让 trap 之后的脚本自然结束。
# 实际上，当 wait -n 返回后，脚本会继续执行并退出，
# 如果其他进程仍在运行，它们将成为孤儿进程并可能被 init (PID 1) 接管或终止。
# trap中的kill仍然是主要的优雅关闭机制。
# 为了确保如果一个进程死了，另一个也被通知，可以再次发送信号：
if [ -n "$gunicorn_pid" ] && kill -0 "$gunicorn_pid" 2>/dev/null; then kill -TERM "$gunicorn_pid" 2>/dev/null || true; fi
if [ -n "$fastmcp_pid" ] && kill -0 "$fastmcp_pid" 2>/dev/null; then kill -TERM "$fastmcp_pid" 2>/dev/null || true; fi

exit $EXIT_CODE
