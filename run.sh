#!/bin/sh
# 使用 set -e，如果任何命令失败，脚本会立即退出
set -e

echo "Starting Gunicorn server..."
# 在后台运行 Gunicorn
gunicorn -w 4 --threads 8 -b 0.0.0.0:5000 main:app &
# 获取 Gunicorn 的进程 ID (PID)
gunicorn_pid=$!
echo "Gunicorn started with PID $gunicorn_pid"

echo "Starting fastmcp server..."
# 在后台运行 fastmcp
fastmcp run mcp_server.py --transport sse --port 10254 &
# 获取 fastmcp 的 PID
fastmcp_pid=$!
echo "fastmcp server started with PID $fastmcp_pid"

# 定义一个清理函数，用于在接收到信号时关闭子进程
# Docker stop 会发送 SIGTERM
cleanup() {
	echo "Caught SIGTERM/SIGINT signal! Shutting down..."
	# 给子进程发送 SIGTERM 信号，让它们有机会优雅关闭
	kill -TERM "$gunicorn_pid" "$fastmcp_pid"
	# 等待子进程实际退出
	echo "Waiting for Gunicorn (PID $gunicorn_pid) to shut down..."
	wait "$gunicorn_pid"
	echo "Gunicorn shut down."
	echo "Waiting for fastmcp (PID $fastmcp_pid) to shut down..."
	wait "$fastmcp_pid"
	echo "fastmcp shut down."
	echo "All processes terminated. Exiting."
}

# 捕获 SIGTERM 和 SIGINT 信号，并调用 cleanup 函数
trap cleanup TERM INT

# 等待任一后台进程退出。
# 如果 Gunicorn 或 fastmcp 中的任何一个意外崩溃，
# `wait -n` 会检测到，然后脚本会退出，导致容器停止。
# 这是期望的行为，因为如果关键服务失败，容器也应该停止。
echo "Waiting for processes to exit..."
wait -n # 等待任何一个后台任务结束
EXIT_CODE=$?

echo "A process exited with code $EXIT_CODE. Initiating shutdown of other processes if any..."
# 即使一个进程退出了，也尝试优雅地关闭另一个（如果还在运行）
# 再次调用cleanup，或者直接kill。为简单起见，这里依赖上面的trap或者让容器自然退出。
# 如果上面的wait -n是因为收到了外部信号而被中断，trap中的cleanup会执行。
# 如果是某个进程自己挂了，wait -n会退出，脚本会继续往下执行并退出。
# 为了确保另一个进程也被通知，可以再次尝试kill (可能它已经被trap处理了)
kill -TERM "$gunicorn_pid" 2>/dev/null || true # 忽略错误，如果进程已不在
kill -TERM "$fastmcp_pid" 2>/dev/null || true  # 忽略错误，如果进程已不在

exit $EXIT_CODE
