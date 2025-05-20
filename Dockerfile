# 使用Python 3.10作为基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# 安装系统依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        default-libmysqlclient-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/
RUN pip install -e /app/MXYZ-Agent-Core
RUN pip install -e /app/multi-agent-centre
RUN pip install -e /app/xyz-databases

# 安装Python依赖
RUN pip install --no-cache-dir -r /app/requirements.txt

# 暴露端口（根据config中的配置）
EXPOSE 10254

# 启动命令
CMD ["gunicorn -w1 -b 0.0.0.0:5000 main:app"]
