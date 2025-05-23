# 使用Python 3.12 作为基础镜像
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
        bash \
        gcc \
        python3-dev \
        default-libmysqlclient-dev \
        pkg-config \
        ncat \
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
EXPOSE 5000

# 添加执行权限
RUN chmod +x /app/run.sh

ENTRYPOINT ["./run.sh"]
