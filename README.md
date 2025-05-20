# XYZ-A2A

## 介绍

适用于 `XYZ` 平台的 `Agent To Agent Protocol` 协议封装.

它使用了非标的 `A2A` 协议实现, 来实现多个 `Agent` 仅仅通过 1 个 `Server` 就具有支持 `A2A` 协议的能力.

## 安装

使用 `uv`:

```bash
cd deps
git clone git@github.com:protagolabs/MXYZ-Agent-Core.git
git clone git@github.com:protagolabs/xyz-databases.git
git clone git@github.com:protagolabs/multi-agent-centre.git

cd ..
uv pip install -e ./deps/MXYZ-Agent-Core
uv pip install -e ./deps/xyz-databases
uv pip install -e ./deps/multi-agent-centre
```

## 部署

使用 `wsgi` 运行服务:

```bash
gunicorn -w1 -b 0.0.0.0:5000 main:app
```

`mcp_server` 也需要启动.
