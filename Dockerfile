FROM python:3.9-slim

# 设置时区为UTC（重要：与Binance API时间同步）
ENV TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 创建必要的目录
RUN mkdir -p logs data/daily_scans config static

# 暴露 Web 仪表板端口
EXPOSE 5000

# 验证关键文件存在
RUN test -f trading_bot.py && \
    test -d core && \
    test -d config && \
    echo "✅ 核心文件验证通过"

# 健康检查
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD test -f logs/bot.log || exit 1

# 运行 - 使用绝对路径和完整命令
CMD ["python", "-u", "trading_bot.py", "--config", "config/"]


