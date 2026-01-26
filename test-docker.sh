#!/bin/bash
# 本地Docker测试脚本

set -e

echo "=========================================="
echo "  Docker 本地测试脚本"
echo "=========================================="
echo ""

# 检查是否有.env文件
if [ ! -f .env ]; then
    echo "⚠️  .env文件不存在，创建测试用配置..."
    cat > .env << EOF
BINANCE_API_KEY=test_key_for_build
BINANCE_API_SECRET=test_secret_for_build
EOF
    echo "✅ 已创建测试.env文件"
fi

echo "🔨 开始构建Docker镜像..."
docker build -t quant-trading-bot:test .

if [ $? -eq 0 ]; then
    echo "✅ Docker镜像构建成功！"
    echo ""
    echo "📦 镜像信息:"
    docker images quant-trading-bot:test
    echo ""
    echo "🔍 镜像层信息:"
    docker history quant-trading-bot:test --no-trunc
    echo ""
    echo "=========================================="
    echo "  测试完成"
    echo "=========================================="
    echo ""
    echo "💡 下一步："
    echo "   1. 填写真实的API密钥到.env文件"
    echo "   2. 运行: docker-compose up -d"
    echo ""
else
    echo "❌ Docker镜像构建失败"
    exit 1
fi
