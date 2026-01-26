#!/bin/bash
# 一键部署脚本 - 用于云服务器部署

set -e  # 遇到错误立即退出

echo "======================================"
echo "  量化交易系统 - 云服务器部署脚本"
echo "======================================"
echo ""

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，正在安装..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "✅ Docker安装完成"
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，正在安装..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✅ Docker Compose安装完成"
fi

# 检查.env文件
if [ ! -f .env ]; then
    echo "⚠️  .env文件不存在，从.env.example复制..."
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ 已创建.env文件"
        echo ""
        echo "⚠️  请编辑.env文件，填入您的Binance API密钥："
        echo "   nano .env"
        echo ""
        read -p "填写完成后按Enter继续..."
    else
        echo "❌ .env.example文件不存在"
        exit 1
    fi
fi

# 验证.env文件中是否有API密钥
if grep -q "your_api_key_here" .env || grep -q "your_api_secret_here" .env; then
    echo "❌ 请先在.env文件中配置正确的API密钥"
    echo "   nano .env"
    exit 1
fi

echo "📦 构建Docker镜像..."
docker-compose build

echo "🚀 启动容器..."
docker-compose up -d

echo ""
echo "✅ 部署完成！"
echo ""
echo "📊 查看运行状态："
echo "   docker-compose ps"
echo ""
echo "📝 查看实时日志："
echo "   docker-compose logs -f"
echo ""
echo "🔄 重启服务："
echo "   docker-compose restart"
echo ""
echo "🛑 停止服务："
echo "   docker-compose stop"
echo ""
echo "🗑️  删除容器："
echo "   docker-compose down"
echo ""
