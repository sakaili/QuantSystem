#!/bin/bash
# 部署前检查脚本

echo "=========================================="
echo "  部署前环境检查"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0

# 检查Python文件语法
echo "📝 检查Python文件语法..."
python_files=$(find . -name "*.py" -not -path "./venv/*" -not -path "./__pycache__/*")
for file in $python_files; do
    if ! python -m py_compile "$file" 2>/dev/null; then
        echo -e "${RED}✗${NC} $file 语法错误"
        ERRORS=$((ERRORS+1))
    fi
done
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Python语法检查通过"
fi
echo ""

# 检查必要文件
echo "📁 检查必要文件..."
required_files=(
    "trading_bot.py"
    "requirements.txt"
    "Dockerfile"
    "docker-compose.yml"
    ".dockerignore"
    "config/strategy_config.yaml"
    "config/risk_config.yaml"
    "config/api_config.yaml"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file 缺失"
        ERRORS=$((ERRORS+1))
    fi
done
echo ""

# 检查目录结构
echo "📂 检查目录结构..."
required_dirs=(
    "core"
    "utils"
    "config"
)

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✓${NC} $dir/"
    else
        echo -e "${RED}✗${NC} $dir/ 缺失"
        ERRORS=$((ERRORS+1))
    fi
done
echo ""

# 检查.env文件
echo "🔑 检查API配置..."
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠${NC}  .env文件不存在"
    echo "   创建.env文件并填入API密钥："
    echo "   BINANCE_API_KEY=your_key"
    echo "   BINANCE_API_SECRET=your_secret"
    ERRORS=$((ERRORS+1))
elif grep -q "your_api_key_here" .env 2>/dev/null || grep -q "test_key" .env 2>/dev/null; then
    echo -e "${YELLOW}⚠${NC}  .env文件包含测试密钥"
    echo "   请填入真实的Binance API密钥"
    ERRORS=$((ERRORS+1))
else
    echo -e "${GREEN}✓${NC} .env文件已配置"
fi
echo ""

# 检查requirements.txt
echo "📦 检查Python依赖..."
if [ -f requirements.txt ]; then
    echo -e "${GREEN}✓${NC} requirements.txt存在"
    echo "   依赖包数量: $(wc -l < requirements.txt)"
else
    echo -e "${RED}✗${NC} requirements.txt缺失"
    ERRORS=$((ERRORS+1))
fi
echo ""

# 检查Docker环境
echo "🐳 检查Docker环境..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker已安装: $(docker --version)"
else
    echo -e "${YELLOW}⚠${NC}  Docker未安装"
    echo "   将在服务器上自动安装"
fi

if command -v docker-compose &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker Compose已安装: $(docker-compose --version)"
else
    echo -e "${YELLOW}⚠${NC}  Docker Compose未安装"
    echo "   将在服务器上自动安装"
fi
echo ""

# 总结
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ 所有检查通过，可以部署！${NC}"
    echo "=========================================="
    echo ""
    echo "下一步："
    echo "  1. 运行: ./deploy-windows.bat (Windows)"
    echo "  或  scp quant-system.tar.gz user@server:/root/"
    echo "  2. SSH到服务器并运行 ./deploy.sh"
    exit 0
else
    echo -e "${RED}✗ 发现 $ERRORS 个问题，请先修复${NC}"
    echo "=========================================="
    exit 1
fi
