# Docker快速部署

## 一键部署（3步）

### 方法A：Windows用户（推荐）

1. **双击运行**：`deploy-windows.bat`
2. **按提示操作**：输入服务器IP和用户名
3. **SSH到服务器**：运行部署脚本

```bash
cd /root/QuantSystem
nano .env  # 配置API密钥
chmod +x deploy.sh && ./deploy.sh
```

### 方法B：Linux/Mac用户

```bash
# 1. 打包代码
tar -czf quant-system.tar.gz --exclude=logs --exclude=data --exclude=__pycache__ .

# 2. 上传到服务器
scp quant-system.tar.gz user@your-server:/root/

# 3. SSH登录并部署
ssh user@your-server
cd /root && mkdir QuantSystem && tar -xzf quant-system.tar.gz -C QuantSystem
cd QuantSystem
nano .env  # 配置API密钥
chmod +x deploy.sh && ./deploy.sh
```

## 常用命令速查

| 操作 | 命令 |
|------|------|
| 查看状态 | `docker-compose ps` |
| 查看日志 | `docker-compose logs -f` |
| 重启 | `docker-compose restart` |
| 停止 | `docker-compose stop` |
| 启动 | `docker-compose up -d` |
| 完全重置 | `docker-compose down && docker-compose up -d` |
| 查看资源 | `docker stats quant-trading-bot` |

## 目录结构

```
QuantSystem/
├── docker-compose.yml    # Docker编排配置
├── Dockerfile           # 镜像构建配置
├── deploy.sh           # Linux部署脚本
├── deploy-windows.bat  # Windows部署脚本
├── .env               # API密钥（需手动创建）
├── config/            # 策略配置文件
├── logs/              # 日志（自动创建）
└── data/              # 数据库（自动创建）
```

## 配置API密钥

创建`.env`文件：

```bash
nano .env
```

内容：

```env
BINANCE_API_KEY=你的API密钥
BINANCE_API_SECRET=你的API密钥密文
```

## 故障排查

### 容器启动失败

```bash
docker-compose logs
```

### 查看详细状态

```bash
docker inspect quant-trading-bot
```

### 进入容器调试

```bash
docker exec -it quant-trading-bot bash
```

### 清理并重建

```bash
docker-compose down
docker system prune -a
docker-compose build --no-cache
docker-compose up -d
```

## 更多信息

详细部署指南请查看：[DEPLOY.md](DEPLOY.md)
