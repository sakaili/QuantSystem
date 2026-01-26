# Docker部署快速参考

## 🚀 一键部署（Windows）

```bash
# 1. 双击运行
deploy-windows.bat

# 2. SSH登录服务器后
cd QuantSystem
nano .env  # 填入API密钥
chmod +x deploy.sh && ./deploy.sh
```

## 📋 部署前检查

```bash
chmod +x check-deploy.sh
./check-deploy.sh
```

## 🔧 Docker命令速查表

### 基础操作

| 命令 | 说明 |
|------|------|
| `docker-compose up -d` | 启动容器（后台） |
| `docker-compose down` | 停止并删除容器 |
| `docker-compose restart` | 重启容器 |
| `docker-compose stop` | 停止容器 |
| `docker-compose start` | 启动已停止的容器 |

### 查看状态

| 命令 | 说明 |
|------|------|
| `docker-compose ps` | 查看容器状态 |
| `docker-compose logs -f` | 实时查看日志 |
| `docker-compose logs --tail=100` | 查看最近100行日志 |
| `docker stats quant-trading-bot` | 查看资源使用 |
| `docker inspect quant-trading-bot` | 查看详细信息 |

### 调试操作

| 命令 | 说明 |
|------|------|
| `docker exec -it quant-trading-bot bash` | 进入容器 |
| `docker-compose logs --since="1h"` | 查看最近1小时日志 |
| `docker-compose logs --until="2h"` | 查看2小时前的日志 |
| `docker-compose build --no-cache` | 重新构建镜像（无缓存） |

### 清理操作

| 命令 | 说明 |
|------|------|
| `docker-compose down -v` | 停止并删除卷 |
| `docker system prune -a` | 清理所有未使用资源 |
| `docker volume prune` | 清理未使用的卷 |
| `docker image prune -a` | 清理未使用的镜像 |

## 📊 监控命令

### 查看实时性能

```bash
# CPU、内存、网络IO
docker stats quant-trading-bot --no-stream

# 持续监控
watch -n 2 'docker stats quant-trading-bot --no-stream'
```

### 查看日志关键词

```bash
# 查看错误日志
docker-compose logs | grep ERROR

# 查看交易日志
docker-compose logs | grep "下单成功"

# 查看持仓信息
docker-compose logs | grep "持仓"
```

## 🔄 更新部署

### 方法1：原地更新（推荐）

```bash
# 1. 上传新代码
scp quant-system.tar.gz user@server:/root/QuantSystem/

# 2. SSH到服务器
cd /root/QuantSystem
tar -xzf quant-system.tar.gz

# 3. 重新构建和部署
docker-compose down
docker-compose build
docker-compose up -d
```

### 方法2：使用Git

```bash
cd /root/QuantSystem
git pull
docker-compose down
docker-compose build
docker-compose up -d
```

## 🚨 故障排查

### 容器无法启动

```bash
# 查看详细错误
docker-compose logs

# 查看容器退出原因
docker inspect quant-trading-bot --format='{{.State.Status}}: {{.State.Error}}'
```

### API连接失败

```bash
# 测试网络
docker exec quant-trading-bot ping -c 3 api.binance.com

# 检查时间同步
docker exec quant-trading-bot date
```

### 磁盘空间不足

```bash
# 查看磁盘使用
df -h
du -sh logs/ data/

# 清理旧日志
find logs/ -name "*.log.*" -mtime +7 -delete
docker system prune -a -f
```

### 内存不足

```bash
# 增加swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 永久生效
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 📁 文件结构

```
QuantSystem/
├── docker-compose.yml     # Docker编排
├── Dockerfile            # 镜像构建
├── .env                  # API密钥（需创建）
├── config/               # 配置文件（只读挂载）
├── logs/                 # 日志目录（持久化）
├── data/                 # 数据目录（持久化）
└── trading_bot.py        # 主程序
```

## 🔐 安全提示

1. **保护.env文件**
   ```bash
   chmod 600 .env
   ```

2. **查看敏感信息**
   ```bash
   # 不要在日志中显示
   docker-compose logs | grep -v "API_KEY"
   ```

3. **定期备份**
   ```bash
   tar -czf backup-$(date +%Y%m%d).tar.gz config/ data/
   ```

## 💡 Pro Tips

1. **创建别名** (添加到 `~/.bashrc`)
   ```bash
   alias qbot='cd /root/QuantSystem && docker-compose'
   alias qlog='docker-compose -f /root/QuantSystem/docker-compose.yml logs -f'
   alias qstat='docker stats quant-trading-bot --no-stream'
   ```

2. **定时重启**
   ```bash
   # 每天凌晨4点重启
   echo "0 4 * * * cd /root/QuantSystem && docker-compose restart" | crontab -
   ```

3. **监控脚本**
   ```bash
   # 每5分钟检查容器状态
   */5 * * * * docker ps | grep -q quant-trading-bot || (cd /root/QuantSystem && docker-compose up -d)
   ```

## 📞 获取帮助

- 查看完整文档: `cat DEPLOY.md`
- 检查部署环境: `./check-deploy.sh`
- 测试Docker构建: `./test-docker.sh`
