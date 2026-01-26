# 云服务器部署指南

本指南将帮助您将量化交易系统一键部署到云服务器。

## 前置要求

- Linux服务器（推荐Ubuntu 20.04+）
- 至少1GB RAM，1核CPU
- 稳定的网络连接
- Binance API密钥

## 快速部署（3步完成）

### 1. 上传代码到服务器

```bash
# 在本地打包代码
cd C:\Users\songz\Documents\Obsidian Vault\QuantSystem
tar -czf quant-system.tar.gz --exclude=logs --exclude=data --exclude=__pycache__ .

# 上传到服务器（替换为您的服务器IP）
scp quant-system.tar.gz user@your-server-ip:/home/user/
```

或者使用Git：

```bash
# 在服务器上
git clone <your-repo-url>
cd QuantSystem
```

### 2. 配置API密钥

```bash
# SSH登录服务器
ssh user@your-server-ip

# 进入项目目录
cd QuantSystem

# 创建.env文件
nano .env
```

填入以下内容（替换为您的真实密钥）：

```env
BINANCE_API_KEY=your_real_api_key_here
BINANCE_API_SECRET=your_real_api_secret_here
```

保存并退出（Ctrl+X，Y，Enter）

### 3. 一键部署

```bash
# 赋予执行权限
chmod +x deploy.sh

# 运行部署脚本
./deploy.sh
```

部署脚本会自动：
- ✅ 检查并安装Docker
- ✅ 检查并安装Docker Compose
- ✅ 验证配置文件
- ✅ 构建Docker镜像
- ✅ 启动交易系统

## 常用命令

### 查看运行状态

```bash
docker-compose ps
```

### 查看实时日志

```bash
# 查看所有日志
docker-compose logs -f

# 查看最近100行
docker-compose logs --tail=100

# 查看具体时间段
docker-compose logs --since="2026-01-25T00:00:00"
```

### 重启系统

```bash
# 重启容器
docker-compose restart

# 完全重新部署（重新构建镜像）
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### 停止系统

```bash
# 停止但保留容器
docker-compose stop

# 停止并删除容器（数据保留）
docker-compose down
```

### 更新代码

```bash
# 拉取最新代码
git pull

# 或上传新的tar.gz并解压
tar -xzf quant-system.tar.gz

# 重新构建并启动
docker-compose down
docker-compose build
docker-compose up -d
```

### 查看资源使用

```bash
# 查看容器资源占用
docker stats quant-trading-bot

# 查看磁盘使用
du -sh logs/ data/
```

## 数据持久化

以下目录会持久化到宿主机：

- `./logs/` - 日志文件
- `./data/` - 数据库和历史数据
- `./config/` - 配置文件（只读）

即使删除容器，这些数据也会保留。

## 监控与维护

### 设置定时重启（可选）

```bash
# 编辑crontab
crontab -e

# 添加每天凌晨4点重启
0 4 * * * cd /path/to/QuantSystem && docker-compose restart
```

### 日志轮转

系统已配置日志自动轮转：
- 单个文件最大10MB
- 保留最近3个文件

### 监控脚本

创建`monitor.sh`：

```bash
#!/bin/bash
# 检查容器是否运行
if ! docker ps | grep -q quant-trading-bot; then
    echo "容器已停止，正在重启..."
    cd /path/to/QuantSystem
    docker-compose up -d
    echo "$(date): 容器重启" >> monitor.log
fi
```

设置定时检查：

```bash
crontab -e
# 每5分钟检查一次
*/5 * * * * /path/to/monitor.sh
```

## 安全建议

### 1. 防火墙配置

```bash
# 只允许SSH访问
sudo ufw allow 22/tcp
sudo ufw enable
```

### 2. API密钥权限

确保Binance API密钥配置：
- ✅ 启用：现货和杠杆交易
- ✅ 启用：读取
- ❌ 禁用：提现
- ❌ 禁用：内部转账

### 3. IP白名单

在Binance设置中绑定服务器IP，增强安全性。

### 4. 定期备份

```bash
# 备份配置和数据
tar -czf backup-$(date +%Y%m%d).tar.gz config/ data/

# 上传到其他服务器或云存储
```

## 故障排查

### 容器无法启动

```bash
# 查看详细错误
docker-compose logs

# 检查配置文件
cat .env
ls -la config/
```

### API连接失败

```bash
# 测试网络连接
ping api.binance.com

# 检查时间同步
date
# 如果时间不准确：
sudo apt-get install ntpdate
sudo ntpdate ntp.ubuntu.com
```

### 内存不足

```bash
# 增加swap空间
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 磁盘空间不足

```bash
# 清理Docker缓存
docker system prune -a

# 清理旧日志
find logs/ -name "*.log.*" -mtime +7 -delete
```

## 性能优化

### 1. 使用国内镜像源（国内服务器）

编辑`Dockerfile`，在`pip install`前添加：

```dockerfile
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 减少日志输出

编辑`config/strategy_config.yaml`：

```yaml
logging:
  level: INFO  # 改为WARNING减少日志
```

### 3. 优化扫描频率

编辑`config/strategy_config.yaml`：

```yaml
schedule:
  monitor_interval: 10  # 根据需要调整（秒）
```

## 联系支持

如遇问题，请查看日志文件或联系技术支持。
