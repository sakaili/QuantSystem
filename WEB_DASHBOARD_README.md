# Web 仪表板部署说明

## 概述

已为您的网格交易系统添加了 Web 仪表板功能,可以通过浏览器实时监控持仓、网格状态、盈亏情况和策略表现。

## 新增文件

### 后端文件
- `web_api.py` - Flask API 服务器,提供 RESTful API 接口
- 修改了 `trading_bot.py` - 集成 Web API 服务器

### 前端文件 (static/ 目录)
- `static/index.html` - 主页面
- `static/app.js` - 前端逻辑
- `static/charts.js` - 图表可视化
- `static/style.css` - 自定义样式

### 配置文件
- 修改了 `requirements.txt` - 添加 Flask 和 flask-cors 依赖
- 修改了 `Dockerfile` - 暴露 5000 端口
- 修改了 `docker-compose.yml` - 添加端口映射
- 修改了 `config/strategy_config.yaml` - 添加 web_dashboard 配置

## 功能特性

### 1. 账户总览
- 总余额
- 可用保证金
- 未实现盈亏
- 保证金使用率

### 2. 持仓列表
- 所有活跃持仓的详细信息
- 实时价格更新
- 盈亏统计
- 点击查看网格详情

### 3. 网格状态
- 入场价和当前价
- 上下网格活跃数量
- 网格成功率
- 止损价格

### 4. 盈利监控
- 各交易对盈利排行
- 平均盈利率
- 达标币种数

### 5. 性能指标
- 日/周/月盈亏图表
- 胜率统计

### 6. 风险告警
- 实时告警列表
- 按级别分类显示

## 部署步骤

### 本地测试

1. 安装依赖:
```bash
pip install -r requirements.txt
```

2. 启动交易机器人:
```bash
python trading_bot.py --config config/
```

3. 访问仪表板:
打开浏览器访问 `http://localhost:5000`

### Docker 部署 (推荐)

#### 首次部署

1. 构建并启动容器:
```bash
docker-compose up -d --build
```

2. 查看日志:
```bash
docker-compose logs -f
```

3. 访问仪表板:
打开浏览器访问 `http://localhost:5000`

#### 远程 ECS 部署

1. 上传代码到 ECS 服务器:
```bash
git clone <your-repo> /opt/quantsystem
cd /opt/quantsystem
```

2. 配置环境变量:
```bash
cp .env.example .env
vim .env  # 填入 Binance API 密钥
```

3. 一键部署:
```bash
docker-compose up -d --build
```

4. 配置 ECS 安全组:
- 在阿里云控制台添加入站规则
- 允许您的 IP 访问 5000 端口
- 或配置 VPN/堡垒机访问

5. 访问仪表板:
打开浏览器访问 `http://<ECS公网IP>:5000`

### 更新部署

```bash
cd /opt/quantsystem
git pull
docker-compose up -d --build
```

### 停止服务

```bash
docker-compose down
```

## 配置说明

在 `config/strategy_config.yaml` 中可以配置 Web 仪表板:

```yaml
web_dashboard:
  enabled: true          # 启用/禁用仪表板
  host: "0.0.0.0"       # 监听地址
  port: 5000            # 监听端口
  debug: false          # 调试模式
```

## API 端点

仪表板通过以下 API 端点获取数据:

- `GET /api/account` - 账户总览
- `GET /api/positions` - 持仓列表
- `GET /api/positions/<symbol>/grid` - 网格状态
- `GET /api/profit` - 盈利监控
- `GET /api/orders` - 订单列表
- `GET /api/trades` - 交易历史
- `GET /api/alerts` - 风险告警
- `GET /api/system` - 系统状态
- `GET /api/performance` - 性能指标

## 安全建议

### 生产环境

1. **配置 ECS 安全组**:
   - 仅允许特定 IP 访问 5000 端口
   - 或使用 VPN/堡垒机访问

2. **使用 Nginx 反向代理** (可选):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. **配置 SSL 证书**:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## 自动刷新

仪表板每 5 秒自动刷新一次数据,无需手动刷新页面。

## 浏览器兼容性

支持所有现代浏览器:
- Chrome/Edge (推荐)
- Firefox
- Safari

## 故障排查

### 无法访问仪表板

1. 检查容器是否运行:
```bash
docker ps
```

2. 检查日志:
```bash
docker-compose logs web-api
```

3. 检查端口是否开放:
```bash
netstat -tulpn | grep 5000
```

### 数据不更新

1. 检查 API 是否正常:
```bash
curl http://localhost:5000/api/account
```

2. 查看浏览器控制台错误信息 (F12)

### 性能问题

如果仪表板响应慢:
1. 增加 Docker 容器资源限制
2. 调整自动刷新间隔 (修改 app.js 中的 5000 毫秒)

## 后续扩展

可以考虑添加以下功能:
1. WebSocket 实时推送 (替代轮询)
2. 控制功能 (启动/停止/紧急关闭)
3. 历史数据回放
4. 移动端适配
5. 用户认证系统
6. 数据导出功能

## 技术栈

- **后端**: Flask + Python
- **前端**: HTML + JavaScript (原生)
- **UI 框架**: Bootstrap 5
- **图表库**: Chart.js
- **部署**: Docker + docker-compose

## 支持

如有问题,请查看:
1. 日志文件: `logs/trading_bot_*.log`
2. Docker 日志: `docker-compose logs -f`
3. 浏览器控制台 (F12)
