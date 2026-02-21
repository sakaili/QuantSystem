# 量化交易系统 (QuantSystem)

基于Python的全自动空头偏向网格交易系统,连接Binance交易所,7x24小时运行。

## 特性

- ✅ 全自动交易(筛选、开仓、网格管理、止损)
- ✅ 空头网格策略(20格,2%间距)
- ✅ 多重风险控制(止损、资金费率监控、仓位限制)
- ✅ 参数可配置(YAML配置文件)
- ✅ 完整日志记录和数据持久化
- ✅ Docker部署支持

## 系统架构

```
trading_bot.py (主控程序)
├── config_manager.py (配置管理)
├── exchange_connector.py (交易执行引擎)
├── position_manager.py (仓位管理器)
├── grid_strategy.py (网格策略执行器)
├── risk_manager.py (风险控制模块)
└── database.py (数据持久化)
```

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库(如果使用git)
cd QuantSystem

# 安装Python 3.9+
python --version

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件,填入Binance API密钥
# BINANCE_API_KEY=your_key
# BINANCE_API_SECRET=your_secret
```

编辑配置文件 [config/strategy_config.yaml](config/strategy_config.yaml):

```yaml
account:
  size: 50.0              # 测试阶段用小额资金
  leverage: 2

position:
  max_symbols: 1          # 测试阶段只持1个币种

screening:
  min_listing_days: 365   # 只交易上市>1年的币种
```

### 3. 运行

```bash
# 直接运行
python trading_bot.py --config config/

# Docker运行(推荐)
docker build -t quantbot .

docker run -d \
  --name quantbot \
  --restart unless-stopped \
  -e BINANCE_API_KEY="your_key" \
  -e BINANCE_API_SECRET="your_secret" \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  quantbot
```

### 4. 监控

```bash
# 查看日志
tail -f logs/trading_bot_*.log

# Docker日志
docker logs -f quantbot

# 查看持仓状态
python -c "from core.database import Database; db = Database('data/database.db'); print('监控中...')"
```

## 策略说明

### 币种筛选标准

1. 技术形态: EMA5 < EMA10 < EMA20 < EMA30 (空头排列)
2. 流动性: 24h成交额排名倒数50名
3. 上市时间: 必须>365天
4. 资金费率: > 0 (多头付费给空头,空头收益)
5. 波动性: ATR未暴涨
6. (可选) 4H Squeeze Momentum < 0 (宏观跟随开仓)

### 排序与得分

候选币种默认按 `weighted_score` 排序，由高到低开仓补满（手动币种优先）。
`weighted_score` 使用各指标的百分位排名（rank pct）归一化后加权求和：

- squeeze_momentum（越负越好）: 0.35
- squeeze_momentum_delta（越负越好）: 0.20
- funding_rate_sum（越高越好）: 0.25  
  - 若未提前计算，扫描阶段会自动回补 `funding_rate_sum`
- EMA30 偏离（越大越好）: 0.20

说明：若启用 `--funding-rate-sort`，则会按历史资金费率累加排序而不是 `weighted_score`。

### 筛选排序的CLI选项

```bash
# 4H squeeze 过滤（默认关闭）
python daily_candidate_scan.py --use-squeeze-filter --squeeze-timeframe 4h

# 打印待开仓 TopN
python daily_candidate_scan.py --select-top 3

# 生成可视化（近一年价格 + squeeze动量）
python daily_candidate_scan.py --plot-squeeze --plot-top 10
```

> 可视化需要 `matplotlib`，已加入 `requirements.txt`。

### 网格配置

- 网格数量: 20个(上10下10)
- 网格间距: 1%
- 覆盖范围: P0 × 1.105 → P0 × 0.904 (±10.5% ~ ±9.6%)
- 订单类型: Post-Only Maker (手续费0.02%)

### 仓位管理

- 基础仓位: 15 USDT保证金
- 单网格: 5.25 USDT保证金
- 单币种最大: 67.5 USDT
- 杠杆: 2倍

### 风险控制

- 止损线: 1.15 × P0 (价格上涨15%)
- 资金费率止损: 负值持续3天
- 总保证金上限: 45 USDT (测试阶段)
- 最大持仓: 1个币种 (测试阶段)

## 目录结构

```
QuantSystem/
├── config/                      # 配置文件
│   ├── strategy_config.yaml     # 策略参数
│   ├── risk_config.yaml         # 风险控制
│   └── api_config.yaml          # API配置
├── core/                        # 核心模块
│   ├── config_manager.py
│   ├── exchange_connector.py
│   ├── position_manager.py
│   ├── grid_strategy.py
│   ├── risk_manager.py
│   └── database.py
├── utils/                       # 工具模块
│   ├── logger.py
│   └── exceptions.py
├── trading_bot.py               # 主程序
├── data_fetcher.py              # 数据获取
├── daily_candidate_scan.py      # 币种筛选
├── logs/                        # 日志目录
├── data/                        # 数据目录
│   ├── database.db              # SQLite数据库
│   └── daily_scans/             # 筛选结果
└── tests/                       # 测试

```

## 交易流程

### 1. 每日筛选 (UTC 00:00)

- 扫描Binance USDT永续合约
- 应用筛选标准
- 过滤上市<365天的币种
- 输出候选列表

### 2. 评估入场

- 检查持仓数量限制
- 检查可用保证金
- 获取当前价格作为入场价P0

### 3. 初始化网格

- 开基础仓位(15 USDT)
- 挂上方10个开空卖单(各5.25 USDT)
- 开始监控

### 4. 网格运行

- 上方网格成交 → 挂对应下方平空单
- 下方网格成交 → 重新挂上方开空单
- 循环往复捕获波动

### 5. 风险监控

- 每10秒检查价格
- Level 1预警: 1.10×P0
- Level 2预警: 1.13×P0
- Level 3止损: 1.15×P0 → 全部平仓

## 安全提示

⚠️ **重要风险提示:**

1. **市场风险**: 单边上涨行情会持续亏损
2. **技术风险**: 网络中断可能无法及时止损
3. **流动性风险**: 垃圾币可能出现极端滑点
4. **资金费率风险**: 持续负费率侵蚀利润

**防范措施:**

- ✅ 严格止损(1.15×P0)
- ✅ 小额测试(50 USDT起步)
- ✅ 7x24监控,异常告警
- ✅ 只用可承受损失的资金
- ✅ 先在测试网验证

## API权限设置

在Binance创建API密钥时:

- ✅ 启用: 读取、现货与杠杆交易
- ❌ 禁用: 提币、内部转账
- ✅ 限制IP(推荐)

## 测试计划

### 第1周: 小额测试

- 资金: 50 USDT
- 持仓: 1个币种
- 观察: 网格运行、止损触发

### 第2-4周: 观察期

- 如正常,增加到100 USDT
- 记录交易日志
- 优化参数

### 第2个月: 扩容

- 如稳定,逐步扩展到3个币种
- 增加到目标规模(900 USDT)

## 常见问题

### Q: 如何查看当前持仓?

```bash
# 查看日志
grep "持仓数量" logs/trading_bot_*.log

# 查询数据库
sqlite3 data/database.db "SELECT * FROM positions ORDER BY timestamp DESC LIMIT 5;"
```

### Q: 如何手动平仓?

```python
# 进入Python环境
from QuantSystem.core.grid_strategy import GridStrategy
# 手动调用close_grid方法
```

### Q: 如何停止机器人?

```bash
# 发送SIGTERM信号(会优雅关闭)
kill -TERM <pid>

# 或使用Ctrl+C

# Docker
docker stop quantbot
```

### Q: 如何备份数据?

```bash
# 备份数据库
cp data/database.db data/backup/database_$(date +%Y%m%d).db

# 备份日志
tar -czf logs_$(date +%Y%m%d).tar.gz logs/
```

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 代码格式化

```bash
black QuantSystem/
```

## 许可证

本项目仅供学习研究使用,请勿用于实际交易。作者不对任何交易损失负责。

## 联系方式

如有问题,请提交Issue。

---

**风险提示**: 量化交易存在风险,本系统不保证盈利。请谨慎使用,自担风险。
