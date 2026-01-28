// 全局变量
let currentSymbol = null;
let refreshInterval = null;

// API 基础 URL
const API_BASE = window.location.origin;

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    updateCurrentTime();
    setInterval(updateCurrentTime, 1000);

    // 首次加载数据
    loadAllData();

    // 每5秒自动刷新
    refreshInterval = setInterval(loadAllData, 5000);
});

// 更新当前时间
function updateCurrentTime() {
    const now = new Date();
    document.getElementById('currentTime').textContent = now.toLocaleString('zh-CN');
}

// 加载所有数据
async function loadAllData() {
    try {
        await Promise.all([
            loadAccountData(),
            loadPositionsData(),
            loadProfitData(),
            loadAlertsData(),
            loadSystemData(),
            loadPerformanceData()
        ]);
    } catch (error) {
        console.error('加载数据失败:', error);
    }
}

// 加载账户数据
async function loadAccountData() {
    try {
        const response = await fetch(`${API_BASE}/api/account`);
        const data = await response.json();

        document.getElementById('totalBalance').textContent = data.total_balance.toFixed(2);
        document.getElementById('availableMargin').textContent = data.available_margin.toFixed(2);

        const pnlElement = document.getElementById('unrealizedPnl');
        pnlElement.textContent = data.total_unrealized_pnl.toFixed(2);
        pnlElement.className = 'card-title ' + (data.total_unrealized_pnl >= 0 ? 'text-success' : 'text-danger');

        document.getElementById('usagePercentage').textContent = data.usage_percentage.toFixed(1);
    } catch (error) {
        console.error('加载账户数据失败:', error);
    }
}

// 加载持仓数据
async function loadPositionsData() {
    try {
        const response = await fetch(`${API_BASE}/api/positions`);
        const data = await response.json();

        const tbody = document.getElementById('positionsBody');

        if (data.positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="9" class="text-center">暂无持仓</td></tr>';
            return;
        }

        tbody.innerHTML = data.positions.map(pos => `
            <tr>
                <td><strong>${pos.symbol}</strong></td>
                <td>${pos.entry_price.toFixed(4)}</td>
                <td>${pos.current_price.toFixed(4)}</td>
                <td>${pos.size.toFixed(4)}</td>
                <td class="${pos.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}">
                    ${pos.unrealized_pnl.toFixed(2)}
                </td>
                <td class="${pos.profit_percentage >= 0 ? 'text-success' : 'text-danger'}">
                    ${pos.profit_percentage.toFixed(2)}%
                </td>
                <td>${pos.peak_profit_percentage.toFixed(2)}%</td>
                <td>${pos.margin_used.toFixed(2)}</td>
                <td>
                    <button class="btn btn-sm btn-primary" onclick="viewGrid('${pos.symbol}')">
                        <i class="bi bi-grid"></i>
                    </button>
                </td>
            </tr>
        `).join('');
    } catch (error) {
        console.error('加载持仓数据失败:', error);
    }
}

// 查看网格详情
async function viewGrid(symbol) {
    currentSymbol = symbol;

    try {
        const response = await fetch(`${API_BASE}/api/positions/${symbol}/grid`);
        const data = await response.json();

        if (data.error) {
            document.getElementById('gridStatus').innerHTML = `
                <p class="text-danger">加载失败: ${data.error}</p>
            `;
            return;
        }

        document.getElementById('gridStatus').innerHTML = `
            <h6>${symbol} 网格状态</h6>
            <div class="row">
                <div class="col-6">
                    <p><strong>入场价 (P0):</strong> ${data.entry_price.toFixed(4)}</p>
                    <p><strong>当前价:</strong> ${data.current_price.toFixed(4)}</p>
                    <p><strong>网格间距:</strong> ${(data.grid_spacing * 100).toFixed(2)}%</p>
                    <p><strong>止损价:</strong> ${data.stop_loss_price.toFixed(4)}</p>
                </div>
                <div class="col-6">
                    <p><strong>上网格:</strong> ${data.upper_grids.active} 活跃, ${data.upper_grids.filled} 已成交</p>
                    <p><strong>上网格成功率:</strong> ${(data.upper_grids.success_rate * 100).toFixed(1)}%</p>
                    <p><strong>下网格:</strong> ${data.lower_grids.active} 活跃</p>
                    <p><strong>下网格成功率:</strong> ${(data.lower_grids.success_rate * 100).toFixed(1)}%</p>
                </div>
            </div>
            <small class="text-muted">最后更新: ${new Date(data.last_update).toLocaleString('zh-CN')}</small>
        `;
    } catch (error) {
        console.error('加载网格数据失败:', error);
    }
}

// 加载盈利数据
async function loadProfitData() {
    try {
        const response = await fetch(`${API_BASE}/api/profit`);
        const data = await response.json();

        if (data.error) {
            document.getElementById('profitMonitor').innerHTML = `
                <p class="text-danger">加载失败: ${data.error}</p>
            `;
            return;
        }

        document.getElementById('profitMonitor').innerHTML = `
            <div class="row">
                <div class="col-6">
                    <p><strong>监控币种数:</strong> ${data.monitored_symbols}</p>
                    <p><strong>平均盈利率:</strong> <span class="${data.avg_profit >= 0 ? 'text-success' : 'text-danger'}">${data.avg_profit.toFixed(2)}%</span></p>
                    <p><strong>达标币种数:</strong> ${data.symbols_over_threshold}</p>
                </div>
                <div class="col-6">
                    <p><strong>最高盈利:</strong> <span class="text-success">${data.max_profit.toFixed(2)}%</span></p>
                    <p><strong>最低盈利:</strong> <span class="text-danger">${data.min_profit.toFixed(2)}%</span></p>
                    <p><strong>盈利阈值:</strong> ${data.profit_threshold.toFixed(2)}%</p>
                </div>
            </div>
            ${data.symbol_profits.length > 0 ? `
                <hr>
                <h6>盈利排行:</h6>
                <ul class="list-unstyled">
                    ${data.symbol_profits.slice(0, 5).map(sp => `
                        <li>${sp.symbol}: <span class="${sp.profit_percentage >= 0 ? 'text-success' : 'text-danger'}">${sp.profit_percentage.toFixed(2)}%</span></li>
                    `).join('')}
                </ul>
            ` : ''}
        `;
    } catch (error) {
        console.error('加载盈利数据失败:', error);
    }
}

// 加载告警数据
async function loadAlertsData() {
    try {
        const response = await fetch(`${API_BASE}/api/alerts`);
        const data = await response.json();

        if (data.error) {
            document.getElementById('alertsBody').innerHTML = `
                <p class="text-danger">加载失败: ${data.error}</p>
            `;
            return;
        }

        if (data.recent_alerts.length === 0) {
            document.getElementById('alertsBody').innerHTML = `
                <p class="text-muted">暂无告警</p>
            `;
            return;
        }

        document.getElementById('alertsBody').innerHTML = data.recent_alerts.slice(0, 10).map(alert => {
            let badgeClass = 'bg-info';
            if (alert.level === 2) badgeClass = 'bg-warning';
            if (alert.level === 3) badgeClass = 'bg-danger';

            return `
                <div class="alert alert-sm mb-2">
                    <span class="badge ${badgeClass}">${alert.level === 1 ? '信息' : alert.level === 2 ? '警告' : '严重'}</span>
                    <strong>${alert.symbol || '系统'}</strong>: ${alert.message}
                    <br><small class="text-muted">${new Date(alert.timestamp).toLocaleString('zh-CN')}</small>
                </div>
            `;
        }).join('');
    } catch (error) {
        console.error('加载告警数据失败:', error);
    }
}

// 加载系统状态
async function loadSystemData() {
    try {
        const response = await fetch(`${API_BASE}/api/system`);
        const data = await response.json();

        const statusBadge = data.running ?
            '<span class="badge bg-success">运行中</span>' :
            '<span class="badge bg-danger">已停止</span>';

        document.getElementById('systemStatus').innerHTML = statusBadge;
    } catch (error) {
        console.error('加载系统状态失败:', error);
        document.getElementById('systemStatus').innerHTML = '<span class="badge bg-secondary">未知</span>';
    }
}

// 加载性能数据
async function loadPerformanceData() {
    try {
        const response = await fetch(`${API_BASE}/api/performance`);
        const data = await response.json();

        if (data.error) {
            console.error('加载性能数据失败:', data.error);
            return;
        }

        // 更新图表 (在 charts.js 中实现)
        if (typeof updatePerformanceChart === 'function') {
            updatePerformanceChart(data);
        }
    } catch (error) {
        console.error('加载性能数据失败:', error);
    }
}
