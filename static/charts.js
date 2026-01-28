// 性能图表实例
let performanceChart = null;

// 初始化性能图表
function initPerformanceChart() {
    const ctx = document.getElementById('performanceChart');
    if (!ctx) return;

    performanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['日盈亏', '周盈亏', '月盈亏'],
            datasets: [{
                label: '盈亏 (USDT)',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.6)',
                    'rgba(54, 162, 235, 0.6)',
                    'rgba(153, 102, 255, 0.6)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(153, 102, 255, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(2) + ' USDT';
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(2) + ' USDT';
                        }
                    }
                }
            }
        }
    });
}

// 更新性能图表
function updatePerformanceChart(data) {
    if (!performanceChart) {
        initPerformanceChart();
    }

    if (performanceChart) {
        performanceChart.data.datasets[0].data = [
            data.daily_pnl || 0,
            data.weekly_pnl || 0,
            data.monthly_pnl || 0
        ];

        // 根据盈亏设置颜色
        performanceChart.data.datasets[0].backgroundColor = [
            data.daily_pnl >= 0 ? 'rgba(75, 192, 192, 0.6)' : 'rgba(255, 99, 132, 0.6)',
            data.weekly_pnl >= 0 ? 'rgba(54, 162, 235, 0.6)' : 'rgba(255, 99, 132, 0.6)',
            data.monthly_pnl >= 0 ? 'rgba(153, 102, 255, 0.6)' : 'rgba(255, 99, 132, 0.6)'
        ];

        performanceChart.update();
    }
}

// 页面加载时初始化图表
document.addEventListener('DOMContentLoaded', function() {
    initPerformanceChart();
});
