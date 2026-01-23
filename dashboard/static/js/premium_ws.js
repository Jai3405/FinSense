// SPIKE Terminal - WebSocket (FastAPI) Version

let ws = null;
let equityChart = null;
let actionsChart = null;
let portfolioHistory = [];
let actionCounts = { BUY: 0, HOLD: 0, SELL: 0 };
let activityCount = 0;
let startingBalance = 50000;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    connectWebSocket();
    animateMetrics();
});

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log('Connecting to WebSocket:', wsUrl);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        addSystemMessage('Connection established', 'info');
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        addSystemMessage('Connection lost', 'warning');
        // Auto-reconnect after 2 seconds
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        addSystemMessage('Connection error', 'danger');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'trading_update') {
            updateTerminal(data);
        } else if (data.status === 'started') {
            console.log('Simulation started');
        } else if (data.status === 'complete') {
            completeTrading();
        } else if (data.error) {
            showErrorBanner('Error: ' + data.error, 'danger');
            resetUI();
        }
    };
}

function initCharts() {
    // Advanced Equity Chart with Gradient
    const equityCtx = document.getElementById('equity-chart').getContext('2d');

    const gradient = equityCtx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(99, 140, 130, 0.4)');
    gradient.addColorStop(1, 'rgba(99, 140, 130, 0.0)');

    equityChart = new Chart(equityCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: '#638C82',
                backgroundColor: gradient,
                borderWidth: 3,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: '#D3E9D7',
                pointHoverBorderColor: '#638C82',
                pointHoverBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(20, 25, 32, 0.9)',
                    titleColor: '#D3E9D7',
                    bodyColor: '#FFFFFF',
                    borderColor: '#638C82',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return '₹' + context.parsed.y.toLocaleString('en-IN', { maximumFractionDigits: 2 });
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.03)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#7D8590',
                        font: {
                            size: 11,
                            family: 'JetBrains Mono'
                        },
                        maxRotation: 0
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.03)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#7D8590',
                        font: {
                            size: 11,
                            family: 'JetBrains Mono'
                        },
                        callback: function(value) {
                            return '₹' + value.toLocaleString('en-IN', { maximumFractionDigits: 0 });
                        }
                    }
                }
            }
        }
    });

    // Actions Chart
    const actionsCtx = document.getElementById('actions-chart').getContext('2d');

    actionsChart = new Chart(actionsCtx, {
        type: 'bar',
        data: {
            labels: ['BUY', 'HOLD', 'SELL'],
            datasets: [{
                label: 'Actions',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(61, 214, 140, 0.8)',
                    'rgba(255, 165, 2, 0.8)',
                    'rgba(255, 71, 87, 0.8)'
                ],
                borderColor: [
                    '#3DD68C',
                    '#FFA502',
                    '#FF4757'
                ],
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#7D8590',
                        font: {
                            size: 12,
                            family: 'JetBrains Mono',
                            weight: '600'
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.03)'
                    },
                    ticks: {
                        color: '#7D8590',
                        font: {
                            size: 11,
                            family: 'JetBrains Mono'
                        },
                        stepSize: 1
                    }
                }
            }
        }
    });
}

function startTrading() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    const capital = parseFloat(document.getElementById('capital').value);

    startingBalance = capital;

    // Show loading state
    showLoading('Initializing AI Agent...');

    // Reset state
    portfolioHistory = [];
    actionCounts = { BUY: 0, HOLD: 0, SELL: 0 };
    activityCount = 0;

    equityChart.data.labels = [];
    equityChart.data.datasets[0].data = [];
    equityChart.update('none');

    actionsChart.data.datasets[0].data = [0, 0, 0];
    actionsChart.update('none');

    document.getElementById('activity-feed').innerHTML = '';

    // Send start command via WebSocket
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            command: 'start',
            start_date: startDate,
            end_date: endDate,
            balance: capital
        }));

        hideLoading();

        // Update UI
        document.getElementById('system-status').classList.add('active');
        document.getElementById('status-label').textContent = 'Running';
        document.getElementById('start-btn').disabled = true;
        document.getElementById('stop-btn').disabled = false;
        addSystemMessage('Simulation started', 'success');
    } else {
        hideLoading();
        showErrorBanner('WebSocket not connected', 'danger');
    }
}

function stopTrading() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ command: 'stop' }));
    }
    completeTrading();
}

function completeTrading() {
    document.getElementById('system-status').classList.remove('active');
    document.getElementById('status-label').textContent = 'Complete';
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
    addSystemMessage('Simulation complete', 'success');
}

function resetUI() {
    document.getElementById('system-status').classList.remove('active');
    document.getElementById('status-label').textContent = 'System Idle';
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
}

function updateTerminal(data) {
    const metrics = data.metrics;
    const timestamp = new Date(data.timestamp).toLocaleDateString('en-IN');

    // Track portfolio history for export
    portfolioHistory.push({
        timestamp: data.timestamp,
        price: data.price,
        portfolio: metrics.portfolio_value,
        action: data.action,
        pnl: data.trade && data.trade.pnl ? data.trade.pnl : 0
    });

    // Update header ticker
    document.getElementById('live-price').textContent = '₹' + data.price.toFixed(2);
    document.getElementById('live-portfolio').textContent = '₹' + metrics.portfolio_value.toLocaleString('en-IN', { maximumFractionDigits: 0 });

    const pnl = metrics.portfolio_value - startingBalance;
    const pnlEl = document.getElementById('live-pnl');
    pnlEl.textContent = (pnl >= 0 ? '+' : '') + '₹' + pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 });
    pnlEl.className = 'ticker-value ' + (pnl >= 0 ? 'up' : 'down');

    // Update main metrics
    document.getElementById('metric-portfolio').textContent =
        '₹' + metrics.portfolio_value.toLocaleString('en-IN', { maximumFractionDigits: 0 });

    const changeEl = document.getElementById('metric-portfolio-change');
    const changePct = metrics.total_return_pct;
    changeEl.textContent = (changePct >= 0 ? '▲ +' : '▼ ') + Math.abs(changePct).toFixed(2) + '%';
    changeEl.className = 'metric-change-pro ' + (changePct >= 0 ? 'positive' : 'negative');

    document.getElementById('metric-sharpe').textContent = metrics.sharpe_ratio.toFixed(3);
    document.getElementById('metric-winrate').textContent = metrics.win_rate.toFixed(1) + '%';
    document.getElementById('metric-drawdown').textContent = metrics.max_drawdown.toFixed(2) + '%';

    // Update control panel stats
    document.getElementById('current-step').textContent = data.step;
    document.getElementById('cash-balance').textContent = '₹' + metrics.balance.toLocaleString('en-IN', { maximumFractionDigits: 0 });
    document.getElementById('shares-held').textContent = metrics.inventory;
    document.getElementById('position-value').textContent = '₹' + (metrics.inventory_value || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 });
    document.getElementById('total-trades').textContent = metrics.total_trades;

    // Update position details
    document.getElementById('pos-price').textContent = '₹' + data.price.toFixed(2);
    document.getElementById('pos-inventory').textContent = metrics.inventory;
    document.getElementById('pos-avgcost').textContent = '₹' + (metrics.avg_cost || 0).toFixed(2);

    const unrealizedPnl = metrics.unrealized_pnl || 0;
    const unrealizedEl = document.getElementById('pos-unrealized');
    unrealizedEl.textContent = (unrealizedPnl >= 0 ? '+' : '') + '₹' + unrealizedPnl.toFixed(2);
    unrealizedEl.style.color = unrealizedPnl >= 0 ? 'var(--success)' : 'var(--danger)';

    const realizedPnl = (metrics.total_profit || 0) - Math.abs(metrics.total_loss || 0);
    const realizedEl = document.getElementById('pos-realized');
    realizedEl.textContent = (realizedPnl >= 0 ? '+' : '') + '₹' + realizedPnl.toFixed(2);
    realizedEl.style.color = realizedPnl >= 0 ? 'var(--success)' : 'var(--danger)';

    // Update AI confidence levels
    const probs = data.action_probs;
    document.getElementById('conf-buy').style.width = (probs[0] * 100) + '%';
    document.getElementById('conf-hold').style.width = (probs[1] * 100) + '%';
    document.getElementById('conf-sell').style.width = (probs[2] * 100) + '%';
    document.getElementById('conf-buy-val').textContent = (probs[0] * 100).toFixed(1) + '%';
    document.getElementById('conf-hold-val').textContent = (probs[1] * 100).toFixed(1) + '%';
    document.getElementById('conf-sell-val').textContent = (probs[2] * 100).toFixed(1) + '%';

    // Update equity chart
    equityChart.data.labels.push(timestamp);
    equityChart.data.datasets[0].data.push(metrics.portfolio_value);

    if (equityChart.data.labels.length > 60) {
        equityChart.data.labels.shift();
        equityChart.data.datasets[0].data.shift();
    }

    equityChart.update('none');

    // Update actions chart
    actionCounts[data.action]++;
    actionsChart.data.datasets[0].data = [
        actionCounts.BUY,
        actionCounts.HOLD,
        actionCounts.SELL
    ];
    actionsChart.update('none');

    // Add activity for trades
    if (data.action === 'BUY' || data.action === 'SELL') {
        const pnl = data.trade && data.trade.pnl ? data.trade.pnl : 0;
        addActivity(data.action, data.price, pnl, data.timestamp);
    }
}

function addActivity(action, price, pnl, timestamp) {
    const feed = document.getElementById('activity-feed');
    const item = document.createElement('div');
    item.className = `activity-item ${action.toLowerCase()}`;

    const actionClass = action === 'BUY' ? 'buy' : action === 'SELL' ? 'sell' : 'hold';

    item.innerHTML = `
        <div class="activity-time">${new Date(timestamp).toLocaleTimeString('en-IN')}</div>
        <div class="activity-content">
            <span class="activity-action">${action} @ ₹${price.toFixed(2)}</span>
            ${action === 'SELL' ? `<span class="activity-pnl ${pnl >= 0 ? 'positive' : 'negative'}">
                ${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}
            </span>` : ''}
        </div>
    `;

    feed.insertBefore(item, feed.firstChild);

    activityCount++;
    document.getElementById('activity-count').textContent = activityCount;

    // Limit feed size
    while (feed.children.length > 50) {
        feed.removeChild(feed.lastChild);
    }
}

function addSystemMessage(message, type) {
    const feed = document.getElementById('activity-feed');
    const item = document.createElement('div');
    item.className = `activity-item ${type}`;

    item.innerHTML = `
        <div class="activity-time">${new Date().toLocaleTimeString('en-IN')}</div>
        <div class="activity-content">
            <span class="activity-action">${message}</span>
        </div>
    `;

    feed.insertBefore(item, feed.firstChild);
}

function showLoading(message) {
    const overlay = document.getElementById('loading-overlay');
    const text = overlay.querySelector('.spinner-text');
    text.textContent = message;
    overlay.style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
}

function showErrorBanner(message, severity) {
    const banner = document.createElement('div');
    banner.className = `error-banner ${severity}`;
    banner.innerHTML = `
        <span class="error-icon">[!]</span>
        <span class="error-message">${message}</span>
        <button class="error-close" onclick="this.parentElement.remove()">×</button>
    `;
    document.body.appendChild(banner);

    setTimeout(() => banner.remove(), 5000);
}

function exportTradesCSV() {
    if (portfolioHistory.length === 0) {
        showErrorBanner('No trading data to export', 'warning');
        return;
    }

    const csvRows = ['Timestamp,Price,Portfolio Value,Action,P&L'];

    portfolioHistory.forEach(point => {
        csvRows.push(`${point.timestamp},${point.price},${point.portfolio},${point.action || 'HOLD'},${point.pnl || 0}`);
    });

    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `spike_trades_${new Date().toISOString().split('T')[0]}_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);

    showErrorBanner('Trading data exported successfully', 'success');
}

function animateMetrics() {
    // Subtle pulsing animation for live indicators
    const indicators = document.querySelectorAll('.status-dot.active');
    indicators.forEach(ind => {
        ind.style.animation = 'pulse-status 2s infinite';
    });
}
