// FinSense Premium Terminal - Advanced Trading Dashboard

const socket = io();

// Chart instances
let equityChart = null;
let actionsChart = null;

// Data storage
let portfolioHistory = [];
let actionCounts = { BUY: 0, HOLD: 0, SELL: 0 };
let activityCount = 0;
let startingBalance = 50000;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    setupSocketListeners();
    animateMetrics();
});

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
                            return '₹' + (value / 1000).toFixed(0) + 'k';
                        }
                    }
                }
            },
            animation: {
                duration: 750,
                easing: 'easeInOutQuart'
            }
        }
    });

    // Advanced Doughnut Chart
    const actionsCtx = document.getElementById('actions-chart').getContext('2d');

    actionsChart = new Chart(actionsCtx, {
        type: 'doughnut',
        data: {
            labels: ['BUY', 'HOLD', 'SELL'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    '#3DD68C',
                    '#FFA502',
                    '#FF4757'
                ],
                borderWidth: 0,
                hoverOffset: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#B8BEC6',
                        font: {
                            size: 12,
                            family: 'Inter',
                            weight: '600'
                        },
                        padding: 15,
                        usePointStyle: true,
                        pointStyle: 'circle'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(20, 25, 32, 0.9)',
                    titleColor: '#D3E9D7',
                    bodyColor: '#FFFFFF',
                    borderColor: '#638C82',
                    borderWidth: 1,
                    padding: 12,
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                            return `${label}: ${value} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

function setupSocketListeners() {
    socket.on('connect', function() {
        console.log('Connected to FinSense Terminal');
        addSystemMessage('Connection established', 'info');
    });

    socket.on('trading_update', function(data) {
        updateTerminal(data);
    });

    socket.on('trading_complete', function(data) {
        completeTrading();
    });

    socket.on('error', function(data) {
        addSystemMessage(`Error: ${data.message}`, 'danger');
    });
}

function startTrading() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    const capital = parseFloat(document.getElementById('capital').value);

    startingBalance = capital;

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

    // Update UI
    document.getElementById('system-status').classList.add('active');
    document.getElementById('status-label').textContent = 'Running';
    document.getElementById('start-btn').disabled = true;
    document.getElementById('stop-btn').disabled = false;

    // Send request
    fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            start_date: startDate,
            end_date: endDate,
            balance: capital
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addSystemMessage('Simulation started', 'success');
        } else {
            alert('Error: ' + data.error);
            resetUI();
        }
    })
    .catch(error => {
        alert('Connection error: ' + error);
        resetUI();
    });
}

function stopTrading() {
    fetch('/api/stop', { method: 'POST' })
    .then(() => {
        completeTrading();
    });
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
    document.getElementById('position-value').textContent = '₹' + metrics.inventory_value.toLocaleString('en-IN', { maximumFractionDigits: 0 });
    document.getElementById('total-trades').textContent = metrics.total_trades;

    // Update position details
    document.getElementById('pos-price').textContent = '₹' + data.price.toFixed(2);
    document.getElementById('pos-inventory').textContent = metrics.inventory;

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
    activityCount++;
    document.getElementById('activity-count').textContent = activityCount;

    const feed = document.getElementById('activity-feed');
    const item = document.createElement('div');
    item.className = `activity-item ${action.toLowerCase()}`;

    const time = new Date(timestamp).toLocaleTimeString('en-IN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });

    const actionText = action === 'BUY'
        ? `BUY @ ₹${price.toFixed(2)}`
        : `SELL @ ₹${price.toFixed(2)}`;

    item.innerHTML = `
        <div class="activity-time">${time}</div>
        <div class="activity-content">
            <span class="activity-action">${actionText}</span>
            ${action === 'SELL' ? `<span class="activity-pnl ${pnl >= 0 ? 'positive' : 'negative'}">${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}</span>` : ''}
        </div>
    `;

    feed.insertBefore(item, feed.firstChild);

    // Keep only last 50
    while (feed.children.length > 50) {
        feed.removeChild(feed.lastChild);
    }
}

function addSystemMessage(message, type = 'info') {
    const feed = document.getElementById('activity-feed');
    const item = document.createElement('div');
    item.className = `activity-item ${type}`;

    const time = new Date().toLocaleTimeString('en-IN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });

    item.innerHTML = `
        <div class="activity-time">${time}</div>
        <div class="activity-content">
            <span class="activity-action">${message}</span>
        </div>
    `;

    feed.insertBefore(item, feed.firstChild);
}

// Animate metrics on load
function animateMetrics() {
    const metrics = document.querySelectorAll('.metric-value-pro');
    metrics.forEach((metric, index) => {
        setTimeout(() => {
            metric.style.opacity = '0';
            metric.style.transform = 'translateY(10px)';
            setTimeout(() => {
                metric.style.transition = 'all 0.5s ease';
                metric.style.opacity = '1';
                metric.style.transform = 'translateY(0)';
            }, 50);
        }, index * 100);
    });
}

// Sound effects (optional - can add trading beeps)
function playSound(type) {
    // Add sound effects for trades if desired
    // const audio = new Audio(`/static/sounds/${type}.mp3`);
    // audio.play();
}
