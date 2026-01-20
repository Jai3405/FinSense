// FinSense Dashboard JavaScript - Real-time Updates

const socket = io();

// Charts
let equityChart = null;
let actionsChart = null;

// Data storage
let equityData = [];
let actionCounts = { BUY: 0, HOLD: 0, SELL: 0 };

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    setupSocketListeners();
});

function initCharts() {
    // Equity Curve Chart
    const equityCtx = document.getElementById('equity-chart').getContext('2d');
    equityChart = new Chart(equityCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: '#638C82',
                backgroundColor: 'rgba(99, 140, 130, 0.1)',
                borderWidth: 3,
                tension: 0.4,
                fill: true
            }, {
                label: 'Starting Value',
                data: [],
                borderColor: '#D3E9D7',
                borderWidth: 2,
                borderDash: [5, 5],
                fill: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#ffffff' }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#b0b0b0' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                },
                y: {
                    ticks: {
                        color: '#b0b0b0',
                        callback: function(value) {
                            return '₹' + value.toLocaleString();
                        }
                    },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' }
                }
            }
        }
    });

    // Actions Distribution Chart
    const actionsCtx = document.getElementById('actions-chart').getContext('2d');
    actionsChart = new Chart(actionsCtx, {
        type: 'doughnut',
        data: {
            labels: ['BUY', 'HOLD', 'SELL'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    '#4ade80',
                    '#fbbf24',
                    '#f87171'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff',
                        font: { size: 14 }
                    }
                }
            }
        }
    });
}

function setupSocketListeners() {
    socket.on('connect', function() {
        console.log('Connected to server');
    });

    socket.on('trading_update', function(data) {
        updateDashboard(data);
    });

    socket.on('trading_complete', function(data) {
        addLogEntry('info', 'Simulation complete');
        document.getElementById('status-text').textContent = 'Complete';
        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
        document.getElementById('status-indicator').classList.remove('active');
    });

    socket.on('error', function(data) {
        addLogEntry('info', `Error: ${data.message}`);
        alert(`Error: ${data.message}`);
    });
}

function startTrading() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    const balance = parseFloat(document.getElementById('balance').value);

    // Reset
    equityData = [];
    actionCounts = { BUY: 0, HOLD: 0, SELL: 0 };
    equityChart.data.labels = [];
    equityChart.data.datasets[0].data = [];
    equityChart.data.datasets[1].data = [];
    equityChart.update();
    actionsChart.data.datasets[0].data = [0, 0, 0];
    actionsChart.update();
    document.getElementById('trade-log').innerHTML = '';

    // Send start request
    fetch('/api/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start_date: startDate, end_date: endDate, balance: balance })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('status-text').textContent = 'Running';
            document.getElementById('status-indicator').classList.add('active');
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            addLogEntry('info', 'Simulation started');
        } else {
            alert('Error starting: ' + data.error);
        }
    })
    .catch(error => {
        alert('Error: ' + error);
    });
}

function stopTrading() {
    fetch('/api/stop', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
        document.getElementById('status-text').textContent = 'Stopped';
        document.getElementById('status-indicator').classList.remove('active');
        document.getElementById('start-btn').disabled = false;
        document.getElementById('stop-btn').disabled = true;
        addLogEntry('info', 'Simulation stopped by user');
    });
}

function updateDashboard(data) {
    // Update metrics
    const metrics = data.metrics;

    document.getElementById('portfolio-value').textContent =
        '₹' + metrics.portfolio_value.toLocaleString('en-IN', { maximumFractionDigits: 2 });

    const change = metrics.total_return_pct;
    const changeEl = document.getElementById('portfolio-change');
    changeEl.textContent = (change >= 0 ? '+' : '') + change.toFixed(2) + '%';
    changeEl.className = 'metric-change ' + (change >= 0 ? 'positive' : 'negative');

    document.getElementById('sharpe').textContent = metrics.sharpe_ratio.toFixed(4);
    document.getElementById('drawdown').textContent = metrics.max_drawdown.toFixed(2) + '%';
    document.getElementById('win-rate').textContent = metrics.win_rate.toFixed(1) + '%';

    document.getElementById('current-price').textContent =
        '₹' + data.price.toFixed(2);
    document.getElementById('inventory').textContent = metrics.inventory;
    document.getElementById('balance').textContent =
        '₹' + metrics.balance.toLocaleString('en-IN', { maximumFractionDigits: 2 });
    document.getElementById('total-trades').textContent = metrics.total_trades;

    // Update equity chart
    const timestamp = new Date(data.timestamp).toLocaleDateString('en-IN');
    equityChart.data.labels.push(timestamp);
    equityChart.data.datasets[0].data.push(metrics.portfolio_value);
    equityChart.data.datasets[1].data.push(parseFloat(document.getElementById('balance').getAttribute('data-starting') || 50000));

    // Keep only last 50 points for performance
    if (equityChart.data.labels.length > 50) {
        equityChart.data.labels.shift();
        equityChart.data.datasets[0].data.shift();
        equityChart.data.datasets[1].data.shift();
    }

    equityChart.update('none'); // No animation for better performance

    // Update actions chart
    actionCounts[data.action]++;
    actionsChart.data.datasets[0].data = [
        actionCounts.BUY,
        actionCounts.HOLD,
        actionCounts.SELL
    ];
    actionsChart.update('none');

    // Add log entry for trades
    if (data.action === 'BUY' || data.action === 'SELL') {
        const pnl = data.trade && data.trade.pnl ? data.trade.pnl : 0;
        const message = `${data.action} @ ₹${data.price.toFixed(2)}` +
                       (data.action === 'SELL' ? ` | P&L: ₹${pnl.toFixed(2)}` : '');
        addLogEntry(data.action.toLowerCase(), message, pnl);
    }
}

function addLogEntry(type, message, pnl = null) {
    const log = document.getElementById('trade-log');
    const entry = document.createElement('div');
    entry.className = `trade-item ${type}`;

    const timestamp = new Date().toLocaleTimeString('en-IN');

    let html = `
        <span class="timestamp">${timestamp}</span>
        <span class="message">${message}</span>
    `;

    if (pnl !== null && type === 'sell') {
        html += `<span class="pnl ${pnl >= 0 ? 'positive' : 'negative'}">${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}</span>`;
    }

    entry.innerHTML = html;
    log.insertBefore(entry, log.firstChild);

    // Keep only last 100 entries
    while (log.children.length > 100) {
        log.removeChild(log.lastChild);
    }
}
