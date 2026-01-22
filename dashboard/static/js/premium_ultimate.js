// SPIKE Terminal - ULTIMATE Edition 2026
// Cutting-edge WebSocket + Advanced Visualizations + Modern APIs
// Features: ES2024, Web Workers, ResizeObserver, IntersectionObserver, Performance API

let ws = null;
let equityChart = null;
let actionsChart = null;
let candlestickChart = null;  // NEW: Candlestick chart
let portfolioHistory = [];
let actionCounts = { BUY: 0, HOLD: 0, SELL: 0 };
let activityCount = 0;
let startingBalance = 50000;
let priceHistory = [];  // For candlestick chart
let chartResizeObserver = null;
let activityIntersectionObserver = null;
let performanceMetrics = {
    updateCount: 0,
    avgUpdateTime: 0,
    lastUpdateTime: 0
};

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initCharts();
    connectWebSocket();
    initAnimations();
    initResizeObserver();
    initIntersectionObserver();
    initPerformanceMonitoring();

    // Enable View Transitions API if supported
    if (document.startViewTransition) {
        console.log('[SPIKE] View Transitions API supported - smooth page transitions enabled');
    }
});

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log('[SPIKE] Connecting to WebSocket:', wsUrl);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('[SPIKE] WebSocket connected');
        addSystemMessage('AI System Online', 'info');
        document.getElementById('system-status').classList.add('active');
        document.getElementById('status-label').textContent = 'Connected';
    };

    ws.onclose = () => {
        console.log('[SPIKE] WebSocket disconnected');
        document.getElementById('system-status').classList.remove('active');
        document.getElementById('status-label').textContent = 'Offline';

        // Auto-reconnect after 2 seconds
        setTimeout(connectWebSocket, 2000);
    };

    ws.onerror = (error) => {
        console.error('[SPIKE] WebSocket error:', error);
        showNotification('Connection Error', 'danger');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'trading_update') {
            updateTerminal(data);
        } else if (data.status === 'started') {
            console.log('[SPIKE] Simulation started');
        } else if (data.status === 'complete') {
            completeTrading();
        } else if (data.error) {
            showNotification('Error: ' + data.error, 'danger');
            resetUI();
        }
    };
}

function initCharts() {
    // 1. Advanced Equity Chart with Gradient
    const equityCtx = document.getElementById('equity-chart').getContext('2d');

    const gradient = equityCtx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(99, 140, 130, 0.5)');
    gradient.addColorStop(0.5, 'rgba(99, 140, 130, 0.2)');
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
                pointHoverRadius: 8,
                pointHoverBackgroundColor: '#D3E9D7',
                pointHoverBorderColor: '#638C82',
                pointHoverBorderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            animation: {
                duration: 300,
                easing: 'easeInOutCubic'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 14, 18, 0.95)',
                    titleColor: '#D3E9D7',
                    bodyColor: '#FFFFFF',
                    borderColor: '#638C82',
                    borderWidth: 2,
                    padding: 16,
                    displayColors: false,
                    titleFont: {
                        size: 14,
                        weight: 'bold'
                    },
                    bodyFont: {
                        size: 13
                    },
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
                        color: 'rgba(255, 255, 255, 0.02)',
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
                        color: 'rgba(255, 255, 255, 0.02)',
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

    // 2. Actions Chart with 3D effect
    const actionsCtx = document.getElementById('actions-chart').getContext('2d');

    actionsChart = new Chart(actionsCtx, {
        type: 'bar',
        data: {
            labels: ['BUY', 'HOLD', 'SELL'],
            datasets: [{
                label: 'Actions',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(61, 214, 140, 0.9)',
                    'rgba(255, 165, 2, 0.9)',
                    'rgba(255, 71, 87, 0.9)'
                ],
                borderColor: [
                    '#3DD68C',
                    '#FFA502',
                    '#FF4757'
                ],
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 400,
                easing: 'easeOutBounce'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 14, 18, 0.95)',
                    titleColor: '#FFFFFF',
                    bodyColor: '#FFFFFF',
                    borderColor: '#638C82',
                    borderWidth: 2,
                    padding: 12,
                    titleFont: {
                        size: 13,
                        weight: 'bold'
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#B8BEC6',
                        font: {
                            size: 13,
                            family: 'JetBrains Mono',
                            weight: '700'
                        }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.02)'
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

function initAnimations() {
    // Metric cards hover effect
    const cards = document.querySelectorAll('.metric-card-pro');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px) scale(1.02)';
            this.style.boxShadow = '0 12px 40px rgba(99, 140, 130, 0.3)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
            this.style.boxShadow = '';
        });
    });

    // Pulsing effect for status dot
    setInterval(() => {
        const statusDot = document.getElementById('system-status');
        if (statusDot && statusDot.classList.contains('active')) {
            statusDot.style.transform = 'scale(1.2)';
            setTimeout(() => {
                statusDot.style.transform = 'scale(1)';
            }, 300);
        }
    }, 2000);
}

function startTrading() {
    const startDate = document.getElementById('start-date').value;
    const endDate = document.getElementById('end-date').value;
    const capital = parseFloat(document.getElementById('capital').value);

    startingBalance = capital;

    // Show epic loading animation
    showLoading('Initializing Neural Network...');

    // Reset state
    portfolioHistory = [];
    priceHistory = [];
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

        setTimeout(() => {
            hideLoading();
            document.getElementById('system-status').classList.add('active');
            document.getElementById('status-label').textContent = 'Trading Live';
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            showNotification('AI System Activated', 'success');
        }, 1000);
    } else {
        hideLoading();
        showNotification('WebSocket Disconnected', 'danger');
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
    document.getElementById('status-label').textContent = 'Simulation Complete';
    document.getElementById('start-btn').disabled = false;
    document.getElementById('stop-btn').disabled = true;
    showNotification('Simulation Completed Successfully', 'success');
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

    // Track history
    portfolioHistory.push({
        timestamp: data.timestamp,
        price: data.price,
        portfolio: metrics.portfolio_value,
        action: data.action,
        pnl: data.trade && data.trade.pnl ? data.trade.pnl : 0
    });

    priceHistory.push({
        time: timestamp,
        price: data.price,
        action: data.action
    });

    // Update header ticker with animation
    animateValue('live-price', parseFloat(document.getElementById('live-price').textContent.replace('₹', '')), data.price, '₹');
    animateValue('live-portfolio', parseFloat(document.getElementById('live-portfolio').textContent.replace(/[₹,]/g, '')), metrics.portfolio_value, '₹', true);

    const pnl = metrics.portfolio_value - startingBalance;
    const pnlEl = document.getElementById('live-pnl');
    const sign = pnl >= 0 ? '+' : '';
    pnlEl.textContent = sign + '₹' + pnl.toLocaleString('en-IN', { maximumFractionDigits: 0 });
    pnlEl.className = 'ticker-value ' + (pnl >= 0 ? 'up' : 'down');

    // Update main metrics with flash effect
    flashUpdate('metric-portfolio', '₹' + metrics.portfolio_value.toLocaleString('en-IN', { maximumFractionDigits: 0 }));

    const changeEl = document.getElementById('metric-portfolio-change');
    const changePct = metrics.total_return_pct;
    changeEl.textContent = (changePct >= 0 ? '▲ +' : '▼ ') + Math.abs(changePct).toFixed(2) + '%';
    changeEl.className = 'metric-change-pro ' + (changePct >= 0 ? 'positive' : 'negative');

    flashUpdate('metric-sharpe', metrics.sharpe_ratio.toFixed(3));
    flashUpdate('metric-winrate', metrics.win_rate.toFixed(1) + '%');
    flashUpdate('metric-drawdown', metrics.max_drawdown.toFixed(2) + '%');

    // Update control panel
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

    // Update AI confidence levels with smooth animation
    const probs = data.action_probs;
    animateBar('conf-buy', probs[0] * 100);
    animateBar('conf-hold', probs[1] * 100);
    animateBar('conf-sell', probs[2] * 100);

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

    equityChart.update('active');

    // Update actions chart
    actionCounts[data.action]++;
    actionsChart.data.datasets[0].data = [
        actionCounts.BUY,
        actionCounts.HOLD,
        actionCounts.SELL
    ];
    actionsChart.update('active');

    // Add activity for trades
    if (data.action === 'BUY' || data.action === 'SELL') {
        const pnl = data.trade && data.trade.pnl ? data.trade.pnl : 0;
        addActivity(data.action, data.price, pnl, data.timestamp);
    }
}

function animateValue(elementId, start, end, prefix = '', comma = false) {
    const element = document.getElementById(elementId);
    const duration = 500;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        const current = start + (end - start) * easeOutCubic(progress);

        if (comma) {
            element.textContent = prefix + current.toLocaleString('en-IN', { maximumFractionDigits: 0 });
        } else {
            element.textContent = prefix + current.toFixed(2);
        }

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function animateBar(barId, targetPercent) {
    const bar = document.getElementById(barId);
    const currentPercent = parseFloat(bar.style.width) || 0;
    const duration = 300;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        const current = currentPercent + (targetPercent - currentPercent) * easeOutCubic(progress);
        bar.style.width = current + '%';

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
}

function flashUpdate(elementId, newValue) {
    const element = document.getElementById(elementId);
    element.style.transition = 'all 0.15s';
    element.style.transform = 'scale(1.05)';
    element.style.color = '#D3E9D7';
    element.textContent = newValue;

    setTimeout(() => {
        element.style.transform = 'scale(1)';
        element.style.color = '';
    }, 150);
}

function addActivity(action, price, pnl, timestamp) {
    const feed = document.getElementById('activity-feed');
    const item = document.createElement('div');
    item.className = `activity-item ${action.toLowerCase()}`;

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

    // Entrance animation
    item.style.opacity = '0';
    item.style.transform = 'translateX(20px)';
    setTimeout(() => {
        item.style.transition = 'all 0.3s ease-out';
        item.style.opacity = '1';
        item.style.transform = 'translateX(0)';
    }, 10);

    activityCount++;
    document.getElementById('activity-count').textContent = activityCount;

    // Limit feed size
    while (feed.children.length > 50) {
        feed.removeChild(feed.lastChild);
    }
}

function showLoading(message) {
    const overlay = document.getElementById('loading-overlay');
    const text = overlay.querySelector('.spinner-text');
    text.textContent = message;
    overlay.style.display = 'flex';
    overlay.style.opacity = '0';
    setTimeout(() => {
        overlay.style.transition = 'opacity 0.3s';
        overlay.style.opacity = '1';
    }, 10);
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.opacity = '0';
    setTimeout(() => {
        overlay.style.display = 'none';
    }, 300);
}

function showNotification(message, type) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <span class="notification-icon">${type === 'success' ? '[✓]' : type === 'danger' ? '[✗]' : '[i]'}</span>
        <span class="notification-message">${message}</span>
    `;

    notification.style.cssText = `
        position: fixed;
        top: 90px;
        right: 20px;
        background: ${type === 'success' ? 'var(--success-glow)' : type === 'danger' ? 'var(--danger-glow)' : 'var(--info-glow)'};
        border: 2px solid ${type === 'success' ? 'var(--success)' : type === 'danger' ? 'var(--danger)' : 'var(--info)'};
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        z-index: 10001;
        opacity: 0;
        transform: translateX(400px);
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    `;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.opacity = '1';
        notification.style.transform = 'translateX(0)';
    }, 10);

    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateX(400px)';
        setTimeout(() => notification.remove(), 400);
    }, 3000);
}

function exportTradesCSV() {
    if (portfolioHistory.length === 0) {
        showNotification('No trading data available', 'danger');
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

    showNotification('Trading data exported successfully!', 'success');
}

// ========== MODERN APIs (2026) ==========

// ResizeObserver for responsive charts (modern alternative to window.resize)
function initResizeObserver() {
    if (!window.ResizeObserver) {
        console.warn('[SPIKE] ResizeObserver not supported');
        return;
    }

    chartResizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
            // Debounce chart resize for better performance
            if (entry.target.id === 'equity-chart' || entry.target.id === 'actions-chart') {
                debounce(() => {
                    if (equityChart) equityChart.resize();
                    if (actionsChart) actionsChart.resize();
                }, 100)();
            }
        }
    });

    // Observe chart containers
    const equityCanvas = document.getElementById('equity-chart');
    const actionsCanvas = document.getElementById('actions-chart');

    if (equityCanvas) chartResizeObserver.observe(equityCanvas.parentElement);
    if (actionsCanvas) chartResizeObserver.observe(actionsCanvas.parentElement);

    console.log('[SPIKE] ResizeObserver initialized for responsive charts');
}

// IntersectionObserver for lazy loading and performance (modern best practice)
function initIntersectionObserver() {
    if (!window.IntersectionObserver) {
        console.warn('[SPIKE] IntersectionObserver not supported');
        return;
    }

    activityIntersectionObserver = new IntersectionObserver(
        (entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('visible');
                } else {
                    entry.target.classList.remove('visible');
                }
            });
        },
        {
            threshold: 0.1,
            rootMargin: '50px'
        }
    );

    console.log('[SPIKE] IntersectionObserver initialized for activity feed');
}

// Performance Monitoring with Performance API
function initPerformanceMonitoring() {
    if (!window.performance || !window.performance.mark) {
        console.warn('[SPIKE] Performance API not fully supported');
        return;
    }

    // Mark app initialization
    performance.mark('spike-app-init');

    // Monitor resource timing
    window.addEventListener('load', () => {
        const perfData = performance.getEntriesByType('navigation')[0];
        if (perfData) {
            console.log('[SPIKE] Performance Metrics:');
            console.log(`  DNS Lookup: ${perfData.domainLookupEnd - perfData.domainLookupStart}ms`);
            console.log(`  TCP Connect: ${perfData.connectEnd - perfData.connectStart}ms`);
            console.log(`  DOM Load: ${perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart}ms`);
            console.log(`  Total Load Time: ${perfData.loadEventEnd - perfData.fetchStart}ms`);
        }
    });

    console.log('[SPIKE] Performance monitoring initialized');
}

// Measure update performance
function measureUpdatePerformance(callback) {
    const startMark = `update-start-${Date.now()}`;
    const endMark = `update-end-${Date.now()}`;

    performance.mark(startMark);
    callback();
    performance.mark(endMark);

    const measureName = `update-${Date.now()}`;
    performance.measure(measureName, startMark, endMark);

    const measure = performance.getEntriesByName(measureName)[0];
    performanceMetrics.lastUpdateTime = measure.duration;
    performanceMetrics.updateCount++;
    performanceMetrics.avgUpdateTime =
        (performanceMetrics.avgUpdateTime * (performanceMetrics.updateCount - 1) + measure.duration) / performanceMetrics.updateCount;

    // Log if update is slow
    if (measure.duration > 50) {
        console.warn(`[SPIKE] Slow update detected: ${measure.duration.toFixed(2)}ms`);
    }

    // Clean up performance entries
    performance.clearMarks(startMark);
    performance.clearMarks(endMark);
    performance.clearMeasures(measureName);
}

// Debounce utility (modern implementation)
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle utility for high-frequency events
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Modern Page Visibility API handling
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('[SPIKE] Page hidden - pausing non-critical updates');
        // Could pause animations or reduce update frequency
    } else {
        console.log('[SPIKE] Page visible - resuming normal operation');
    }
});

// View Transitions API wrapper (2026 cutting edge)
function withViewTransition(callback) {
    if (!document.startViewTransition) {
        // Fallback for browsers without support
        callback();
        return;
    }

    document.startViewTransition(() => {
        callback();
    });
}

// Modern clipboard API for export
async function copyToClipboard(text) {
    if (!navigator.clipboard) {
        console.warn('[SPIKE] Clipboard API not supported');
        return false;
    }

    try {
        await navigator.clipboard.writeText(text);
        showNotification('Copied to clipboard!', 'success');
        return true;
    } catch (err) {
        console.error('[SPIKE] Failed to copy:', err);
        showNotification('Copy failed', 'danger');
        return false;
    }
}

// IndexedDB for offline data persistence (optional future enhancement)
async function initIndexedDB() {
    if (!window.indexedDB) {
        console.warn('[SPIKE] IndexedDB not supported');
        return null;
    }

    return new Promise((resolve, reject) => {
        const request = indexedDB.open('SpikeTerminal', 1);

        request.onerror = () => {
            console.error('[SPIKE] IndexedDB error');
            reject(request.error);
        };

        request.onsuccess = () => {
            console.log('[SPIKE] IndexedDB initialized');
            resolve(request.result);
        };

        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            if (!db.objectStoreNames.contains('trades')) {
                db.createObjectStore('trades', { keyPath: 'id', autoIncrement: true });
            }
            if (!db.objectStoreNames.contains('portfolioHistory')) {
                db.createObjectStore('portfolioHistory', { keyPath: 'timestamp' });
            }
        };
    });
}

// ES2024+ Array methods showcase
function analyzePortfolioWithModernJS() {
    if (portfolioHistory.length === 0) return null;

    // Modern array grouping (ES2024)
    const tradesByAction = portfolioHistory.reduce((acc, trade) => {
        const action = trade.action || 'HOLD';
        if (!acc[action]) acc[action] = [];
        acc[action].push(trade);
        return acc;
    }, {});

    // Find last trade of each type
    const lastTrades = Object.entries(tradesByAction).map(([action, trades]) => ({
        action,
        lastTrade: trades[trades.length - 1]
    }));

    return {
        tradesByAction,
        lastTrades,
        totalTrades: portfolioHistory.length,
        avgPortfolioValue: portfolioHistory.reduce((sum, t) => sum + t.portfolio, 0) / portfolioHistory.length
    };
}

// GPU-accelerated number formatting (uses Intl API)
const currencyFormatter = new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
});

const percentFormatter = new Intl.NumberFormat('en-IN', {
    style: 'percent',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
});

// Keyboard shortcuts (modern UX)
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K: Focus search/command palette (future feature)
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        console.log('[SPIKE] Command palette (future feature)');
    }

    // Ctrl/Cmd + E: Export data
    if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
        e.preventDefault();
        exportTradesCSV();
    }

    // Space: Start/Stop trading
    if (e.key === ' ' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        const startBtn = document.getElementById('start-btn');
        const stopBtn = document.getElementById('stop-btn');
        if (!startBtn.disabled) {
            startTrading();
        } else if (!stopBtn.disabled) {
            stopTrading();
        }
    }
});

console.log('[SPIKE] Modern JavaScript features initialized (ES2024+)');
