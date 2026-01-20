"""
FinSense Dashboard - Web Interface for Capstone Presentation

Real-time trading dashboard with your brand colors:
- Primary: #D3E9D7 (mint green)
- Secondary: #638C82 (teal)
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import threading
import time
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from live_data.streamer import HistoricalDataSimulator
from live_trading.ppo_inference import PPOInference
from live_trading.paper_executor import PaperTradingExecutor
from utils.config import load_config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'finsense-dashboard-2026'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
trading_state = {
    'running': False,
    'executor': None,
    'streamer': None,
    'agent': None,
    'current_step': 0,
    'trades': [],
    'equity_curve': [],
    'metrics': {}
}


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('dashboard.html')


@app.route('/api/status')
def get_status():
    """Get current trading status."""
    if trading_state['executor']:
        metrics = trading_state['executor'].get_metrics(
            trading_state['streamer'].get_latest_price()
        )
        return jsonify({
            'running': trading_state['running'],
            'step': trading_state['current_step'],
            'metrics': metrics
        })
    return jsonify({'running': False})


@app.route('/api/start', methods=['POST'])
def start_trading():
    """Start paper trading simulation."""
    global trading_state

    if trading_state['running']:
        return jsonify({'error': 'Already running'}), 400

    # Get parameters
    data = request.json
    start_date = data.get('start_date', '2024-01-01')
    end_date = data.get('end_date', '2024-03-31')
    balance = data.get('balance', 50000)

    try:
        # Load config
        config = load_config('config.yaml')

        # Initialize components
        trading_state['streamer'] = HistoricalDataSimulator(
            ticker='RELIANCE.NS',
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        trading_state['streamer'].initialize_buffer(window_size=20)

        trading_state['agent'] = PPOInference('models/ppo_final.pt', config)

        trading_state['executor'] = PaperTradingExecutor(
            starting_balance=balance,
            max_positions=40,
            max_position_value=0.95,
            transaction_costs=True
        )

        trading_state['running'] = True
        trading_state['current_step'] = 0
        trading_state['trades'] = []
        trading_state['equity_curve'] = []

        # Start trading thread
        thread = threading.Thread(target=run_trading_loop)
        thread.daemon = True
        thread.start()

        return jsonify({'success': True, 'message': 'Trading started'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stop', methods=['POST'])
def stop_trading():
    """Stop paper trading."""
    global trading_state
    trading_state['running'] = False
    return jsonify({'success': True, 'message': 'Trading stopped'})


def run_trading_loop():
    """Main trading loop (runs in background thread)."""
    global trading_state

    while trading_state['running']:
        try:
            # Get new data
            if not trading_state['streamer'].update():
                # End of data
                trading_state['running'] = False
                socketio.emit('trading_complete', {'message': 'Simulation complete'})
                break

            # Get current data
            data = trading_state['streamer'].get_data()
            current_price = trading_state['streamer'].get_latest_price()
            timestamp = trading_state['streamer'].get_latest_timestamp()

            # Get action mask
            action_mask = trading_state['executor'].get_action_mask(current_price)

            # Agent predicts
            action, action_probs, value = trading_state['agent'].predict(data, action_mask=action_mask)
            action_name = trading_state['agent'].get_action_name(action)

            # Execute action
            trade_result = trading_state['executor'].execute_action(action, current_price, timestamp)

            # Update equity curve
            trading_state['executor'].update_equity_curve(timestamp, current_price)

            # Get metrics
            metrics = trading_state['executor'].get_metrics(current_price)

            # Prepare data for frontend
            update_data = {
                'step': trading_state['current_step'],
                'timestamp': timestamp.isoformat(),
                'price': float(current_price),
                'action': action_name,
                'action_probs': [float(p) for p in action_probs],
                'metrics': {
                    'portfolio_value': float(metrics['portfolio_value']),
                    'total_return_pct': float(metrics['total_return_pct']),
                    'sharpe_ratio': float(metrics['sharpe_ratio']),
                    'max_drawdown': float(metrics['max_drawdown']),
                    'win_rate': float(metrics['win_rate']),
                    'total_trades': int(metrics['total_trades']),
                    'balance': float(metrics['current_balance']),
                    'inventory': int(metrics['inventory'])
                }
            }

            # Add trade if BUY/SELL
            if action_name in ['BUY', 'SELL'] and trade_result.get('success'):
                trade_data = {
                    'timestamp': timestamp.isoformat(),
                    'action': action_name,
                    'price': float(current_price),
                    'pnl': float(trade_result.get('pnl', 0)) if action_name == 'SELL' else 0
                }
                trading_state['trades'].append(trade_data)
                update_data['trade'] = trade_data

            # Emit update to all connected clients
            socketio.emit('trading_update', update_data)

            trading_state['current_step'] += 1

            # Slow down for presentation (0.5 seconds per step)
            time.sleep(0.5)

        except Exception as e:
            print(f"Error in trading loop: {e}")
            trading_state['running'] = False
            socketio.emit('error', {'message': str(e)})
            break


if __name__ == '__main__':
    import socket

    # Try to find an available port
    def get_available_port(start_port=5000):
        for port in range(start_port, start_port + 10):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('', port))
                sock.close()
                return port
            except OSError:
                continue
        return start_port

    port = get_available_port(5000)

    print("="*80)
    print(" FINSENSE DASHBOARD - CAPSTONE PRESENTATION")
    print("="*80)
    print()
    print(f" 🚀 Dashboard starting on http://localhost:{port}")
    print()
    print(f" Open this URL in your browser: http://localhost:{port}")
    print(" Press Ctrl+C to stop the server.")
    print()
    print("="*80)

    socketio.run(app, debug=False, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
