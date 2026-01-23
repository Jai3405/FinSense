"""
Debug version of dashboard app with extensive logging
"""
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import threading
import time
from datetime import datetime
import sys
from pathlib import Path
import traceback

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from live_data.streamer import HistoricalDataSimulator
from live_trading.ppo_inference import PPOInference
from live_trading.paper_executor import PaperTradingExecutor
from utils.config import load_config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'finsense-dashboard-2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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
    print("[DEBUG] Serving index page")
    return render_template('premium.html')

@app.route('/test_socket')
def test_socket():
    print("[DEBUG] Serving test socket page")
    return render_template('test_socket.html')

@app.route('/api/status')
def get_status():
    print("[DEBUG] Status check requested")
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
    global trading_state

    print("[DEBUG] ========== START TRADING REQUESTED ==========")

    if trading_state['running']:
        print("[DEBUG] Already running, returning error")
        return jsonify({'error': 'Already running'}), 400

    # Get parameters
    data = request.json
    print(f"[DEBUG] Request data: {data}")

    start_date = data.get('start_date', '2024-01-01')
    end_date = data.get('end_date', '2024-03-31')
    balance = data.get('balance', 50000)

    print(f"[DEBUG] Parameters: start={start_date}, end={end_date}, balance={balance}")

    try:
        # Load config
        print("[DEBUG] Loading config...")
        config_path = Path(__file__).parent.parent / 'config.yaml'
        config = load_config(str(config_path))
        print("[DEBUG] ✓ Config loaded")

        # Initialize streamer
        print("[DEBUG] Initializing streamer...")
        trading_state['streamer'] = HistoricalDataSimulator(
            ticker='RELIANCE.NS',
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        # Use window_size=10 to avoid running out of data
        trading_state['streamer'].initialize_buffer(window_size=10)
        print(f"[DEBUG] ✓ Streamer initialized with {len(trading_state['streamer'].simulation_data['close'])} data points")
        print(f"[DEBUG]   Starting at index {trading_state['streamer'].current_index}")

        # Load agent
        print("[DEBUG] Loading PPO agent...")
        model_path = Path(__file__).parent.parent / 'models' / 'ppo_final.pt'
        trading_state['agent'] = PPOInference(str(model_path), config)
        print("[DEBUG] ✓ Agent loaded")

        # Initialize executor
        print("[DEBUG] Initializing executor...")
        trading_state['executor'] = PaperTradingExecutor(
            starting_balance=balance,
            max_positions=40,
            max_position_value=0.95,
            transaction_costs=True
        )
        print("[DEBUG] ✓ Executor initialized")

        trading_state['running'] = True
        trading_state['current_step'] = 0
        trading_state['trades'] = []
        trading_state['equity_curve'] = []

        # Start trading thread
        print("[DEBUG] Starting trading thread...")
        thread = threading.Thread(target=run_trading_loop)
        thread.daemon = True
        thread.start()
        print("[DEBUG] ✓ Thread started")

        return jsonify({'success': True, 'message': 'Trading started'})

    except Exception as e:
        print(f"[DEBUG] ✗ ERROR in start_trading: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_trading():
    global trading_state
    print("[DEBUG] Stop requested")
    trading_state['running'] = False
    return jsonify({'success': True, 'message': 'Trading stopped'})

def run_trading_loop():
    """Main trading loop with debug logging."""
    global trading_state

    print("[DEBUG] ========== TRADING LOOP STARTED ==========")

    while trading_state['running']:
        try:
            print(f"[DEBUG] Step {trading_state['current_step']}: Getting new data...")

            # Get new data
            if not trading_state['streamer'].update():
                print("[DEBUG] No more data - simulation complete")
                trading_state['running'] = False
                socketio.emit('trading_complete', {'message': 'Simulation complete'})
                break

            # Get current data
            data = trading_state['streamer'].get_data()
            current_price = trading_state['streamer'].get_latest_price()
            timestamp = trading_state['streamer'].get_latest_timestamp()

            print(f"[DEBUG] Price: ₹{current_price:.2f}, Time: {timestamp}")

            # Get action mask
            action_mask = trading_state['executor'].get_action_mask(current_price)
            print(f"[DEBUG] Action mask: {action_mask}")

            # Agent predicts
            action, action_probs, value = trading_state['agent'].predict(data, action_mask=action_mask)
            action_name = trading_state['agent'].get_action_name(action)
            print(f"[DEBUG] Action: {action_name}, Probs: {action_probs}")

            # Execute action
            trade_result = trading_state['executor'].execute_action(action, current_price, timestamp)
            print(f"[DEBUG] Trade result: {trade_result}")

            # Update equity curve
            trading_state['executor'].update_equity_curve(timestamp, current_price)

            # Get metrics
            metrics = trading_state['executor'].get_metrics(current_price)
            print(f"[DEBUG] Portfolio: ₹{metrics['portfolio_value']:.2f}")

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
                    'inventory': int(metrics['inventory']),
                    'inventory_value': float(metrics.get('inventory_value', 0)),
                    'avg_cost': float(metrics.get('avg_cost', 0)),
                    'unrealized_pnl': float(metrics.get('unrealized_pnl', 0)),
                    'total_profit': float(metrics.get('total_profit', 0)),
                    'total_loss': float(metrics.get('total_loss', 0))
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
                print(f"[DEBUG] Trade added: {trade_data}")

            # Emit update to all connected clients
            print(f"[DEBUG] Emitting trading_update...")
            socketio.emit('trading_update', update_data)
            print(f"[DEBUG] ✓ Emitted trading_update for step {trading_state['current_step']}")

            trading_state['current_step'] += 1

            # Slow down for presentation
            time.sleep(0.5)

        except Exception as e:
            print(f"[DEBUG] ✗ ERROR in trading loop: {e}")
            traceback.print_exc()
            trading_state['running'] = False
            socketio.emit('error', {'message': str(e)})
            break

    print("[DEBUG] ========== TRADING LOOP ENDED ==========")

if __name__ == '__main__':
    import socket

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
    print(" SPIKE TERMINAL - DEBUG MODE")
    print("="*80)
    print()
    print(f" Dashboard starting on http://localhost:{port}")
    print()
    print(" Open browser console (F12) to see client-side errors")
    print(" This terminal will show server-side debug logs")
    print()
    print("="*80)
    print()

    socketio.run(app, debug=True, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
