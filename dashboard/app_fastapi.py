"""
SPIKE Terminal - FastAPI + WebSocket Implementation
Modern, async, production-ready dashboard
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from live_data.streamer import HistoricalDataSimulator
from live_trading.ppo_inference import PPOInference
from live_trading.paper_executor import PaperTradingExecutor
from utils.config import load_config

# FastAPI app
app = FastAPI(title="SPIKE Terminal")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global state
trading_state = {
    'running': False,
    'executor': None,
    'streamer': None,
    'agent': None,
    'current_step': 0,
    'websocket': None
}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve main dashboard."""
    return templates.TemplateResponse("premium_ws.html", {"request": request})


@app.get("/test_websocket", response_class=HTMLResponse)
async def test_websocket(request: Request):
    """Serve test websocket page."""
    return templates.TemplateResponse("test_websocket.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    print("[FASTAPI] WebSocket connected")

    trading_state['websocket'] = websocket

    try:
        while True:
            # Keep connection alive and receive commands
            data = await websocket.receive_json()

            if data.get('command') == 'start':
                print(f"[FASTAPI] Start command received: {data}")
                await start_trading(
                    websocket,
                    data.get('start_date', '2024-01-01'),
                    data.get('end_date', '2024-03-31'),
                    data.get('balance', 50000)
                )
            elif data.get('command') == 'stop':
                print("[FASTAPI] Stop command received")
                trading_state['running'] = False

    except WebSocketDisconnect:
        print("[FASTAPI] WebSocket disconnected")
        trading_state['websocket'] = None
        trading_state['running'] = False


async def start_trading(websocket: WebSocket, start_date: str, end_date: str, balance: float):
    """Start trading simulation."""
    global trading_state

    if trading_state['running']:
        await websocket.send_json({'error': 'Already running'})
        return

    try:
        print(f"[FASTAPI] Initializing trading system...")
        print(f"  Date range: {start_date} to {end_date}")
        print(f"  Balance: ₹{balance}")

        # Load config
        config_path = Path(__file__).parent.parent / 'config.yaml'
        config = load_config(str(config_path))
        print("[FASTAPI] ✓ Config loaded")

        # Initialize streamer
        trading_state['streamer'] = HistoricalDataSimulator(
            ticker='RELIANCE.NS',
            start_date=start_date,
            end_date=end_date,
            interval='1d'
        )
        # PPO agent needs window_size=20, so buffer must start with 20 candles
        trading_state['streamer'].initialize_buffer(window_size=20)

        total_candles = len(trading_state['streamer'].simulation_data['close'])
        current_idx = trading_state['streamer'].current_index
        print(f"[FASTAPI] ✓ Streamer initialized: {total_candles} candles, starting at index {current_idx}")

        # Load PPO agent
        model_path = Path(__file__).parent.parent / 'models' / 'ppo_final.pt'
        trading_state['agent'] = PPOInference(str(model_path), config)
        print("[FASTAPI] ✓ PPO agent loaded")

        # Initialize executor
        trading_state['executor'] = PaperTradingExecutor(
            starting_balance=balance,
            max_positions=40,
            max_position_value=0.95,
            transaction_costs=True
        )
        print("[FASTAPI] ✓ Executor initialized")

        # Start simulation
        trading_state['running'] = True
        trading_state['current_step'] = 0

        await websocket.send_json({'status': 'started', 'message': 'Simulation started'})

        # Run trading loop
        await run_trading_loop(websocket)

    except Exception as e:
        print(f"[FASTAPI] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.send_json({'error': str(e)})


async def run_trading_loop(websocket: WebSocket):
    """Main trading loop - async version."""
    global trading_state

    print("[FASTAPI] ========== TRADING LOOP STARTED ==========")

    while trading_state['running']:
        try:
            # Get new data
            if not trading_state['streamer'].update():
                print("[FASTAPI] End of simulation data")
                trading_state['running'] = False
                await websocket.send_json({'status': 'complete', 'message': 'Simulation complete'})
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

            # Prepare update data
            update_data = {
                'type': 'trading_update',
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
                update_data['trade'] = trade_data

            # Send update
            print(f"[FASTAPI] Step {trading_state['current_step']}: {action_name} @ ₹{current_price:.2f}")
            await websocket.send_json(update_data)

            trading_state['current_step'] += 1

            # Delay for presentation (0.5s)
            await asyncio.sleep(0.5)

        except WebSocketDisconnect:
            print("[FASTAPI] WebSocket disconnected during loop")
            trading_state['running'] = False
            break
        except Exception as e:
            print(f"[FASTAPI] ✗ Error in trading loop: {e}")
            import traceback
            traceback.print_exc()
            trading_state['running'] = False
            try:
                await websocket.send_json({'error': str(e)})
            except:
                pass
            break

    print("[FASTAPI] ========== TRADING LOOP ENDED ==========")


if __name__ == "__main__":
    import uvicorn

    print("="*80)
    print(" SPIKE TERMINAL - FastAPI Edition")
    print("="*80)
    print()
    print(" Starting on http://localhost:8000")
    print()
    print(" Modern async WebSocket implementation")
    print(" Better performance, better debugging")
    print()
    print("="*80)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
