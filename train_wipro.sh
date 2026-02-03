#!/bin/bash

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

clear

# ASCII Art Banner
echo -e "${MAGENTA}"
cat << "EOF"
   ██╗    ██╗██╗██████╗ ██████╗  ██████╗
   ██║    ██║██║██╔══██╗██╔══██╗██╔═══██╗
   ██║ █╗ ██║██║██████╔╝██████╔╝██║   ██║
   ██║███╗██║██║██╔═══╝ ██╔══██╗██║   ██║
   ╚███╔███╔╝██║██║     ██║  ██║╚██████╔╝
    ╚══╝╚══╝ ╚═╝╚═╝     ╚═╝  ╚═╝ ╚═════╝
EOF
echo -e "${CYAN}     PPO TRAINING SESSION${NC}"
echo -e "${DIM}   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Training info
echo -e "${BLUE}[CONFIG]${NC}"
echo -e "   ${DIM}Stock:${NC}      ${BOLD}WIPRO.NS${NC}"
echo -e "   ${DIM}Episodes:${NC}   ${BOLD}450${NC}"
echo -e "   ${DIM}Brokerage:${NC}  ${BOLD}Zerodha Intraday (MIS)${NC}"
echo -e "   ${DIM}Algorithm:${NC}  ${BOLD}PPO (Proximal Policy Optimization)${NC}"
echo ""

# Create logs directory
mkdir -p logs

# Timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/wipro_training_${TIMESTAMP}.log"

echo -e "${YELLOW}[SETUP]${NC} ${DIM}Preparing training environment...${NC}"
sleep 0.5

# Activate environment
source finsense_env/bin/activate
echo -e "${GREEN}[OK]${NC}    ${DIM}Python environment activated${NC}"

# Start training with nohup
echo ""
echo -e "${MAGENTA}[TRAIN]${NC} ${BOLD}Launching PPO training in background...${NC}"
echo ""

nohup python3 train_ppo.py --episodes 450 --verbose > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

sleep 2

# Check if process started
if ps -p $TRAIN_PID > /dev/null; then
    echo -e "${GREEN}"
    cat << "EOF"
   ╔══════════════════════════════════════════════╗
   ║                                              ║
   ║      🧠  TRAINING STARTED  🧠                ║
   ║                                              ║
   ╚══════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    echo -e "${BOLD}   PID:${NC}       ${CYAN}$TRAIN_PID${NC}"
    echo -e "${BOLD}   Log File:${NC}  ${CYAN}$LOG_FILE${NC}"
    echo ""
    echo -e "${DIM}   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${YELLOW}   LIVE MONITORING COMMANDS:${NC}"
    echo ""
    echo -e "   ${BOLD}Watch live progress:${NC}"
    echo -e "   ${CYAN}tail -f $LOG_FILE${NC}"
    echo ""
    echo -e "   ${BOLD}Check training status:${NC}"
    echo -e "   ${CYAN}ps aux | grep train_ppo${NC}"
    echo ""
    echo -e "   ${BOLD}Stop training:${NC}"
    echo -e "   ${CYAN}kill $TRAIN_PID${NC}"
    echo ""
    echo -e "${DIM}   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo -e "${GREEN}   Training is running in background. You can close this terminal.${NC}"
    echo ""

    # Save PID to file for later reference
    echo $TRAIN_PID > logs/training.pid
    echo "$LOG_FILE" > logs/training.log_path

    # Ask if user wants to start monitoring
    echo -e "${YELLOW}   Start live monitoring now? (y/n)${NC}"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${CYAN}   Starting live monitor... (Ctrl+C to exit monitor, training continues)${NC}"
        echo ""
        sleep 1
        tail -f "$LOG_FILE"
    fi
else
    echo -e "${RED}[ERROR]${NC} Training failed to start. Check logs for details."
    exit 1
fi
