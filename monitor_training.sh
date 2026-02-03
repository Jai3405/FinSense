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

echo -e "${CYAN}"
cat << "EOF"
   ████████╗██████╗  █████╗ ██╗███╗   ██╗██╗███╗   ██╗ ██████╗
   ╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║██║████╗  ██║██╔════╝
      ██║   ██████╔╝███████║██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗
      ██║   ██╔══██╗██╔══██║██║██║╚██╗██║██║██║╚██╗██║██║   ██║
      ██║   ██║  ██║██║  ██║██║██║ ╚████║██║██║ ╚████║╚██████╔╝
      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝
EOF
echo -e "${GREEN}              MONITOR${NC}"
echo -e "${DIM}   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check for active training
if [ -f "logs/training.pid" ]; then
    TRAIN_PID=$(cat logs/training.pid)
    LOG_FILE=$(cat logs/training.log_path 2>/dev/null)

    if ps -p $TRAIN_PID > /dev/null 2>&1; then
        echo -e "${GREEN}[STATUS]${NC} Training is ${GREEN}RUNNING${NC}"
        echo -e "${BOLD}   PID:${NC}      ${CYAN}$TRAIN_PID${NC}"
        echo -e "${BOLD}   Log:${NC}      ${CYAN}$LOG_FILE${NC}"
        echo ""

        # Show recent progress
        echo -e "${BLUE}[RECENT PROGRESS]${NC}"
        echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        if [ -f "$LOG_FILE" ]; then
            # Show last 10 episode results
            grep -E "Episode [0-9]+/[0-9]+" "$LOG_FILE" | tail -10
        fi
        echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""

        # Calculate progress
        if [ -f "$LOG_FILE" ]; then
            CURRENT=$(grep -oE "Episode [0-9]+" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+")
            TOTAL=450
            if [ ! -z "$CURRENT" ]; then
                PERCENT=$((CURRENT * 100 / TOTAL))
                echo -e "${YELLOW}[PROGRESS]${NC} Episode ${BOLD}$CURRENT${NC} / $TOTAL (${PERCENT}%)"

                # Progress bar
                FILLED=$((PERCENT / 5))
                EMPTY=$((20 - FILLED))
                echo -ne "   ["
                for ((i=0; i<FILLED; i++)); do echo -ne "${GREEN}█${NC}"; done
                for ((i=0; i<EMPTY; i++)); do echo -ne "${DIM}░${NC}"; done
                echo -e "]"
            fi
        fi
        echo ""
        echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo -e "${YELLOW}OPTIONS:${NC}"
        echo -e "   ${BOLD}1)${NC} Watch live log"
        echo -e "   ${BOLD}2)${NC} Show best results so far"
        echo -e "   ${BOLD}3)${NC} Stop training"
        echo -e "   ${BOLD}4)${NC} Exit monitor"
        echo ""
        read -p "   Select option (1-4): " choice

        case $choice in
            1)
                echo ""
                echo -e "${CYAN}Live monitoring... (Ctrl+C to exit)${NC}"
                echo ""
                tail -f "$LOG_FILE"
                ;;
            2)
                echo ""
                echo -e "${MAGENTA}[BEST RESULTS]${NC}"
                if [ -f "$LOG_FILE" ]; then
                    echo -e "${DIM}Top 5 profitable episodes:${NC}"
                    grep -E "Profit: ₹[0-9]+" "$LOG_FILE" | sort -t'₹' -k2 -rn | head -5
                fi
                echo ""
                read -p "Press Enter to continue..."
                exec "$0"
                ;;
            3)
                echo ""
                read -p "   Are you sure you want to stop training? (y/n): " confirm
                if [[ "$confirm" =~ ^[Yy]$ ]]; then
                    kill $TRAIN_PID 2>/dev/null
                    echo -e "${RED}[STOPPED]${NC} Training terminated."
                    rm -f logs/training.pid
                fi
                ;;
            4)
                echo -e "${GREEN}Goodbye!${NC}"
                exit 0
                ;;
        esac
    else
        echo -e "${YELLOW}[STATUS]${NC} Training ${YELLOW}COMPLETED${NC} or stopped"
        echo ""

        # Show final results
        if [ -f "$LOG_FILE" ]; then
            echo -e "${MAGENTA}[FINAL RESULTS]${NC}"
            echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            tail -20 "$LOG_FILE"
            echo -e "${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        fi

        rm -f logs/training.pid
    fi
else
    echo -e "${YELLOW}[STATUS]${NC} No active training session found."
    echo ""

    # List recent log files
    echo -e "${BLUE}[RECENT LOGS]${NC}"
    ls -lt logs/wipro_training_*.log 2>/dev/null | head -5
    echo ""
    echo -e "${DIM}To start training, run:${NC}"
    echo -e "${CYAN}./train_wipro.sh${NC}"
fi
