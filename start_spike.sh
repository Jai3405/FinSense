#!/bin/bash

# SPIKE TERMINAL LAUNCHER
# Enterprise AI Trading System

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

clear

# 1. ASCII Art Banner
echo -e "${GREEN}"
cat << "EOF"
   _____ ____  ____  __ __  ______
  / ___/|  _ \|    ||  |  ||   ___|
 (   \_ | |_| ||  | |  |  ||   ___|
  \__  ||  __/ |  | |  _  ||   ___|
  /  \ ||  |   |  | |  |  ||   ___|
  \    ||  |   |  | |  |  ||   ___|
   \___||__|   |____|__|__||______| 
        TERMINAL v2.0.26
EOF
echo -e "${NC}"

# 2. System Initialization Sequence
echo -e "${BLUE}[SYSTEM]${NC} Initializing SPIKE Trading Environment..."
sleep 0.5

echo -ne "${BLUE}[LOAD]${NC}   Loading Neural Weights (PPO_v2)... "
for i in {1..20}; do echo -ne "▓"; sleep 0.05; done
echo -e " ${GREEN}OK${NC}"

echo -ne "${BLUE}[NET]${NC}    Connecting to Market Data Stream... "
for i in {1..15}; do echo -ne "▓"; sleep 0.05; done
echo -e " ${GREEN}CONNECTED${NC}"

echo -ne "${BLUE}[GPU]${NC}    Allocating Tensor Resources...      "
for i in {1..10}; do echo -ne "▓"; sleep 0.05; done
echo -e " ${GREEN}READY${NC}"

echo -e "\n${YELLOW}>>> LAUNCHING DASHBOARD INTERFACE...${NC}\n"
sleep 1

# 3. Launch Dashboard
# Using the existing start_dashboard.sh or python command
if [ -f "start_dashboard.sh" ]; then
    ./start_dashboard.sh
else
    python3 dashboard/app.py
fi
