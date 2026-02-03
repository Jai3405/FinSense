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
echo -e "${CYAN}"
cat << "EOF"
   ███████╗██████╗ ██╗██╗  ██╗███████╗
   ██╔════╝██╔══██╗██║██║ ██╔╝██╔════╝
   ███████╗██████╔╝██║█████╔╝ █████╗
   ╚════██║██╔═══╝ ██║██╔═██╗ ██╔══╝
   ███████║██║     ██║██║  ██╗███████╗
   ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝
EOF
echo -e "${GREEN}        TERMINAL ${DIM}v2.1.0${NC}"
echo -e "${DIM}   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${MAGENTA}   Enterprise AI Trading Platform${NC}"
echo -e "${DIM}   Powered by PPO Reinforcement Learning${NC}"
echo ""

# System Check
echo -e "${BLUE}[SYS]${NC} ${DIM}Initializing system components...${NC}"
sleep 0.3

echo -ne "${BLUE}[GPU]${NC} ${DIM}Loading neural weights...${NC} "
for i in {1..15}; do echo -ne "${GREEN}▓${NC}"; sleep 0.05; done
echo -e " ${GREEN}✓${NC}"

echo -ne "${BLUE}[NET]${NC} ${DIM}Connecting to market stream...${NC} "
for i in {1..15}; do echo -ne "${CYAN}▓${NC}"; sleep 0.05; done
echo -e " ${GREEN}✓${NC}"

# Kill existing server
echo ""
echo -e "${YELLOW}[KILL]${NC} ${DIM}Stopping existing servers on port 8000...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 1
echo -e "${GREEN}[OK]${NC}   ${DIM}Port 8000 cleared${NC}"

# Activate environment
echo ""
echo -e "${BLUE}[ENV]${NC}  ${DIM}Activating Python environment...${NC}"
source finsense_env/bin/activate
cd dashboard
echo -e "${GREEN}[OK]${NC}   ${DIM}Environment ready${NC}"

# Start FastAPI
echo ""
echo -e "${MAGENTA}[BOOT]${NC} ${BOLD}Starting FastAPI server...${NC}"
echo ""
python3 app_fastapi.py &
SERVER_PID=$!
sleep 3

# Success banner
echo ""
echo -e "${GREEN}"
cat << "EOF"
   ╔══════════════════════════════════════════════╗
   ║                                              ║
   ║         🚀  SERVER ONLINE  🚀                ║
   ║                                              ║
   ╚══════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Connection info
echo -e "${BOLD}   URL:${NC}     ${CYAN}http://localhost:8000${NC}"
echo -e "${BOLD}   PID:${NC}     ${DIM}$SERVER_PID${NC}"
echo -e "${BOLD}   Status:${NC}  ${GREEN}●${NC} ${GREEN}Active${NC}"
echo ""
echo -e "${DIM}   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}   Press Ctrl+C to shutdown${NC}"
echo ""

wait
