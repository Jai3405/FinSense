#!/bin/bash

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    🚀 SPIKE Platform - Development Server                   ║
# ║              AI Wealth Intelligence Platform - India's First                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ASCII Art
echo -e "${PURPLE}"
cat << "EOF"
   _____ _____ _____ _  _______
  / ____|  __ \_   _| |/ /  ___|
 | (___ | |__) || | | ' /| |__
  \___ \|  ___/ | | |  < |  __|
  ____) | |    _| |_| . \| |___
 |_____/|_|   |_____|_|\_\_____|

  AI Wealth Intelligence Platform
EOF
echo -e "${NC}"

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Starting Development Environment...${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo -e "${RED}Error: Must run from spike-platform directory${NC}"
    exit 1
fi

# Check for pnpm
if ! command -v pnpm &> /dev/null; then
    echo -e "${YELLOW}Installing pnpm...${NC}"
    npm install -g pnpm
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}📦 Installing dependencies...${NC}"
    pnpm install
fi

# Create .env.local if it doesn't exist
if [ ! -f "apps/web/.env.local" ]; then
    echo -e "${YELLOW}Creating .env.local from .env.example...${NC}"
    cp .env.example apps/web/.env.local 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}🌐 Starting services:${NC}"
echo -e "   ${CYAN}• Web App:${NC}     http://localhost:3000"
echo -e "   ${CYAN}• API Server:${NC}  http://localhost:8000"
echo -e "   ${CYAN}• API Docs:${NC}    http://localhost:8000/api/v1/docs"
echo ""

# Run development servers
echo -e "${BLUE}Starting Turbo dev server...${NC}"
pnpm dev
