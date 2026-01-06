#!/usr/bin/env bash
set -euo pipefail

# Install Python dependencies for analysis scripts
# Run this on the OKE VM (not in containers)

echo "Installing Python analysis dependencies..."

pip3 install --user pandas matplotlib numpy || {
  echo "ERROR: Failed to install dependencies"
  echo "Try: sudo yum install python3-pip"
  exit 1
}

echo "✓ Dependencies installed successfully"
echo ""
echo "Installed packages:"
pip3 list | grep -E "(pandas|matplotlib|numpy)"
