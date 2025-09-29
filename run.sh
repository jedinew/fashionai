#!/usr/bin/env bash
set -euo pipefail

# Determine project root (directory of this script)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Create venv if missing
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activate venv
# shellcheck source=/dev/null
source venv/bin/activate

# Ensure latest pip/wheel and install deps
python -m pip install -U pip wheel
pip install -r requirements.txt

# Run Streamlit
exec streamlit run app.py
