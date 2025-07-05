#!/bin/bash
# Fix Virtual Environment Setup Script
# This script fixes the environment to use the correct project virtual environment

echo "🔧 Fixing Virtual Environment Setup..."
echo "Current VIRTUAL_ENV: $VIRTUAL_ENV"

# Get the project root directory (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CORRECT_VENV="$PROJECT_ROOT/.venv"

echo "Project root: $PROJECT_ROOT"
echo "Correct venv path: $CORRECT_VENV"

# Check if the correct virtual environment exists
if [ ! -d "$CORRECT_VENV" ]; then
    echo "❌ Virtual environment not found at $CORRECT_VENV"
    echo "Creating virtual environment with uv..."
    cd "$PROJECT_ROOT"
    uv sync
fi

# Deactivate any existing virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Deactivating current environment: $VIRTUAL_ENV"
    deactivate 2>/dev/null || true
fi

# Activate the correct virtual environment
echo "Activating correct environment: $CORRECT_VENV"
source "$CORRECT_VENV/bin/activate"

# Verify the setup
echo ""
echo "✅ Environment Fixed!"
echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "Python: $(which python)"
echo "Project: $(basename "$PROJECT_ROOT")"

# Test that the platform works
echo ""
echo "🧪 Testing platform import..."
cd "$SCRIPT_DIR"
python -c "
try:
    from src.config.manager import ConfigurationManager
    print('✅ Configuration system: OK')
    
    from src.agents import AgentFactory
    print('✅ Agent system: OK')
    
    from src.services import create_development_services
    print('✅ Service system: OK')
    
    print('')
    print('🎉 Platform is ready to use!')
    print('')
    print('Quick start commands:')
    print('  # Production interface')
    print('  uv run python src/streamlit/launcher.py -e development')
    print('')
    print('  # Debug interface')
    print('  uv run python src/web/launcher.py -e debug')
    
except Exception as e:
    print(f'❌ Import error: {e}')
    print('Try running: uv sync')
"

echo ""
echo "💡 To make this permanent, add to your shell profile:"
echo "   export VIRTUAL_ENV=\"$CORRECT_VENV\""
echo "   export PATH=\"$CORRECT_VENV/bin:\$PATH\""