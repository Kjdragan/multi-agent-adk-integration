#!/bin/bash
# Shell Profile Setup Script
# This script adds the correct virtual environment to your shell profile

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CORRECT_VENV="$PROJECT_ROOT/.venv"

echo "🔧 Setting up shell profile for Multi-Agent Research Platform..."

# Detect shell
SHELL_NAME=$(basename "$SHELL")
case "$SHELL_NAME" in
    "bash")
        PROFILE_FILE="$HOME/.bashrc"
        ;;
    "zsh")
        PROFILE_FILE="$HOME/.zshrc"
        ;;
    *)
        PROFILE_FILE="$HOME/.profile"
        ;;
esac

echo "Detected shell: $SHELL_NAME"
echo "Profile file: $PROFILE_FILE"

# Create the environment setup lines
ENV_SETUP="
# Multi-Agent Research Platform Environment
export MULTI_AGENT_PROJECT_ROOT=\"$PROJECT_ROOT\"
export VIRTUAL_ENV=\"$CORRECT_VENV\"
export PATH=\"$CORRECT_VENV/bin:\$PATH\"

# Platform aliases for quick access
alias ma-platform='cd \"$PROJECT_ROOT/multi_agent_research_platform\"'
alias ma-streamlit='uv run python src/streamlit/launcher.py -e development'
alias ma-debug='uv run python src/web/launcher.py -e debug'
alias ma-test='uv run python run_tests.py'
alias ma-activate='source \"$CORRECT_VENV/bin/activate\"'
"

# Check if already configured
if grep -q "Multi-Agent Research Platform Environment" "$PROFILE_FILE" 2>/dev/null; then
    echo "⚠️  Profile already configured. Updating..."
    # Remove old configuration
    sed -i '/# Multi-Agent Research Platform Environment/,/^$/d' "$PROFILE_FILE"
fi

# Add new configuration
echo "$ENV_SETUP" >> "$PROFILE_FILE"

echo "✅ Shell profile updated!"
echo ""
echo "🔄 To apply changes, either:"
echo "   1. Run: source $PROFILE_FILE"
echo "   2. Open a new terminal"
echo ""
echo "🚀 Available commands after setup:"
echo "   ma-platform   # Navigate to platform directory"
echo "   ma-streamlit  # Start Streamlit interface"
echo "   ma-debug      # Start debug interface"
echo "   ma-test       # Run test suite"
echo "   ma-activate   # Activate virtual environment"
echo ""
echo "📁 Project paths:"
echo "   Project root: $PROJECT_ROOT"
echo "   Platform dir: $PROJECT_ROOT/multi_agent_research_platform"
echo "   Virtual env:  $CORRECT_VENV"