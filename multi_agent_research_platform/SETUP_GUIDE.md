# Setup Guide - Multi-Agent Research Platform

## 🔧 Environment Setup Issue Fixed

Your setup was pointing to an old repository (`gemini-test1`). This has been resolved with automated scripts.

## 🚀 Quick Fix (Immediate)

Run this command to fix your current session:

```bash
./fix_environment.sh
```

This will:
- ✅ Deactivate the old `gemini-test1` environment
- ✅ Activate the correct project environment
- ✅ Test that all imports work
- ✅ Show you the ready-to-use commands

## 🔄 Permanent Fix (For Future Sessions)

Run this command to update your shell profile:

```bash
./setup_shell.sh
```

This will:
- ✅ Add correct environment variables to your shell profile
- ✅ Create convenient aliases for platform commands
- ✅ Ensure future terminal sessions use the right environment

### After running setup_shell.sh:

Either restart your terminal or run:
```bash
source ~/.bashrc  # or ~/.zshrc depending on your shell
```

## 🎯 Available Commands After Setup

| Command | Description |
|---------|-------------|
| `ma-platform` | Navigate to platform directory |
| `ma-streamlit` | Start Streamlit interface (production) |
| `ma-debug` | Start debug interface (development) |
| `ma-test` | Run test suite |
| `ma-activate` | Manually activate virtual environment |

## 📁 Correct Project Structure

```
/home/kjdrag/lrepos/multi-agent-adk-integration/
├── .venv/                           # ✅ Correct virtual environment
└── multi_agent_research_platform/   # ✅ Current platform code
    ├── src/                         # Platform source code
    ├── .env                         # Environment configuration
    ├── fix_environment.sh           # Immediate fix script
    ├── setup_shell.sh              # Permanent shell setup
    └── SETUP_GUIDE.md              # This guide
```

## 🧪 Verify Setup

After running the fix, test that everything works:

```bash
# Test configuration
python -c "from src.config.manager import ConfigurationManager; print('✅ Config OK')"

# Test agents
python -c "from src.agents import AgentFactory; print('✅ Agents OK')"

# Test services
python -c "from src.services import create_development_services; print('✅ Services OK')"
```

## 🚀 Start Using the Platform

### Production Interface (User-friendly)
```bash
uv run python src/streamlit/launcher.py -e development
# Available at: http://localhost:8501
```

### Debug Interface (Development/Monitoring)
```bash
uv run python src/web/launcher.py -e debug
# Available at: http://localhost:8081
```

## ❓ Troubleshooting

### If you see "No module named..." errors:
```bash
# Ensure dependencies are installed
uv sync

# Check virtual environment
echo $VIRTUAL_ENV
# Should show: /home/kjdrag/lrepos/multi-agent-adk-integration/.venv
```

### If imports still fail:
```bash
# Run the fix script again
./fix_environment.sh
```

### If you need to reset everything:
```bash
# Remove old environment reference and rerun setup
unset VIRTUAL_ENV
./fix_environment.sh
./setup_shell.sh
```

## 📋 Environment Variables

The platform uses these key environment variables:

```bash
# Project paths
MULTI_AGENT_PROJECT_ROOT="/home/kjdrag/lrepos/multi-agent-adk-integration"
VIRTUAL_ENV="/home/kjdrag/lrepos/multi-agent-adk-integration/.venv"

# API configuration (from .env file)
GOOGLE_API_KEY=your_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=false
```

## ✅ Success Indicators

When everything is working correctly, you should see:

1. **Environment**: `echo $VIRTUAL_ENV` shows the correct path
2. **Python**: `which python` points to project's `.venv/bin/python`
3. **Imports**: All platform modules import without errors
4. **Interfaces**: Both Streamlit and Web interfaces start successfully

## 🎉 You're All Set!

Your setup is now fixed and pointing to the correct Multi-Agent Research Platform repository. The old `gemini-test1` references have been completely removed from your environment.